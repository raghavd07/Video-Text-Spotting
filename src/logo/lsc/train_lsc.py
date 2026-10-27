"""Fine-tune the pretrained ABINet (LSC recogniser) on LSCDataset (data/lsc/{train,val}).

Reuses third_party/ABINet unmodified: model = demo.get_model(train_abinet.yaml config) loaded from
lsc.ckpt; loss = losses.MultiLosses(one_hot=True), i.e. vision + language + alignment cross-entropy
over the 3 iterative-refinement outputs; labels = utils.CharsetMapper.get_labels + utils.onehot.
A negative (label "") encodes as [end, pad, pad, ...] with length 1, so only the end token at step 0
is supervised (MultiLosses ignores steps >= length).

Schedule: Adam(lr) for epochs [0, decay_epoch), lr_final afterwards; AMP on CUDA; gradient
accumulation (accum_steps); grad-norm clipping; on CUDA OOM the micro-batch is halved and the
batch retried (effective batch unchanged). Validation uses the alignment head: text/background
accuracy (background = first decoded token is the end token; GT background = label "") and word
accuracy on positives. Best checkpoint = max (tb_acc, word_acc).

Writes runs/<exp_id>/{config.yaml, log.txt, metrics.json (meta.lsc_train),
checkpoints/epoch_XX.pth, checkpoints/best.pth}. Checkpoints are {"model": state_dict, "opt": None}
and load directly with ABINetRecognizer(ckpt). --smoke: 200 train / 200 val samples, 1 epoch,
exp_id <exp_id>_smoke.

Usage: python -m logo.lsc.train_lsc --config configs/E3-lsc.yaml [--smoke] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from logo.lsc.abinet_wrapper import ABINET_ROOT, DEFAULT_CKPT, _import_abinet, decode, make_config
from logo.lsc.dataset import LSCDataset
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import project_root

TRAIN_DEFAULTS = dict(data_root="data/lsc", epochs=10, batch_size=48, val_batch_size=64, lr=1e-4,
                      lr_final=1e-5, decay_epoch=6, accum_steps=1, amp=True, clip_grad=20.0,
                      num_workers=2, log_every=50, smoke_samples=200)


def _import_losses():
    """ABINet's losses.py (needs the fastai shim and sys.path set up by _import_abinet)."""
    _import_abinet()
    m = sys.modules.get("losses")
    f = getattr(m, "__file__", None)
    if m is not None and (f is None or ABINET_ROOT not in Path(f).resolve().parents):
        raise ImportError(f"a different top-level module 'losses' is already imported ({f})")
    import losses
    return losses


def train_settings(cfg: dict) -> dict:
    """lsc.train merged over TRAIN_DEFAULTS."""
    tc = dict(TRAIN_DEFAULTS, **((cfg.get("lsc") or {}).get("train") or {}))
    for k in ("epochs", "batch_size", "val_batch_size", "decay_epoch", "accum_steps", "num_workers",
              "log_every", "smoke_samples"):
        tc[k] = int(tc[k])
    if tc["batch_size"] < 1 or tc["accum_steps"] < 1 or tc["epochs"] < 1:
        raise ValueError("lsc.train: batch_size, accum_steps and epochs must be >= 1")
    return tc


def lr_at(epoch: int, tc: dict) -> float:
    """0-based epoch -> learning rate (step decay at decay_epoch)."""
    return float(tc["lr"] if epoch < tc["decay_epoch"] else tc["lr_final"])


def build_model(ckpt, device):
    """(model, abinet_config, charset) with weights from `ckpt` (repo or our {"model","opt"} format)."""
    ab = _import_abinet()
    ckpt = Path(DEFAULT_CKPT if ckpt is None else ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"ABINet checkpoint not found: {ckpt}")
    config = make_config(ckpt)
    config.global_phase = "train"
    model = ab["demo"].get_model(config)
    model = ab["demo"].load(model, str(ckpt), device="cpu")
    charset = ab["utils"].CharsetMapper(filename=config.dataset_charset_path,
                                        max_length=config.dataset_max_length + 1)
    return model.to(device), config, charset


def encode_labels(texts, charset) -> tuple[torch.Tensor, torch.Tensor]:
    """Texts -> (one-hot float [N,T,C], lengths long [N]) as ABINet's ImageDataset builds them.

    length = len(text) + 1 (end token); "" -> step 0 is the end token, then padding (also null).
    """
    onehot = _import_abinet()["utils"].onehot
    T = charset.max_length
    known = set(charset.char_to_label) - {charset.null_char}
    labels, lengths = [], []
    for t in texts:
        if len(t) > T - 1:
            raise ValueError(f"label {t!r} longer than max_length {T - 1}")
        if set(t) - known:
            raise ValueError(f"label {t!r} has characters outside the charset: {sorted(set(t) - known)}")
        labels.append(charset.get_labels(t, case_sensitive=False))
        lengths.append(len(t) + 1)
    lab = torch.tensor(labels, dtype=torch.long).view(len(labels), T)
    return onehot(lab, charset.num_classes), torch.tensor(lengths, dtype=torch.long)


def score_predictions(preds: list[str], labels: list[str]) -> dict:
    """Decoded texts vs GT labels ("" = background on both sides). Values in percent."""
    if len(preds) != len(labels):
        raise ValueError("preds and labels differ in length")
    n = len(labels)
    n_pos = sum(1 for l in labels if l)
    n_neg = n - n_pos
    tb_ok = sum((p == "") == (l == "") for p, l in zip(preds, labels))
    word_ok = sum(1 for p, l in zip(preds, labels) if l and p == l)
    text_ok = sum(1 for p, l in zip(preds, labels) if l and p)
    bg_ok = sum(1 for p, l in zip(preds, labels) if not l and not p)
    pct = lambda a, b: 100.0 * a / b if b else 0.0  # noqa: E731
    return {"num_samples": n, "num_pos": n_pos, "num_neg": n_neg, "tb_acc": pct(tb_ok, n),
            "word_acc": pct(word_ok, n_pos), "text_recall": pct(text_ok, n_pos),
            "bg_recall": pct(bg_ok, n_neg)}


def _as_float(o):
    """Copy of model outputs with fp32 logits (loss computed outside autocast)."""
    if isinstance(o, dict):
        return {k: (v.float() if k == "logits" else v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_as_float(x) for x in o)
    return o


def _backward_batch(model, crit, x, gt, lengths, micro, scale, scaler, use_amp):
    """Forward + backward one loader batch in chunks of `micro`; returns (loss, {name: loss})."""
    n = x.shape[0]
    total, parts = 0.0, Counter()
    for s in range(0, n, micro):
        e = min(n, s + micro)
        w = (e - s) / n
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x[s:e])
        loss = crit(_as_float(out), gt[s:e], lengths[s:e])
        scaler.scale(loss * (w * scale)).backward()
        total += loss.item() * w
        for k, v in crit.last_losses.items():
            parts[k] += v.item() * w
    return total, dict(parts)


@torch.no_grad()
def evaluate(model, crit, loader, charset, device, use_amp) -> dict:
    model.eval()
    preds, labels = [], []
    loss_sum, n = 0.0, 0
    for x, texts in loader:
        texts = list(texts)
        gt, lengths = encode_labels(texts, charset)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x.to(device, non_blocking=True))
        out = _as_float(out)
        loss_sum += crit(out, gt.to(device), lengths.to(device)).item() * len(texts)
        n += len(texts)
        probs = torch.softmax(out[0]["logits"], dim=-1).cpu().numpy()
        preds += [t for t, _, _ in decode(probs, charset)]  # "" <=> first token is the end token
        labels += texts
    res = score_predictions(preds, labels)
    res["loss"] = loss_sum / n if n else 0.0
    return res


def _subset(ds, k: int, seed: int):
    if k >= len(ds):
        return ds
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), size=k, replace=False))
    return Subset(ds, idx.tolist())


def _loader(ds, bs, shuffle, workers, seed, pin):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=pin,
                      generator=g, persistent_workers=workers > 0, drop_last=False)


def _write_metrics(run_dir: Path, result: dict) -> None:
    path = run_dir / "metrics.json"
    metrics = {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}
    if path.is_file():
        metrics.update(load_json(path))
    metrics.setdefault("meta", {})["lsc_train"] = result
    save_json(metrics, path)


def train(cfg: dict, smoke: bool = False, device=None) -> dict:
    """Runs the fine-tuning described by `cfg`; returns meta.lsc_train."""
    root = project_root()
    tc = train_settings(cfg)
    seed = int(cfg["seed"])
    exp_id = f"{cfg['exp_id']}_smoke" if smoke else str(cfg["exp_id"])
    epochs = 1 if smoke else tc["epochs"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tc["amp"]) and device.type == "cuda"

    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    run_dir = root / "runs" / exp_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train_lsc", run_dir / "log.txt")
    data_root = Path(tc["data_root"])
    data_root = data_root if data_root.is_absolute() else root / data_root
    init_ckpt = (cfg.get("lsc") or {}).get("ckpt")
    init_ckpt = Path(DEFAULT_CKPT if init_ckpt is None else init_ckpt)
    init_ckpt = init_ckpt if init_ckpt.is_absolute() else root / init_ckpt

    dump = copy.deepcopy(cfg)
    dump["exp_id"] = exp_id
    dump.setdefault("lsc", {})["train"] = dict(tc, smoke=bool(smoke))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")

    model, config, charset = build_model(init_ckpt, device)
    if getattr(model, "iter_size", 1) not in (1, 3):  # MultiLosses repeats GT exactly 3 times
        raise ValueError(f"MultiLosses supports iter_size 1 or 3, got {model.iter_size}")
    max_len = int(config.dataset_max_length)
    out_hw = (int(config.dataset_image_height), int(config.dataset_image_width))
    train_ds = LSCDataset(data_root / "train", train=True, out_hw=out_hw, max_length=max_len)
    val_ds = LSCDataset(data_root / "val", train=False, out_hw=out_hw, max_length=max_len)
    if smoke:
        train_ds = _subset(train_ds, tc["smoke_samples"], seed)
        val_ds = _subset(val_ds, tc["smoke_samples"], seed + 1)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"empty LSC split under {data_root} (train={len(train_ds)}, val={len(val_ds)})")
    pin = device.type == "cuda"
    train_dl = _loader(train_ds, tc["batch_size"], True, tc["num_workers"], seed, pin)
    val_dl = _loader(val_ds, tc["val_batch_size"], False, tc["num_workers"], seed, pin)

    crit = _import_losses().MultiLosses(one_hot=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr_at(0, tc), betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum, micro = tc["accum_steps"], tc["batch_size"]
    log.info("exp=%s smoke=%s device=%s amp=%s init=%s train=%d val=%d bs=%d accum=%d epochs=%d",
             exp_id, smoke, device, use_amp, init_ckpt, len(train_ds), len(val_ds), tc["batch_size"],
             accum, epochs)

    result = {"exp_id": exp_id, "smoke": bool(smoke), "init_ckpt": str(init_ckpt),
              "data_root": str(data_root), "num_train": len(train_ds), "num_val": len(val_ds),
              "settings": tc, "epochs": [], "best": None, "best_ckpt": None}
    best_key = None
    step = 0
    for ep in range(epochs):
        lr = lr_at(ep, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, win, seen = time.time(), 0, 0
        loss_sum, part_sum = 0.0, Counter()
        for it, (x, texts) in enumerate(train_dl):
            gt, lengths = encode_labels(list(texts), charset)
            x, gt, lengths = x.to(device, non_blocking=True), gt.to(device), lengths.to(device)
            while True:
                try:
                    loss, parts = _backward_batch(model, crit, x, gt, lengths, micro, 1.0 / accum,
                                                  scaler, use_amp)
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)  # drop the partial accumulation window
                    win = 0
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
                    torch.cuda.empty_cache()
                    log.warning("CUDA OOM: micro-batch -> %d (effective batch unchanged)", micro)
            win += 1
            n = x.shape[0]
            seen += n
            loss_sum += loss * n
            for k, v in parts.items():
                part_sum[k] += v * n
            if win == accum or it == len(train_dl) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, float(tc["clip_grad"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                win = 0
                step += 1
                if step % tc["log_every"] == 0:
                    log.info("ep %d step %d loss %.4f lr %.1e", ep + 1, step, loss, lr)
        train_loss = loss_sum / max(seen, 1)
        val = evaluate(model, crit, val_dl, charset, device, use_amp)
        ck = ckpt_dir / f"epoch_{ep + 1:02d}.pth"
        torch.save({"model": model.state_dict(), "opt": None}, ck)
        rec = {"epoch": ep + 1, "lr": lr, "steps": step, "micro_batch": micro,
               "train_loss": train_loss, "train_losses": {k: v / max(seen, 1) for k, v in part_sum.items()},
               "val": val, "ckpt": str(ck), "sec": time.time() - t0}
        key = (val["tb_acc"], val["word_acc"])
        if best_key is None or key > best_key:
            best_key = key
            torch.save({"model": model.state_dict(), "opt": None}, ckpt_dir / "best.pth")
            result["best"] = {"epoch": ep + 1, **val}
            result["best_ckpt"] = str(ckpt_dir / "best.pth")
        result["epochs"].append(rec)
        _write_metrics(run_dir, result)
        log.info("ep %d/%d lr %.1e train_loss %.4f | val loss %.4f tb_acc %.2f%% word_acc %.2f%% "
                 "(pos %d, neg %d) %.0fs%s", ep + 1, epochs, lr, train_loss, val["loss"], val["tb_acc"],
                 val["word_acc"], val["num_pos"], val["num_neg"], rec["sec"],
                 " *best*" if result["best"]["epoch"] == ep + 1 else "")
    log.info("best epoch %d: tb_acc %.2f%% word_acc %.2f%% -> %s", result["best"]["epoch"],
             result["best"]["tb_acc"], result["best"]["word_acc"], result["best_ckpt"])
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/E3-lsc.yaml"))
    ap.add_argument("--smoke", action="store_true", help="200 train/val samples, 1 epoch")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args(argv)
    train(load_yaml(args.config), smoke=args.smoke, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
