"""Train VPMM + a projection head (2d -> emb_dim) as an appearance embedding for tracker.type embed.

Per iteration: one LSTCacheDataset clip (clip_len consecutive frames of one video, default 6) ->
VPMM(quads, LSC feats, img_wh) [n,2d] -> EmbedHead -> L2-normalised [n,emb_dim] -> supervised contrastive
loss (SupCon, temperature): anchors are GT-matched instances; positives = same GT track id in another frame
of the clip; every other instance of the clip (other tracks and unmatched detections, track id -1) is a
negative. Only clips with at least one track seen in >= 2 frames are sampled. VPMM may start from an LST
checkpoint (tracker.train.init_ckpt, key "vpmm"). AdamW, linear warm-up + cosine (train_lst.lr_factor),
AMP on CUDA.

Validation every val_every iterations: the val caches are tracked with run_tracker as tracker.type embed
(tracker.lam, emb_gate, emb_momentum) and, once, as tracker.type simple (IoU baseline); both are scored
with tools/evaluate.py --mode track into runs/<exp_id>/val_track/{embed,iou}/. Best = max (IDF1, MOTA).

Writes runs/<exp_id>/{config.yaml, log.txt, metrics.json meta.embed_train, checkpoints/{best,last}.pth}.
Checkpoint = {"vpmm", "head", "c", "d", "emb_dim", "hidden", "iter", "val"}. Settings: tracker.train
(EMBED_TRAIN_DEFAULTS). --smoke: smoke_iters iterations, val every smoke_val_every on the first
smoke_val_videos val videos, exp_id <exp_id>_smoke.

Usage: python -m logo.tracking.train_embed --config configs/E5-embed.yaml [--smoke] [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from logo.data.schema import write_split
from logo.tracking.cache_dataset import CACHE_KEYS, LSTCacheDataset, npz_memmap
from logo.tracking.run_track import save_jsonl
from logo.tracking.simple_tracker import embed_settings, run_tracker
from logo.tracking.train_lst import _evaluate_tool, cache_files, feat_shape, lr_factor
from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed
from logo.utils.paths import project_root
from logo.vpmm.module import VPMM

EMBED_TRAIN_DEFAULTS = dict(train_split="train", val_split="val", iters=3000, lr=1.0e-4, weight_decay=1.0e-4,
                            warmup_iters=100, warmup_factor=0.001, min_lr=0.0, clip_len=6, stride=1,
                            temperature=0.1, emb_dim=128, hidden=256, init_ckpt="", amp=True, log_every=20,
                            val_every=250, smoke_iters=100, smoke_val_every=50, smoke_val_videos=2)
_INT_KEYS = ("iters", "warmup_iters", "clip_len", "stride", "emb_dim", "hidden", "log_every", "val_every",
             "smoke_iters", "smoke_val_every", "smoke_val_videos")
_FLOAT_KEYS = ("lr", "weight_decay", "warmup_factor", "min_lr", "temperature")


def embed_train_settings(cfg: dict, smoke: bool = False) -> dict:
    """tracker.train merged over EMBED_TRAIN_DEFAULTS; smoke replaces iters / val_every."""
    tc = dict(EMBED_TRAIN_DEFAULTS, **(((cfg.get("tracker") or {}).get("train")) or {}))
    for k in _INT_KEYS:
        tc[k] = int(tc[k])
    for k in _FLOAT_KEYS:
        tc[k] = float(tc[k])
    tc["amp"] = bool(tc["amp"])
    tc["init_ckpt"] = str(tc["init_ckpt"] or "")
    if smoke:
        tc["iters"], tc["val_every"] = tc["smoke_iters"], tc["smoke_val_every"]
        tc["warmup_iters"] = min(tc["warmup_iters"], tc["iters"] // 10)
    if tc["iters"] < 1 or tc["val_every"] < 1 or tc["log_every"] < 1:
        raise ValueError("tracker.train: iters, val_every and log_every must be >= 1")
    if tc["clip_len"] < 2 or tc["temperature"] <= 0 or tc["emb_dim"] < 1:
        raise ValueError("tracker.train: need clip_len >= 2, temperature > 0, emb_dim >= 1")
    return tc


class EmbedHead(nn.Module):
    """VPMM output [N,2d] -> L2-normalised [N,emb_dim] (Linear-ReLU-Linear)."""

    def __init__(self, in_dim: int, emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x).float(), dim=1)


class EmbedNet(nn.Module):
    def __init__(self, c: int, d: int, emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.c, self.d, self.emb_dim, self.hidden = int(c), int(d), int(emb_dim), int(hidden)
        self.vpmm = VPMM(c, d)
        self.head = EmbedHead(2 * d, emb_dim, hidden)

    def forward(self, quads, feats: torch.Tensor, img_wh) -> torch.Tensor:
        return self.head(self.vpmm(quads, feats, img_wh))

    def checkpoint(self, **extra) -> dict:
        return {"vpmm": self.vpmm.state_dict(), "head": self.head.state_dict(), "c": self.c, "d": self.d,
                "emb_dim": self.emb_dim, "hidden": self.hidden, **extra}


def load_embed_net(path, device="cpu") -> EmbedNet:
    ck = torch.load(Path(path), map_location="cpu")
    net = EmbedNet(ck["c"], ck["d"], ck["emb_dim"], ck["hidden"])
    net.vpmm.load_state_dict(ck["vpmm"])
    net.head.load_state_dict(ck["head"])
    return net.to(device).eval()


def supcon_loss(emb: torch.Tensor, labels, frames, temperature: float):
    """SupCon over one clip; positives = same label in a different frame. None if no anchor has one."""
    lab = torch.as_tensor(np.asarray(labels), device=emb.device)
    fr = torch.as_tensor(np.asarray(frames), device=emb.device)
    pos = (lab[:, None] == lab[None, :]) & (fr[:, None] != fr[None, :])
    anchors = pos.any(dim=1)
    if not bool(anchors.any()):
        return None
    e = emb.float()
    eye = torch.eye(len(lab), dtype=torch.bool, device=emb.device)
    logits = (e @ e.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = log_prob.masked_fill(~pos, 0.0).sum(dim=1)[anchors] / pos.sum(dim=1)[anchors]
    return -per_anchor.mean()


def clip_labels(clip: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(labels, frame index) per clip instance; unmatched detections (-1) get unique negative labels."""
    ids = np.concatenate([f["track_ids"] for f in clip]).astype(np.int64)
    frames = np.concatenate([np.full(len(f["track_ids"]), k, dtype=np.int64) for k, f in enumerate(clip)])
    labels = np.where(ids >= 0, ids, -1 - np.arange(len(ids), dtype=np.int64))
    return labels, frames


def positive_clips(ds: LSTCacheDataset) -> list[int]:
    """Indices of clips holding a GT track id seen in >= 2 different frames."""
    out = []
    for i, (v, start) in enumerate(ds.index):
        m = ds._meta[v]
        off = m["frame_offsets"]
        lo, hi = int(off[start]), int(off[start + ds.clip_len])
        ids, fr = m["track_ids"][lo:hi], m["frame_ids"][lo:hi]
        keep = ids >= 0
        if keep.sum() < 2:
            continue
        pairs = np.unique(np.stack([ids[keep], fr[keep].astype(np.int64)], axis=1), axis=0)
        if (np.unique(pairs[:, 0], return_counts=True)[1] >= 2).any():
            out.append(i)
    return out


@torch.no_grad()
def embed_rows(net: EmbedNet, quads, feats, rows, img_wh, device, use_amp: bool, batch: int = 256) -> np.ndarray:
    """float32 [N,emb_dim] for quads [N,8] whose LSC features are feats[rows] (feats may be a memmap)."""
    quads = np.asarray(quads, dtype=np.float32).reshape(-1, 8)
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros((len(rows), net.emb_dim), dtype=np.float32)
    if len(rows) and (rows.min() < 0 or rows.max() >= len(feats)):
        raise IndexError(f"feature rows outside 0..{len(feats) - 1}")
    for b in range(0, len(rows), batch):
        r = rows[b:b + batch]
        f = torch.from_numpy(np.asarray(feats[r], dtype=np.float16)).to(device)
        q = torch.from_numpy(quads[b:b + batch]).view(-1, 4, 2).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out[b:b + batch] = net(q, f, img_wh).float().cpu().numpy()
    return out


def attach_embeddings(fused: dict, feats, net: EmbedNet, img_wh, device, use_amp: bool) -> int:
    """Adds an in-memory "emb" (float32 [emb_dim]) to every fused detection; feats row = feat_idx."""
    objs = [o for k in sorted(fused["frames"], key=int) for o in fused["frames"][k]]
    if not objs:
        return 0
    embs = embed_rows(net, [o["quad"] for o in objs], feats, [int(o["feat_idx"]) for o in objs], img_wh,
                      device, use_amp)
    for o, e in zip(objs, embs):
        o["emb"] = e
    return len(objs)


def cache_to_fused(path: Path, net: EmbedNet, device, use_amp: bool) -> dict:
    """Cache npz -> fused-like dict (score_fused, empty text) with "emb" on every detection."""
    with np.load(path) as z:
        m = {k: z[k] for k in CACHE_KEYS if k != "feats"}
    wh = (int(m["img_wh"][0]), int(m["img_wh"][1]))
    embs = embed_rows(net, m["quads"], npz_memmap(path, "feats"), np.arange(len(m["quads"])), wh, device, use_amp)
    off = m["frame_offsets"]
    frames = {}
    for k, f in enumerate(m["frames"]):
        frames[str(int(f))] = [{"quad": m["quads"][j], "score_fused": float(m["score_fused"][j]), "text": "",
                                "text_conf": 0.0, "emb": embs[j]} for j in range(int(off[k]), int(off[k + 1]))]
    return {"video": path.stem, "dataset": "", "frames": frames}


def _evaluate(out_dir: Path, files: list[Path], ds: str, iou: float) -> dict:
    split_txt = out_dir / f"{ds}_val.txt"
    write_split(split_txt, [p.stem for p in files])
    out = out_dir / "metrics.json"
    _evaluate_tool().main(["--gt_dir", str(project_root() / "data" / "processed" / ds), "--pred_dir",
                           str(out_dir / ds), "--videos", str(split_txt), "--mode", "track", "--out", str(out),
                           "--iou", str(iou)])
    return dict(load_json(out)["track"])


def validate(net: EmbedNet, files: list[Path], ds: str, out_dir: Path, tracker: dict, device, use_amp: bool,
             iou: float, baseline: bool = False) -> dict:
    """{"embed": track metrics[, "iou": track metrics of tracker.type simple on the same input]}."""
    net.eval()
    cfgs = {"embed": dict(tracker, type="embed")}
    if baseline:
        cfgs["iou"] = dict(tracker, type="simple")
    for p in files:
        fused = cache_to_fused(p, net, device, use_amp)
        for name, tcfg in cfgs.items():
            save_jsonl(run_tracker(fused, {"tracker": tcfg}), out_dir / name / ds / f"{p.stem}.jsonl")
    net.train()
    return {name: _evaluate(out_dir / name, files, ds, iou) for name in cfgs}


def _write_metrics(run_dir: Path, result: dict) -> None:
    path = run_dir / "metrics.json"
    metrics = {"det": {}, "track": {}, "spot": {}, "per_video": {}, "meta": {}}
    if path.is_file():
        metrics.update(load_json(path))
    metrics.setdefault("meta", {})["embed_train"] = result
    save_json(metrics, path)


def train(cfg: dict, smoke: bool = False, device=None) -> dict:
    """Runs the training described by `cfg`; returns meta.embed_train."""
    root = project_root()
    tc = embed_train_settings(cfg, smoke)
    seed, ds = int(cfg["seed"]), cfg["dataset"]
    exp_id = f"{cfg['exp_id']}_smoke" if smoke else str(cfg["exp_id"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = tc["amp"] and device.type == "cuda"
    iou = float((cfg.get("eval") or {}).get("iou", 0.5))
    tracker = {k: v for k, v in (cfg.get("tracker") or {}).items() if k != "train"}
    es = embed_settings(tracker)

    set_seed(seed)
    run_dir = root / "runs" / exp_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train_embed", run_dir / "log.txt")
    dump = copy.deepcopy(cfg)
    dump["exp_id"] = exp_id
    dump.setdefault("tracker", {})["train"] = dict(tc, smoke=bool(smoke))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dump, sort_keys=False), encoding="utf-8")

    train_files = cache_files(ds, tc["train_split"])
    val_files = cache_files(ds, tc["val_split"])
    if smoke:
        val_files = val_files[:tc["smoke_val_videos"]]
    c, d = feat_shape(train_files[0])
    net = EmbedNet(c, d, tc["emb_dim"], tc["hidden"]).to(device)
    if tc["init_ckpt"]:
        p = Path(tc["init_ckpt"])
        net.vpmm.load_state_dict(torch.load(p if p.is_absolute() else root / p, map_location="cpu")["vpmm"])
        log.info("VPMM initialised from %s", tc["init_ckpt"])
    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, tc))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ds_train = LSTCacheDataset(train_files, clip_len=tc["clip_len"], stride=tc["stride"])
    usable = positive_clips(ds_train)
    if not usable:
        raise ValueError(f"no {tc['clip_len']}-frame clip in the {ds}/{tc['train_split']} cache has a GT track "
                         f"in >= 2 frames; rebuild the cache from a working detector")
    result = {"exp_id": exp_id, "smoke": bool(smoke), "device": str(device), "amp": use_amp, "c": c, "d": d,
              "settings": tc, "tracker": dict(tracker, **es), "num_train_videos": len(train_files),
              "num_val_videos": len(val_files), "clips": len(ds_train), "positive_clips": len(usable),
              "num_params": sum(p.numel() for p in net.parameters())}
    log.info("exp=%s smoke=%s device=%s amp=%s feats=[%d,%d] train=%d videos, %d/%d clips with positives, "
             "val=%d videos, params=%d, iters=%d, lam=%.2f", exp_id, smoke, device, use_amp, c, d,
             len(train_files), len(usable), len(ds_train), len(val_files), result["num_params"], tc["iters"],
             es["lam"])

    def save(path: Path, val) -> None:
        tmp = path.with_name(path.name + ".tmp")
        torch.save(net.checkpoint(iter=it, val=val), tmp)
        tmp.replace(path)

    net.train()
    it, epoch, best, history, baseline = 0, 0, None, [], None
    t0, loss_sum, n_win = time.time(), 0.0, 0
    while it < tc["iters"]:
        for i in np.random.default_rng(seed + epoch).permutation(usable).tolist():
            clip = ds_train[i]
            labels, frames = clip_labels(clip)
            quads = torch.from_numpy(np.concatenate([f["quads"].reshape(-1, 8) for f in clip])).view(-1, 4, 2)
            feats = torch.cat([f["feats"] for f in clip]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                emb = net(quads.to(device), feats, clip[0]["img_wh"])
            loss = supcon_loss(emb, labels, frames, tc["temperature"])
            if loss is None:  # cannot happen for positive_clips, kept as a guard
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            loss_sum, n_win = loss_sum + float(loss), n_win + 1
            if it % tc["log_every"] == 0 or it == tc["iters"]:
                log.info("iter %d/%d loss %.4f lr %.2e epoch %d %.1fs", it, tc["iters"], loss_sum / n_win,
                         opt.param_groups[0]["lr"], epoch, time.time() - t0)
                loss_sum, n_win = 0.0, 0
            if it % tc["val_every"] == 0 or it == tc["iters"]:
                res = validate(net, val_files, ds, run_dir / "val_track", tracker, device, use_amp, iou,
                               baseline=baseline is None)
                baseline = res.get("iou", baseline)
                val = res["embed"]
                rec = {"iter": it, **val, "sec": round(time.time() - t0, 1)}
                history.append(rec)
                is_best = best is None or (val["IDF1"], val["MOTA"]) > (best["IDF1"], best["MOTA"])
                if is_best:
                    best = rec
                    save(ckpt_dir / "best.pth", val)
                save(ckpt_dir / "last.pth", val)
                log.info("val iter %d: embed IDF1 %.2f MOTA %.2f | iou IDF1 %.2f MOTA %.2f%s", it, val["IDF1"],
                         val["MOTA"], baseline["IDF1"], baseline["MOTA"], " *best*" if is_best else "")
                _write_metrics(run_dir, dict(result, iters_done=it, best=best, baseline_iou=baseline,
                                             delta_IDF1=round(best["IDF1"] - baseline["IDF1"], 4),
                                             best_ckpt=str(ckpt_dir / "best.pth"), history=history))
            if it >= tc["iters"]:
                break
        epoch += 1
    final = dict(result, iters_done=it, best=best, baseline_iou=baseline,
                 delta_IDF1=round(best["IDF1"] - baseline["IDF1"], 4), best_ckpt=str(ckpt_dir / "best.pth"),
                 history=history)
    _write_metrics(run_dir, final)
    log.info("done: %d iters, best iter %d IDF1 %.2f vs IoU tracker %.2f (delta %+.2f)", it, best["iter"],
             best["IDF1"], baseline["IDF1"], final["delta_IDF1"])
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/E5-embed.yaml"))
    ap.add_argument("--smoke", action="store_true", help="smoke_iters iterations, exp_id <exp_id>_smoke")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args(argv)
    train(load_yaml(args.config), smoke=args.smoke, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
