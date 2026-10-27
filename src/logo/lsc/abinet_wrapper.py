"""ABINet (FangShancheng/ABINet, PyTorch) as the LSC recogniser.

The official repo lives in third_party/ABINet and is imported as top-level modules (`demo`, `utils`,
`modules.*`). `_import_abinet` is the ONLY place that touches sys.path for it. The repo star-imports
`fastai.vision` (fastai 1.0.60, incompatible with torch 2.x) but only uses `ifnone` and re-exported
torch names from it, so a minimal in-memory `fastai.vision` module is registered when fastai is absent.

Model: configs/train_abinet.yaml (ABINetIterModel, 3 iterations), input 32x128 RGB, ImageNet
normalisation, charset_36 (lowercase + digits), max_length 25 + 1 end token.
  c = dataset_max_length + 1 = 26 (reading positions), d = model_vision_d_model (default 512).
Outputs per crop: text / text_conf / p_eos from the final (alignment) head, feat = vision-model
position-attention output attn_vecs[c, d] (float16).
"""
from __future__ import annotations

import contextlib
import math
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from logo.lsc.roi_rotate import _as_bgr_uint8

ABINET_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "ABINet"
ABINET_CONFIG = "configs/train_abinet.yaml"  # relative to ABINET_ROOT
DEFAULT_CKPT = ABINET_ROOT / "workdir" / "train-abinet" / "best-train-abinet.pth"
MEAN = (0.485, 0.456, 0.406)  # demo.preprocess
STD = (0.229, 0.224, 0.225)

_abinet: dict = {}


def _install_fastai_shim() -> None:
    try:
        import fastai.vision  # noqa: F401  real fastai v1 installed -> use it
        return
    except ImportError:
        pass
    import copy
    import logging
    import warnings
    from typing import Optional

    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    def ifnone(a, b):
        return b if a is None else a

    names = dict(ifnone=ifnone, torch=torch, nn=nn, F=F, np=np, math=math, Tensor=Tensor,
                 Optional=Optional, copy=copy, warnings=warnings, logging=logging)
    fv = types.ModuleType("fastai.vision")
    fv.__dict__.update(names)
    fv.__all__ = list(names)
    pkg = types.ModuleType("fastai")
    pkg.__path__ = []
    pkg.vision = fv
    sys.modules["fastai"], sys.modules["fastai.vision"] = pkg, fv


def _import_abinet() -> dict:
    """Import ABINet's demo/utils modules once; returns {"demo", "utils"}."""
    if _abinet:
        return _abinet
    if not (ABINET_ROOT / "demo.py").is_file():
        raise FileNotFoundError(f"ABINet not found at {ABINET_ROOT}; clone "
                                "https://github.com/FangShancheng/ABINet there")
    for name in ("utils", "modules", "demo"):
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None) or next(iter(getattr(m, "__path__", [])), None)
        if m is not None and (f is None or ABINET_ROOT not in Path(f).resolve().parents):
            raise ImportError(f"a different top-level module {name!r} is already imported ({f}); "
                              "it clashes with third_party/ABINet")
    _install_fastai_shim()
    root = str(ABINET_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)  # the single sys.path insert for third_party/ABINet
    import demo
    import utils
    _abinet.update(demo=demo, utils=utils)
    return _abinet


def make_config(ckpt=None):
    """ABINet Config for inference, as demo.main builds it (paths made absolute)."""
    utils = _import_abinet()["utils"]
    with contextlib.chdir(ABINET_ROOT):  # Config reads configs/template.yaml relative to cwd
        config = utils.Config(ABINET_CONFIG)
    config.dataset_charset_path = str(ABINET_ROOT / config.dataset_charset_path)
    config.model_checkpoint = None if ckpt is None else str(ckpt)
    config.model_eval = "alignment"
    config.global_phase = "test"
    config.model_vision_checkpoint, config.model_language_checkpoint = None, None
    return config


def decode(probs: np.ndarray, charset) -> list[tuple[str, float, float]]:
    """Greedy decode like demo.postprocess. probs [N,T,C] softmax -> (text, text_conf, p_eos).

    text_conf = mean max-probability over the decoded characters (0.0 for an empty string);
    p_eos = probs[:, 0, null_label] (end token at step 0).
    """
    out = []
    for p in probs:
        ids = p.argmax(axis=1)
        text = charset.get_text(ids, padding=False, trim=False).split(charset.null_char)[0]
        n = len(text)
        conf = float(p[np.arange(n), ids[:n]].mean()) if n else 0.0
        out.append((text.lower(), conf, float(p[0, charset.null_label])))
    return out


class ABINetRecognizer:
    """Batched ABINet inference on uint8 crops (BGR, as produced by roi_rotate.crop_quad)."""

    batch_size = 64

    def __init__(self, ckpt, device, fp16=True):
        ab = _import_abinet()
        ckpt = Path(DEFAULT_CKPT if ckpt is None else ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"ABINet checkpoint not found: {ckpt}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device is cuda but CUDA is not available")
        self.config = make_config(ckpt)
        self.charset = ab["utils"].CharsetMapper(filename=self.config.dataset_charset_path,
                                                 max_length=self.config.dataset_max_length + 1)
        model = ab["demo"].get_model(self.config)
        model = ab["demo"].load(model, self.config.model_checkpoint, device="cpu")
        self.model = model.to(self.device).eval()
        self.fp16 = bool(fp16) and self.device.type == "cuda"  # AMP; weights stay fp32
        self.h, self.w = int(self.config.dataset_image_height), int(self.config.dataset_image_width)
        self.c = int(self.config.dataset_max_length) + 1
        d = self.config.model_vision_d_model
        self.d = 512 if d is None else int(d)
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        """BGR uint8 crops -> normalised float [N,3,h,w] on device (== demo.preprocess on RGB)."""
        arr = np.empty((len(crops), self.h, self.w, 3), np.uint8)
        for i, c in enumerate(crops):
            c = _as_bgr_uint8(c)
            if c.shape[:2] != (self.h, self.w):
                c = cv2.resize(c, (self.w, self.h))
            arr[i] = c[:, :, ::-1]
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def _forward(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = self.preprocess(crops)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            a_res, _, v_res = self.model(x)
        probs = torch.softmax(a_res["logits"].float(), dim=-1)
        feats = v_res["feature"]
        if feats.shape[1:] != (self.c, self.d):
            raise RuntimeError(f"vision feature shape {tuple(feats.shape)} != (N, {self.c}, {self.d})")
        return probs.cpu().numpy(), feats.to(torch.float16).cpu().numpy()

    def predict(self, crops: list[np.ndarray]) -> list[dict]:
        """One dict {text, text_conf, p_eos, feat[c,d] float16} per crop, in input order."""
        out = []
        for s in range(0, len(crops), max(1, int(self.batch_size))):
            probs, feats = self._forward(crops[s:s + self.batch_size])
            for (text, conf, p_eos), f in zip(decode(probs, self.charset), feats):
                out.append({"text": text, "text_conf": conf, "p_eos": p_eos, "feat": f})
        return out
