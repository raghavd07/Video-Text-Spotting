"""LSCDataset plus glyph pseudo-masks (data/lsc/<split>/masks.tsv from tools/make_glyph_masks.py).

The mask follows the image through the geometric augmentation (warped as a 4th channel with the
same random draws); quality and colour augmentation touch the image only. With the same RNG,
augment_with_mask produces exactly the image of logo.lsc.dataset.augment, so glyph on/off runs
see identical training crops.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from logo.lsc.dataset import (LSCDataset, aug_affine, aug_color, aug_perspective, aug_quality,
                              aug_rotation, to_tensor)


def read_masks(tsv) -> dict[str, str]:
    """masks.tsv ("<image rel>\\t<mask rel>\\t<fg_frac>") -> {image rel: mask rel}."""
    out = {}
    for ln in Path(tsv).read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[0]:
            out[parts[0]] = parts[1]
    return out


def augment_with_mask(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, p_geo=0.5,
                      p_quality=0.25, p_color=0.25) -> tuple[np.ndarray, np.ndarray]:
    """logo.lsc.dataset.augment on a BGR uint8 image, applying the geometry to the uint8 mask too."""
    if rng.random() < p_geo:
        f = (aug_rotation, aug_affine, aug_perspective)[int(rng.integers(3))]
        both = f(np.dstack([img, mask]), rng)
        img, mask = np.ascontiguousarray(both[..., :3]), np.ascontiguousarray(both[..., 3])
    if rng.random() < p_quality:
        img = aug_quality(img, rng)
    if rng.random() < p_color:
        img = aug_color(img, rng)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class GlyphLSCDataset(LSCDataset):
    """(image [3,h,w], label, mask [1,h,w] float32 in [0,1], valid float32 scalar).

    valid = 1 for crops listed in masks.tsv; crops without a mask get an all-zero mask with
    valid = 1 only if neg_zero and the label is "" (background), else valid = 0 (no L_seg).
    """

    def __init__(self, root, train: bool = False, out_hw=(32, 128), max_length: Optional[int] = None,
                 masks_tsv: str = "masks.tsv", neg_zero: bool = False):
        super().__init__(root, train=train, out_hw=out_hw, max_length=max_length)
        tsv = self.root / masks_tsv
        if not tsv.is_file():
            raise FileNotFoundError(f"{tsv} (run tools/make_glyph_masks.py --splits {self.root.name})")
        self.masks = read_masks(tsv)
        self.neg_zero = bool(neg_zero)

    def load_mask(self, i: int) -> Optional[np.ndarray]:
        rel = self.masks.get(self.items[i][0])
        if rel is None:
            return None
        p = self.root / rel
        m = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
        if m is None:
            raise FileNotFoundError(p)
        h, w = self.out_hw
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m

    def __getitem__(self, i: int):
        img, label = self.load(i), self.items[i][1]
        m = self.load_mask(i)
        valid = m is not None or (self.neg_zero and label == "")
        if m is None:
            m = np.zeros(self.out_hw, np.uint8)
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
            img, m = augment_with_mask(img, m, rng)
        mask = torch.from_numpy(m.astype(np.float32) / 255.0)[None]
        return to_tensor(img), label, mask, torch.tensor(float(valid))
