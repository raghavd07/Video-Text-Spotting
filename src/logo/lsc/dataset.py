"""LSC training data: charset filtering of GT text and a PyTorch Dataset over data/lsc/<split>.

Layout written by tools/build_lsc_dataset.py:
  data/lsc/<split>/labels.tsv   one line per crop: "<relative/path.png>\t<label>" (label "" = negative)
  data/lsc/<split>/images/...   BGR PNG crops (crop_quad, 32x128)
Tensors match ABINetRecognizer.preprocess: RGB, /255, ImageNet mean/std, [3,h,w] float32.
"""
from __future__ import annotations

import string
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from logo.lsc.roi_rotate import _as_bgr_uint8

ABINET_CHARSET = Path(__file__).resolve().parents[3] / "third_party" / "ABINet" / "data" / "charset_36.txt"
FALLBACK_CHARSET = string.ascii_lowercase + "1234567890"  # == charset_36.txt order
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_charset(path=None) -> str:
    """Characters of an ABINet charset file ("<idx>\\t<char>" lines); charset_36 fallback if absent."""
    p = Path(path) if path is not None else ABINET_CHARSET
    if not p.is_file():
        if path is not None:
            raise FileNotFoundError(p)
        return FALLBACK_CHARSET
    chars = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[1]:
            chars.append(parts[1])
    return "".join(chars)


def clean_label(text: str, charset: str) -> str:
    """Lowercase and drop every character not in `charset` ("" means: drop the instance)."""
    allowed = set(charset)
    return "".join(ch for ch in str(text).lower() if ch in allowed)


def read_labels(tsv) -> list[tuple[str, str]]:
    out = []
    for ln in Path(tsv).read_text(encoding="utf-8").splitlines():
        if not ln:
            continue
        rel, _, label = ln.partition("\t")
        out.append((rel, label))
    return out


# ---------------------------------------------------------------- augmentation (cv2, uint8 BGR)

def _border(img, m, persp=False):
    h, w = img.shape[:2]
    f = cv2.warpPerspective if persp else cv2.warpAffine
    return f(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def aug_rotation(img, rng, degrees=10.0):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-degrees, degrees)), 1.0)
    return _border(img, m)


def aug_affine(img, rng, degrees=5.0, shear=15.0, scale=(0.85, 1.1), translate=0.05):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-degrees, degrees)),
                                float(rng.uniform(*scale)))
    sx = np.tan(np.deg2rad(rng.uniform(-shear, shear)))
    sh = np.array([[1, sx, -sx * h / 2], [0, 1, 0], [0, 0, 1]], np.float64)
    m = (np.vstack([m, [0, 0, 1]]) @ sh)[:2]
    m[0, 2] += rng.uniform(-translate, translate) * w
    m[1, 2] += rng.uniform(-translate, translate) * h
    return _border(img, m)


def aug_perspective(img, rng, distortion=0.25):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    jit = rng.uniform(0, distortion, (4, 2)) * np.float32([w / 2, h / 2])
    sign = np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    m = cv2.getPerspectiveTransform(src, (src + jit * sign).astype(np.float32))
    return _border(img, m, persp=True)


def aug_quality(img, rng):
    """One of: gaussian blur, motion blur, gaussian noise, down/up rescale, JPEG compression."""
    h, w = img.shape[:2]
    k = int(rng.integers(5))
    if k == 0:
        return cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.5)))
    if k == 1:
        n = int(rng.integers(3, 8))
        ker = np.zeros((n, n), np.float32)
        ker[n // 2, :] = 1.0 / n
        rot = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), float(rng.uniform(0, 180)), 1.0)
        ker = cv2.warpAffine(ker, rot, (n, n))
        ker /= max(float(ker.sum()), 1e-6)
        return cv2.filter2D(img, -1, ker)
    if k == 2:
        noise = rng.normal(0, rng.uniform(3, 15), img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if k == 3:
        f = float(rng.uniform(1.5, 4.0))
        small = cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(10, 50))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def aug_color(img, rng, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05):
    x = img.astype(np.float32)
    x = x * rng.uniform(1 - brightness, 1 + brightness)
    mean = x.mean()
    x = (x - mean) * rng.uniform(1 - contrast, 1 + contrast) + mean
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(1 - saturation, 1 + saturation)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue) * 180) % 180
    hsv = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment(img: np.ndarray, rng: np.random.Generator, p_geo=0.5, p_quality=0.25,
            p_color=0.25) -> np.ndarray:
    """ABINet-style pipeline: geometry (rotation | affine | perspective), deterioration, color jitter."""
    if rng.random() < p_geo:
        img = (aug_rotation, aug_affine, aug_perspective)[int(rng.integers(3))](img, rng)
    if rng.random() < p_quality:
        img = aug_quality(img, rng)
    if rng.random() < p_color:
        img = aug_color(img, rng)
    return np.ascontiguousarray(img)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> normalised RGB float32 [3,H,W]."""
    x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float().div_(255.0)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class LSCDataset(Dataset):
    """(image [3,h,w] float32, label str) pairs from data/lsc/<split>; label "" is a negative.

    With train=True every sample is augmented. The augmentation RNG is drawn from torch's RNG, so
    runs are reproducible after set_seed() and DataLoader workers get distinct streams.
    """

    def __init__(self, root, train: bool = False, out_hw=(32, 128), max_length: Optional[int] = None):
        self.root = Path(root)
        self.train = bool(train)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        items = read_labels(self.root / "labels.tsv")
        if max_length is not None:
            items = [it for it in items if len(it[1]) <= int(max_length)]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def load(self, i: int) -> np.ndarray:
        rel, _ = self.items[i]
        img = cv2.imread(str(self.root / rel), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / rel)
        img = _as_bgr_uint8(img)
        h, w = self.out_hw
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        return img

    def __getitem__(self, i: int):
        img = self.load(i)
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
            img = augment(img, rng)
        return to_tensor(img), self.items[i][1]
