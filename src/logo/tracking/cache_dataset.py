"""LST-Matcher training cache (tools/build_lst_cache.py) and its clip dataset.

data/cache/<ds>/<video>.npz (np.savez, uncompressed), instances sorted by frame:
  frame_ids [N] int32, quads [N,8] float32 px, feats [N,c,d] float16 (LSC rows), score_fused [N] float32,
  track_ids [N] int64 (GT track id, -1 = no GT), frames [F] int32 (every frame 1..num_frames),
  frame_offsets [F+1] int64 (instances of frames[k] are rows frame_offsets[k]:frame_offsets[k+1]),
  img_wh [2] int32.
LSTCacheDataset item = clip_len consecutive frames of one video, each an LSTAdapter frame dict whose
"feats" are the raw LSC features [n,c,d]; the training loop replaces them by VPMM(quads, feats, img_wh)
[n,2d] before train_step. -1 track ids count as "no GT" in the adapter (it only uses ids > 0).
feats are memory-mapped from the npz, so a clip reads only its own rows.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from logo.utils.paths import check_dataset, project_root

CACHE_KEYS = ("frame_ids", "quads", "feats", "score_fused", "track_ids", "frames", "frame_offsets", "img_wh")


def cache_dir(ds: str) -> Path:
    return project_root() / "data" / "cache" / check_dataset(ds)


def save_cache(path, arrays: dict) -> None:
    """Validates and writes one video cache atomically (tmp file + replace)."""
    missing = [k for k in CACHE_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"cache missing arrays {missing}")
    n, off = len(arrays["frame_ids"]), np.asarray(arrays["frame_offsets"])
    if any(len(arrays[k]) != n for k in ("quads", "feats", "score_fused", "track_ids")):
        raise ValueError("per-instance arrays differ in length")
    if len(off) != len(arrays["frames"]) + 1 or off[0] != 0 or off[-1] != n or (np.diff(off) < 0).any():
        raise ValueError("frame_offsets inconsistent with frames / instances")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **{k: np.asarray(arrays[k]) for k in CACHE_KEYS})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def npz_memmap(path, key: str) -> np.ndarray:
    """Read-only memmap of an uncompressed npz member; falls back to a full load if it is compressed."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path) as z:
            return z[key]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError(f"{path}: bad zip local header for {key}")
        name_len, extra_len = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) \
            else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        offset = f.tell()
    if dtype.hasobject:
        raise ValueError(f"{path}:{key} holds Python objects")
    if int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape,
                     order="F" if fortran else "C")


def collate_first(batch: list) -> list[dict]:
    """DataLoader collate_fn for batch_size=1: returns the single clip unchanged."""
    if len(batch) != 1:
        raise ValueError("LSTCacheDataset clips have variable size; use batch_size=1")
    return batch[0]


class LSTCacheDataset(Dataset):
    """Clips of clip_len consecutive frames (start every `stride` frames), never crossing videos.

    paths: a cache directory (all *.npz in it) or an iterable of npz files. Videos with fewer than
    clip_len frames give no clip. Item: list of clip_len dicts
      {"video", "frame", "quads" float32 [n,8], "feats" float16 Tensor [n,c,d], "scores" float32 [n],
       "img_wh" (w, h), "track_ids" int64 [n]}.
    """

    def __init__(self, paths: Union[str, Path, Iterable], clip_len: int = 6, stride: int = 1):
        if clip_len < 1 or stride < 1:
            raise ValueError("clip_len and stride must be >= 1")
        if isinstance(paths, (str, Path)) and Path(paths).is_dir():
            files = sorted(Path(paths).glob("*.npz"))
        elif isinstance(paths, (str, Path)):
            files = [Path(paths)]
        else:
            files = [Path(p) for p in paths]
        if not files:
            raise FileNotFoundError(f"no cache npz in {paths}")
        self.clip_len, self.stride = int(clip_len), int(stride)
        self.videos, self._meta, self.index = [], [], []
        for p in files:
            with np.load(p) as z:
                meta = {k: z[k] for k in CACHE_KEYS if k != "feats"}
            v = len(self.videos)
            self.videos.append(p)
            self._meta.append(meta)
            n_frames = len(meta["frames"])
            self.index.extend((v, s) for s in range(0, n_frames - self.clip_len + 1, self.stride))
        self._feats: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _feat_rows(self, v: int, lo: int, hi: int) -> np.ndarray:
        if v not in self._feats:  # opened lazily so each DataLoader worker maps its own handle
            self._feats[v] = npz_memmap(self.videos[v], "feats")
        return np.array(self._feats[v][lo:hi], dtype=np.float16)

    def __getitem__(self, i: int) -> list[dict]:
        v, start = self.index[i]
        m = self._meta[v]
        off = m["frame_offsets"]
        lo_all, hi_all = int(off[start]), int(off[start + self.clip_len])
        feats = torch.from_numpy(self._feat_rows(v, lo_all, hi_all))
        wh = (int(m["img_wh"][0]), int(m["img_wh"][1]))
        clip = []
        for k in range(start, start + self.clip_len):
            lo, hi = int(off[k]), int(off[k + 1])
            clip.append({
                "video": self.videos[v].stem, "frame": int(m["frames"][k]),
                "quads": m["quads"][lo:hi].astype(np.float32),
                "feats": feats[lo - lo_all: hi - lo_all],
                "scores": m["score_fused"][lo:hi].astype(np.float32),
                "img_wh": wh, "track_ids": m["track_ids"][lo:hi].astype(np.int64)})
        return clip
