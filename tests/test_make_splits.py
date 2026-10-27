import importlib.util
import logging
from pathlib import Path

import pytest

from logo.data.schema import read_split
from logo.data.splits import check_disjoint, make_splits

_spec = importlib.util.spec_from_file_location(
    "make_splits_tool", Path(__file__).resolve().parents[1] / "tools" / "make_splits.py")
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)

LOG = logging.getLogger("test_make_splits")


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _raw(root: Path, test_gt: bool) -> None:
    for i in range(10):
        _touch(root, f"dstext/train/Sports/Video_{i}_0_{i}.mp4")
        _touch(root, f"dstext/train/Sports/Video_{i}_0_{i}_GT.xml")
    for i in range(10, 15):
        _touch(root, f"dstext/train/Game/Video_{i}_1_0.mp4")
        _touch(root, f"dstext/train/Game/video_{i}_1_0_GT.xml")
    _touch(root, "dstext/train/Game/Video_99_9_9.mp4")  # no GT -> excluded
    for i in range(20, 23):
        _touch(root, f"dstext/test/Game/Video_{i}_2_0.mp4")
        if test_gt:
            _touch(root, f"dstext/test/Game/Video_{i}_2_0_GT.xml")


def test_make_splits_80_20_disjoint_deterministic():
    vids = {f"v{i}": "g" for i in range(20)}
    s = make_splits(vids, ["t1", "t0"], 0.2, seed=3)
    assert s == make_splits(vids, ["t0", "t1"], 0.2, seed=3)
    assert len(s["val"]) == 4 and len(s["train"]) == 16 and s["test"] == ["t0", "t1"]
    assert set(s["train"]) | set(s["val"]) == set(vids)
    assert make_splits(vids, [], 0.2, seed=4)["val"] != s["val"]


def test_check_disjoint_rejects_overlap():
    check_disjoint({"train": ["a"], "val": ["b"], "test": ["c"]})
    with pytest.raises(AssertionError):
        check_disjoint({"train": ["a"], "val": ["b"], "test": ["A"]})
    with pytest.raises(AssertionError):
        check_disjoint({"train": ["a", "a"]})
    with pytest.raises(AssertionError):
        make_splits({"a": "g", "b": "g"}, ["a"], 0.5, seed=0)


def test_tool_writes_files_and_warns_without_test_gt(tmp_path, caplog):
    _raw(tmp_path / "raw", test_gt=False)
    out = tmp_path / "splits"
    log = logging.getLogger("ms_tool")
    log.propagate = True
    with caplog.at_level(logging.WARNING, logger="ms_tool"):
        s = ms.write_dataset("dstext", tmp_path / "raw", out, 0.2, 42, False, log)
    assert s is not None
    tr, va, te = (read_split(out / f"dstext_{k}.txt") for k in ("train", "val", "test"))
    assert len(va) == 3 and len(tr) == 12 and "Video_99_9_9" not in tr + va  # 2 Sports + 1 Game
    assert te == ["Video_20_2_0", "Video_21_2_0", "Video_22_2_0"]
    assert sum(v.endswith("_1_0") for v in va) == 1
    assert "val acts as the held-out test set" in caplog.text
    # rerun is a no-op; a different seed is refused without --force, accepted with it
    assert ms.write_dataset("dstext", tmp_path / "raw", out, 0.2, 42, False, LOG) == s
    assert ms.write_dataset("dstext", tmp_path / "raw", out, 0.2, 7, False, LOG) is None
    assert read_split(out / "dstext_val.txt") == va
    assert ms.write_dataset("dstext", tmp_path / "raw", out, 0.2, 7, True, LOG) is not None


def test_tool_no_warning_with_test_gt(tmp_path, caplog):
    _raw(tmp_path / "raw", test_gt=True)
    log = logging.getLogger("ms_tool2")
    log.propagate = True
    with caplog.at_level(logging.WARNING, logger="ms_tool2"):
        ms.write_dataset("dstext", tmp_path / "raw", tmp_path / "s", 0.2, 1, False, log)
    assert "held-out" not in caplog.text
