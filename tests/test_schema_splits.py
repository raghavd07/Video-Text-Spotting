import pytest

from logo.data.schema import list_videos, validate_gt, write_split
from logo.data.splits import train_val_split
from logo.utils.paths import check_split, split_file


def test_list_videos(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    write_split(split_file("dstext", "val"), ["Video_1", "Video_2"])
    assert list_videos("dstext", "val") == ["Video_1", "Video_2"]
    with pytest.raises(FileNotFoundError):
        list_videos("dstext", "train")
    with pytest.raises(ValueError):
        list_videos("bovtext", "val")


def test_test_split_guard():
    with pytest.raises(PermissionError):
        check_split("test")
    assert check_split("test", final=True) == "test"


def test_validate_gt_rejects():
    good = {"video": "v", "dataset": "dstext", "width": 4, "height": 4, "num_frames": 1,
            "frames": {"1": [{"track_id": 1, "quad": [0] * 8, "text": "a", "dont_care": False,
                              "quality": None}]}}
    validate_gt(good)
    with pytest.raises(ValueError):
        validate_gt({**good, "frames": {"2": []}})
    with pytest.raises(ValueError):
        validate_gt({k: v for k, v in good.items() if k != "width"})


def test_train_val_split_stratified_deterministic():
    vids = {f"a{i}": "A" for i in range(10)} | {f"b{i}": "B" for i in range(5)} | {"c0": "C"}
    tr, va = train_val_split(vids, 0.2, seed=1)
    assert (tr, va) == train_val_split(vids, 0.2, seed=1)
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(vids)
    assert sum(v.startswith("a") for v in va) == 2 and sum(v.startswith("b") for v in va) == 1
    assert "c0" in tr
