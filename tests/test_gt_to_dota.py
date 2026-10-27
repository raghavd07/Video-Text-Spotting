import importlib.util
from pathlib import Path

import pytest

from logo.utils.io import load_json, save_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g2d = _load("gt_to_dota")


def obj(x, dont_care=False, tid=1, w=40, h=10):
    return {"track_id": tid, "quad": [x, 5, x + w, 5, x + w, 5 + h, x, 5 + h], "text": "AB",
            "dont_care": dont_care, "quality": None}


def make_tree(root, ds="icdar2015v", n=6, missing=(6,)):
    (root / "splits").mkdir(parents=True)
    (root / "splits" / f"{ds}_train.txt").write_text("v1\n", encoding="utf-8")
    (root / "splits" / f"{ds}_test.txt").write_text("t1\n", encoding="utf-8")
    frames = {
        "1": [obj(10), obj(60, dont_care=True, tid=2)],
        "3": [obj(70, dont_care=True, tid=2)],           # only dont_care -> frame skipped
        "5": [obj(10), obj(90, w=0.5, h=0.5, tid=3), obj(150, tid=4, w=100)],  # degenerate box dropped
        "2": [obj(10)],                                    # not on the stride
    }
    save_json({"video": "v1", "dataset": ds, "width": 200, "height": 100, "num_frames": n,
               "frames": frames}, root / "data" / "processed" / ds / "v1.json")
    d = root / "data" / "frames" / ds / "v1"
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        if i not in missing:
            (d / f"{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def test_helpers():
    assert list(g2d.frame_ids(7, 2)) == [1, 3, 5, 7]
    assert g2d.clean_quad([0, 0, 1, 0, 1, 1, 0, 1], 100, 100, 4.0) is None
    q = g2d.clean_quad([-5, 2, 30, 2, 30, 120, -5, 120], 100, 100, 4.0)
    assert q == [0.0, 2.0, 30.0, 2.0, 30.0, 99.0, 0.0, 99.0]
    assert g2d.dota_line([1, 2, 3, 4, 5, 6, 7, 8]) == "1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 text 0"
    a = g2d.coco_annotation([10, 5, 50, 5, 50, 15, 10, 15], 3)
    assert a["bbox"] == [10, 5, 40, 10] and a["area"] == 400 and a["iscrowd"] == 0
    assert a["category_id"] == 1 and a["segmentation"] == [[10, 5, 50, 5, 50, 15, 10, 15]]


def test_convert_split(tmp_path):
    make_tree(tmp_path)
    out = tmp_path / "data" / "det_dota"
    stale = out / "train" / "labelTxt" / "old.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("x", encoding="utf-8")

    st = g2d.convert_split(tmp_path, ["icdar2015v"], "train", 2, out, smoke_n=1)
    assert not stale.exists()
    assert st["images"] == 2 and st["boxes"] == 3
    assert st["dont_care_omitted"] == 2 and st["degenerate_dropped"] == 1
    assert st["frames_only_dont_care"] == 1 and st["frames_missing"] == 0  # 6 is not on stride 2

    coco = load_json(out / "train.json")
    assert [im["file_name"] for im in coco["images"]] == ["icdar2015v/v1/000001.jpg",
                                                          "icdar2015v/v1/000005.jpg"]
    assert coco["categories"] == [{"id": 1, "name": "text", "supercategory": "text"}]
    assert [a["id"] for a in coco["annotations"]] == [1, 2, 3]
    assert [a["image_id"] for a in coco["annotations"]] == [1, 2, 2]
    assert (out / "train" / "images.txt").read_text(encoding="utf-8").splitlines() == \
        [im["file_name"] for im in coco["images"]]

    lines = (out / "train" / "labelTxt" / "icdar2015v__v1__000001.txt").read_text(
        encoding="utf-8").splitlines()
    assert lines == ["10.0 5.0 50.0 5.0 50.0 15.0 10.0 15.0 text 0"]
    assert sorted(p.name for p in (out / "train" / "labelTxt").glob("*.txt")) == [
        "icdar2015v__v1__000001.txt", "icdar2015v__v1__000005.txt"]

    smoke = load_json(out / "train_smoke.json")
    assert len(smoke["images"]) == 1 and all(a["image_id"] == smoke["images"][0]["id"]
                                             for a in smoke["annotations"])


def test_missing_frames_and_test_guard(tmp_path):
    make_tree(tmp_path, missing=(1,))
    out = tmp_path / "out"
    st = g2d.convert_split(tmp_path, ["icdar2015v"], "train", 2, out)
    assert st["frames_missing"] == 1 and st["images"] == 1
    with pytest.raises(PermissionError):
        g2d.convert_split(tmp_path, ["icdar2015v"], "test", 2, out)


def test_main_cli(tmp_path):
    make_tree(tmp_path)
    assert g2d.main(["--root", str(tmp_path), "--datasets", "icdar2015v", "--stride", "1"]) == 0
    st = load_json(tmp_path / "data" / "det_dota" / "train_stats.json")
    assert st["stride"] == 1 and st["images"] == 3  # frames 1, 2, 5
