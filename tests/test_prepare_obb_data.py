import logging

import numpy as np
import pytest
import yaml

from logo.detector import prepare_obb_data as po
from logo.utils.io import load_json, save_json

DS = "icdar2015v"
W, H = 80, 40


def obj(quad, dont_care=False, tid=1):
    return {"track_id": tid, "quad": quad, "text": "AB", "dont_care": dont_care, "quality": None}


def make_tree(root, n=6):
    cv2 = pytest.importorskip("cv2")
    (root / "splits").mkdir()
    for split, vids in (("train", ["v1"]), ("val", ["v2"]), ("test", ["t1"])):
        (root / "splits" / f"{DS}_{split}.txt").write_text("\n".join(vids) + "\n", encoding="utf-8")
    img = np.full((H, W, 3), 255, np.uint8)
    for v in ("v1", "v2"):
        d = root / "data" / "frames" / DS / v
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            if not (v == "v1" and i == 5):  # one missing frame
                cv2.imwrite(str(d / f"{i:06d}.jpg"), img)
        frames = {
            "1": [obj([8, 4, 40, 4, 40, 20, 8, 20]), obj([50, 25, 70, 25, 70, 35, 50, 35], True, 2)],
            "3": [obj([-10, 10, 20, 10, 20, 30, -10, 30])],  # partly outside -> clipped
            "4": [obj([100, 50, 120, 50, 120, 60, 100, 60])],  # fully outside -> dropped
        }
        save_json({"video": v, "dataset": DS, "width": W, "height": H, "num_frames": n, "frames": frames},
                  root / "data" / "processed" / DS / f"{v}.json")
    return {"exp_id": "e1", "dataset": DS, "split": "val", "seed": 0,
            "detector": {"obb_data": {"root": "data/obb", "stride": 2, "val_stride": 3,
                                      "mask_dont_care": True}}}


def test_quad_to_label_normalizes_and_clips():
    assert po.quad_to_label([8, 4, 40, 4, 40, 20, 8, 20], W, H) == pytest.approx(
        [0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5])
    lab = po.quad_to_label([-10, 10, 20, 10, 20, 30, -10, 30], W, H)
    assert min(lab) >= 0 and max(lab) <= 1 and lab[0] == 0.0
    assert po.quad_to_label([100, 50, 120, 50, 120, 60, 100, 60], W, H) is None


def test_prepare_writes_ultralytics_layout(tmp_path):
    cv2 = pytest.importorskip("cv2")
    cfg = make_tree(tmp_path)
    res = po.prepare(cfg, tmp_path, log=logging.getLogger("t"))
    out = tmp_path / "data/obb" / DS

    tr = res["splits"]["train"]  # stride 2 over 6 frames -> 1, 3, 5 (5 missing)
    assert sorted(p.name for p in (out / "images/train").iterdir()) == ["v1_000001.jpg", "v1_000003.jpg"]
    assert tr["missing_frames"] == 1 and tr["images"] == 2 and tr["boxes"] == 2 and tr["masked"] == 1
    lines = (out / "labels/train/v1_000001.txt").read_text().splitlines()
    assert len(lines) == 1 and lines[0].split()[0] == "0" and len(lines[0].split()) == 9
    assert [float(v) for v in lines[0].split()[1:]] == pytest.approx([0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5])

    masked = cv2.imread(str(out / "images/train/v1_000001.jpg"))
    assert abs(int(masked[30, 60, 0]) - 114) < 8 and masked[10, 20, 0] > 240  # dont_care grey, text kept

    va = res["splits"]["val"]  # stride 3 -> 1, 4; frame 4 box outside -> empty label
    assert va["images"] == 2 and va["empty"] == 1
    assert (out / "labels/val/v2_000004.txt").read_text() == ""

    data = yaml.safe_load((out / f"{DS}.yaml").read_text())
    assert data["names"] == {0: "text"} and data["train"] == "images/train" and data["val"] == "images/val"
    assert load_json(out / "stats.json")["splits"]["val"]["stride"] == 3
    assert load_json(tmp_path / "runs/e1/metrics.json")["meta"]["obb_data"]["yaml"] == res["yaml"]

    res2 = po.prepare(cfg, tmp_path, log=logging.getLogger("t"))  # idempotent, skips existing
    assert res2["splits"]["train"]["skipped_existing"] == 2


def test_prepare_refuses_test_split(tmp_path):
    cfg = make_tree(tmp_path)
    with pytest.raises(ValueError):
        po.prepare(cfg, tmp_path, splits=("test",), log=logging.getLogger("t"))


def test_labels_pass_ultralytics_verifier(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics.data.utils import img2label_paths, verify_image_label

    cfg = make_tree(tmp_path)
    po.prepare(cfg, tmp_path, log=logging.getLogger("t"))
    imgs = sorted(str(p) for p in (tmp_path / "data/obb" / DS / "images").rglob("*.jpg"))
    for im, lb in zip(imgs, img2label_paths(imgs)):
        r = verify_image_label((im, lb, "", False, 1, 0, 0))
        assert r[8] == 0, r[9]  # nc (corrupt) == 0
        assert r[6] == 1 or r[7] == 1  # found, or empty
