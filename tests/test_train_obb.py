import importlib.util
from pathlib import Path

import pytest
import yaml

from logo.utils.io import get_logger, load_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


to = _load("train_obb")


class FakeYOLO:
    calls = []
    _project = ""

    def __init__(self, weights, task=None):
        self.weights, self.task = weights, task

    def train(self, **kw):
        FakeYOLO.calls.append((self.weights, self.task, kw))
        wdir = Path(kw.get("project", self._project)) / "train" / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "last.pt").write_bytes(b"x")
        if kw.get("resume") or kw.get("epochs", 0) >= 1:
            (wdir / "best.pt").write_bytes(b"x")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGO_ROOT", str(tmp_path))
    FakeYOLO.calls = []
    (tmp_path / "data/obb/icdar2015v").mkdir(parents=True)
    (tmp_path / "data/obb/icdar2015v/icdar2015v.yaml").write_text("names: {0: text}\n", encoding="utf-8")
    cfg = {"exp_id": "Y", "dataset": "icdar2015v", "split": "val", "seed": 7,
           "detector": {"device": "cpu", "obb_data": {"root": "data/obb"},
                        "train": {"init": "m.pt", "epochs": 3, "batch": 2, "smoke_fraction": 0.2}}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    yield tmp_path
    lg = get_logger("train_obb")
    for h in list(lg.handlers):
        h.close()
        lg.removeHandler(h)


def test_kwargs_and_smoke(root):
    assert to.main(["--config", str(root / "cfg.yaml"), "--smoke"], yolo_cls=FakeYOLO) == 0
    w, task, kw = FakeYOLO.calls[0]
    assert Path(w) == root / "m.pt" and task == "obb"
    assert kw["epochs"] == 1 and kw["fraction"] == 0.2 and kw["seed"] == 7 and kw["device"] == "cpu"
    assert kw["amp"] is False and kw["batch"] == 2 and kw["name"] == "train" and kw["conf"] == 0.05
    assert Path(kw["project"]) == root / "runs/Y_smoke" and kw["data"].endswith("icdar2015v/icdar2015v.yaml")
    m = load_json(root / "runs/Y_smoke/metrics.json")
    assert m["meta"]["det_train"]["finished"] and (root / "runs/Y_smoke/config.yaml").is_file()

    gpu = to.train_kwargs(yaml.safe_load((root / "cfg.yaml").read_text()), root, root / "r", "gpu", False)
    assert gpu["device"] == 0 and gpu["amp"] is True and gpu["epochs"] == 3 and gpu["fraction"] == 1.0


def test_resume_and_skip(root):
    wdir = root / "runs/Y/train/weights"
    wdir.mkdir(parents=True)
    (wdir / "last.pt").write_bytes(b"x")
    FakeYOLO._project = str(root / "runs/Y")
    assert to.main(["--config", str(root / "cfg.yaml")], yolo_cls=FakeYOLO) == 0
    w, _, kw = FakeYOLO.calls[0]
    assert Path(w) == wdir / "last.pt" and kw == {"resume": True}
    assert to.main(["--config", str(root / "cfg.yaml")], yolo_cls=FakeYOLO) == 0  # finished -> no call
    assert len(FakeYOLO.calls) == 1


def test_missing_dataset_yaml(root):
    (root / "data/obb/icdar2015v/icdar2015v.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        to.main(["--config", str(root / "cfg.yaml")], yolo_cls=FakeYOLO)
