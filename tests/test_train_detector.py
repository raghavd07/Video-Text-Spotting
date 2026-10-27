import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from logo.utils.io import load_json, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_tool", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


td = _load("train_detector")

# stands in for PaddleDetection tools/train.py: records argv, writes 2 epoch checkpoints (resumes too)
FAKE_TRAIN = """
import json, sys
from pathlib import Path
a = sys.argv
save = Path([o for o in a if o.startswith('save_dir=')][0].split('=', 1)[1])
save.mkdir(parents=True, exist_ok=True)
start = 0
if '-r' in a:
    start = int(Path(a[a.index('-r') + 1]).name) + 1
for e in range(start, 2):
    name = str(e) if e < 1 else 'model_final'
    for ext in ('.pdparams', '.pdopt'):
        (save / (name + ext)).write_text('x')
print('FAKE TRAIN', json.dumps(a[1:]))
"""


class TagLoader(yaml.SafeLoader):
    pass


TagLoader.add_multi_constructor("!", lambda ld, suffix, node: ld.construct_mapping(node))


def ppdet_yml(name):
    return yaml.load((ROOT / "configs" / name).read_text(encoding="utf-8"), Loader=TagLoader)


def test_checkpoint_helpers(tmp_path):
    assert td.latest_checkpoint(tmp_path / "none") is None
    for name, exts in (("0", (".pdparams", ".pdopt")), ("3", (".pdparams", ".pdopt")),
                       ("7", (".pdparams",)), ("best_model", (".pdparams",))):
        for ext in exts:
            (tmp_path / (name + ext)).write_text("x")
    assert td.latest_checkpoint(tmp_path) == tmp_path / "3"  # 7 lacks .pdopt
    assert td.resolve_resume("auto", tmp_path) == tmp_path / "3"
    assert td.resolve_resume("none", tmp_path) is None
    assert td.resolve_resume(str(tmp_path / "0.pdparams"), tmp_path) == tmp_path / "0"
    with pytest.raises(SystemExit):
        td.resolve_resume(str(tmp_path / "9"), tmp_path)
    assert not td.is_finished(tmp_path)
    assert td.smoke_config(Path("configs/a_s.yml")) == Path("configs/a_s_smoke.yml")


def test_build_argv(tmp_path):
    a = td.build_argv(Path("t.py"), Path("c.yml"), tmp_path, tmp_path / "2", "gpu", True)
    assert a[:4] == ["t.py", "-c", "c.yml", "--amp"]
    assert a[a.index("-r") + 1] == str(tmp_path / "2") and "--eval" in a
    assert a[-2:] == [f"save_dir={tmp_path.as_posix()}", "use_gpu=true"]
    c = td.build_argv(Path("t.py"), Path("c.yml"), tmp_path, None, "cpu", False)
    assert "--amp" not in c and "-r" not in c and c[-1] == "use_gpu=false"


def test_ext_op_check(monkeypatch):
    monkeypatch.setattr(td.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit, match="ext_op"):
        td.check_ext_op()


def make_root(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in ("p.yml", "p_smoke.yml"):
        (tmp_path / "configs" / name).write_text("epoch: 1\n", encoding="utf-8")
    cfg = {"exp_id": "dt", "dataset": "icdar2015v", "split": "val", "seed": 3,
           "detector": {"config": "configs/p.yml", "weights": "runs/dt/ckpt/model_final.pdparams",
                        "device": "cpu"}}
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    tp = tmp_path / td.TRAIN_PY
    tp.parent.mkdir(parents=True)
    tp.write_text(FAKE_TRAIN, encoding="utf-8")


def test_main_resume_and_smoke(tmp_path, monkeypatch):
    make_root(tmp_path)
    monkeypatch.setattr(td, "check_ext_op", lambda: None)
    monkeypatch.setattr(td, "seed_all", td.set_seed)  # paddle import alone takes ~15 s
    monkeypatch.chdir(tmp_path)
    ck = tmp_path / "runs" / "dt" / "ckpt"
    ck.mkdir(parents=True)
    for ext in (".pdparams", ".pdopt"):  # interrupted after epoch 0
        (ck / ("0" + ext)).write_text("x")

    assert td.main(["--config", "cfg.yaml", "--root", str(tmp_path)]) == 0
    run = tmp_path / "runs" / "dt"
    log = (run / "log.txt").read_text(encoding="utf-8")
    assert "FAKE TRAIN" in log and '"-r"' in log and "use_gpu=false" in log
    m = load_json(run / "metrics.json")["meta"]["det_train"]
    assert m["resumed_from"] == "runs/dt/ckpt/0" and m["final_weights"] == "runs/dt/ckpt/model_final.pdparams"
    assert m["seed"] == 3 and m["amp"] is False
    assert load_yaml(run / "config.yaml")["exp_id"] == "dt"
    assert (run / "ppdet_config.yml").is_file()

    # finished -> no retrain; smoke goes to its own run dir with the _smoke ppdet config
    assert td.main(["--config", "cfg.yaml", "--root", str(tmp_path)]) == 0
    assert (run / "log.txt").read_text(encoding="utf-8").count("FAKE TRAIN") == 1
    assert td.main(["--config", "cfg.yaml", "--root", str(tmp_path), "--smoke"]) == 0
    s = tmp_path / "runs" / "dt_smoke"
    assert "configs/p_smoke.yml" in (s / "log.txt").read_text(encoding="utf-8")
    assert load_json(s / "metrics.json")["meta"]["det_train"]["smoke"] is True
    assert (s / "ckpt" / "model_final.pdparams").is_file()


def test_ppdet_configs_consistent():
    main, smoke = ppdet_yml("ppyoloe_r_text_s.yml"), ppdet_yml("ppyoloe_r_text_s_smoke.yml")
    det = load_yaml(ROOT / "configs" / "det_ppyoloe_r_text_s.yaml")["detector"]
    assert (ROOT / "configs" / main["_BASE_"][0]).is_file()
    assert main["num_classes"] == 1 and main["TrainReader"]["batch_size"] == 2
    hw = det["input_hw"]
    for reader, op in (("TrainReader", "RResize"), ("EvalReader", "RResize"), ("TestReader", "Resize")):
        tf = [t[op] for t in main[reader]["sample_transforms"] if op in t]
        assert tf and tf[0]["target_size"] == hw
    assert main["TrainDataset"]["anno_path"] == "det_dota/train.json"
    assert main["EvalDataset"]["anno_path"] == "det_dota/val.json"
    assert det["config"] == "configs/ppyoloe_r_text_s.yml"
    assert smoke["_BASE_"] == ["ppyoloe_r_text_s.yml"] and smoke["epoch"] == 1
    assert smoke["TrainDataset"]["anno_path"] == "det_dota/train_smoke.json"
