import logging
import random
from pathlib import Path

import numpy as np
import pytest

from logo.utils.io import get_logger, load_json, load_yaml, save_json, set_seed


def _close(log):
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def test_json_roundtrip_numpy(tmp_path):
    p = tmp_path / "a" / "b.json"
    save_json({"i": np.int64(3), "f": np.float32(0.5), "arr": np.arange(3), "b": np.bool_(True),
               "p": Path("x"), "t": "Straße"}, p)
    assert load_json(p) == {"i": 3, "f": 0.5, "arr": [0, 1, 2], "b": True, "p": "x", "t": "Straße"}
    assert not (tmp_path / "a" / "b.json.tmp").exists()


def test_save_json_unserializable_keeps_old_file(tmp_path):
    p = tmp_path / "c.json"
    save_json({"ok": 1}, p)
    with pytest.raises(TypeError):
        save_json({"bad": object()}, p)
    assert load_json(p) == {"ok": 1}
    assert not (tmp_path / "c.json.tmp").exists()


def test_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("seed: 7\nfusion: {mode: soft, w: 0.3}\n", encoding="utf-8")
    assert load_yaml(p) == {"seed": 7, "fusion": {"mode": "soft", "w": 0.3}}
    empty = tmp_path / "e.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}


def test_set_seed_reproducible():
    set_seed(5)
    a = (random.random(), np.random.rand(3))
    set_seed(5)
    b = (random.random(), np.random.rand(3))
    assert a[0] == b[0] and np.allclose(a[1], b[1])


def test_set_seed_torch():
    torch = pytest.importorskip("torch")
    set_seed(5)
    a = torch.rand(3)
    set_seed(5)
    assert torch.equal(a, torch.rand(3))


def test_logger_writes_file_once(tmp_path):
    f = tmp_path / "log.txt"
    get_logger("t_logger", f)
    log = get_logger("t_logger", f)
    log.info("hello")
    for h in log.handlers:
        h.flush()
    assert f.read_text(encoding="utf-8").count("hello") == 1
    assert sum(isinstance(h, logging.FileHandler) for h in log.handlers) == 1
    assert len(log.handlers) == 2
    _close(log)


def test_logger_writes_console(tmp_path, capsys):
    log = get_logger("t_logger_console", tmp_path / "sub" / "log.txt")
    log.info("to-console")
    assert "to-console" in capsys.readouterr().out
    assert "to-console" in (tmp_path / "sub" / "log.txt").read_text(encoding="utf-8")
    _close(log)


def test_logger_console_only(capsys):
    log = get_logger("t_logger_nofile")
    log.info("only-console")
    assert "only-console" in capsys.readouterr().out
    assert not any(isinstance(h, logging.FileHandler) for h in log.handlers)
    _close(log)


def test_base_config_has_contract_keys():
    cfg = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    assert {"exp_id", "dataset", "split", "seed", "detector", "lsc", "fusion", "tracker", "voting", "eval"} <= cfg.keys()
    assert {"ckpt", "det_thr", "iou_neg"} <= cfg["lsc"].keys()
    assert {"mode", "w"} <= cfg["fusion"].keys() and cfg["fusion"]["mode"] in ("hard", "soft")
    assert {"type", "thr"} <= cfg["tracker"].keys() and cfg["tracker"]["type"] in ("simple", "lst")
    assert {"enabled", "tau", "min_len"} <= cfg["voting"].keys()
    assert cfg["eval"] == {"iou": 0.5, "spotting": True}
    assert cfg["split"] != "test"
