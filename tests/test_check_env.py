import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_env", Path(__file__).resolve().parents[1] / "tools" / "check_env.py")
check_env = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_env)


def _info(cuda=True, missing=(), torch_ok=True, py=(3, 11)):
    torch = None
    if torch_ok:
        torch = {"version": "2.1.2+cu121", "cuda_build": "12.1", "cuda": cuda,
                 "gpu": "Fake GPU" if cuda else None, "vram_gb": 6.0 if cuda else None, "cudnn": 8801 if cuda else None}
    req = {label: (None if label in missing else "1.0") for label, _ in check_env.REQUIRED}
    req["numpy"] = None if "numpy" in missing else "1.26.4"
    return {"python": f"{py[0]}.{py[1]}.0", "python_tuple": py, "platform": "test", "torch": torch,
            "required": req, "optional": {label: None for label, _ in check_env.OPTIONAL}}


def test_module_version():
    assert check_env.module_version("json") is not None
    assert check_env.module_version("definitely_not_a_module_xyz") is None


def test_evaluate_ok():
    assert check_env.evaluate(_info()) == (0, [], [])


def test_evaluate_missing_and_no_cuda():
    code, problems, _ = check_env.evaluate(_info(missing=("shapely",)))
    assert code == 1 and "shapely" in problems[0]
    code, problems, _ = check_env.evaluate(_info(torch_ok=False))
    assert code == 1 and "torch" in problems[0]
    code, problems, _ = check_env.evaluate(_info(py=(3, 10)))
    assert code == 1
    assert check_env.evaluate(_info(cuda=False))[0] == 2
    code, problems, warnings = check_env.evaluate(_info(cuda=False), require_cuda=False)
    assert code == 0 and not problems and any("CUDA" in w for w in warnings)
    assert check_env.evaluate(_info(cuda=False, missing=("yaml",)))[0] == 1


def test_version_mismatch_is_warning_only():
    info = _info()
    info["required"]["numpy"] = "2.0.0"
    code, _, warnings = check_env.evaluate(info)
    assert code == 0 and any("numpy" in w for w in warnings)


@pytest.mark.parametrize("cuda,expected", [(True, 0), (False, 2)])
def test_main_exit_code(monkeypatch, capsys, cuda, expected):
    monkeypatch.setattr(check_env, "collect", lambda: _info(cuda=cuda))
    assert check_env.main([]) == expected
    out = capsys.readouterr().out
    assert "torch" in out and ("Fake GPU" in out) == cuda
    assert check_env.main(["--allow-cpu"]) == 0
