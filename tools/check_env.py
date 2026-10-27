"""Check the logo-torch environment.

Exit codes: 0 ok, 1 a required package is missing (or Python < 3.11), 2 CUDA unavailable.
Usage: python tools/check_env.py [--allow-cpu]
"""
import argparse
import importlib
import platform
import sys

# (label, import name)
REQUIRED = [("numpy", "numpy"), ("opencv", "cv2"), ("shapely", "shapely"), ("motmetrics", "motmetrics"),
            ("sklearn", "sklearn"), ("yaml", "yaml")]
OPTIONAL = [("pyclipper", "pyclipper"), ("tqdm", "tqdm"), ("matplotlib", "matplotlib"), ("pytest", "pytest"),
            ("ultralytics", "ultralytics")]
EXPECTED = {"torch": "2.1.2+cu121", "numpy": "1.26.4"}
MIN_PYTHON = (3, 11)


def module_version(import_name):
    """Version string of an importable module, or None if it cannot be imported."""
    try:
        mod = importlib.import_module(import_name)
    except Exception:  # ImportError, or a broken install (e.g. DLL load failed on Windows)
        return None
    return str(getattr(mod, "__version__", getattr(mod, "VERSION", "installed")))


def torch_info():
    """Dict with torch version and CUDA/GPU details, or None if torch cannot be imported."""
    try:
        import torch
    except Exception:
        return None
    info = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda": torch.cuda.is_available(),
            "gpu": None, "vram_gb": None, "cudnn": None}
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_gb"] = props.total_memory / 1024 ** 3
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def collect():
    return {"python": platform.python_version(), "python_tuple": tuple(sys.version_info[:2]),
            "platform": platform.platform(), "torch": torch_info(),
            "required": {label: module_version(name) for label, name in REQUIRED},
            "optional": {label: module_version(name) for label, name in OPTIONAL}}


def evaluate(info, require_cuda=True):
    """Return (exit_code, problems, warnings) for an info dict produced by collect()."""
    problems, warnings = [], []
    missing = [k for k, v in info["required"].items() if v is None]
    if info["torch"] is None:
        missing.insert(0, "torch")
    if missing:
        problems.append("missing required packages: " + ", ".join(missing))
    if tuple(info["python_tuple"]) < MIN_PYTHON:
        problems.append(f"Python {info['python']} < {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    code = 1 if problems else 0
    if info["torch"] is not None and not info["torch"]["cuda"]:
        msg = "CUDA is not available to torch"
        if require_cuda:
            problems.append(msg)
            code = code or 2
        else:
            warnings.append(msg)
    if info["torch"] is not None and info["torch"]["version"] != EXPECTED["torch"]:
        warnings.append(f"torch {info['torch']['version']} != expected {EXPECTED['torch']}")
    np_ver = info["required"].get("numpy")
    if np_ver is not None and np_ver != EXPECTED["numpy"]:
        warnings.append(f"numpy {np_ver} != expected {EXPECTED['numpy']}")
    return code, problems, warnings


def report(info):
    t = info["torch"]
    lines = [f"{'python':<12}{info['python']}  ({info['platform']})"]
    if t is None:
        lines.append(f"{'torch':<12}MISSING")
    else:
        lines.append(f"{'torch':<12}{t['version']}  (built for CUDA {t['cuda_build']})")
        lines.append(f"{'cuda':<12}{'available' if t['cuda'] else 'NOT available'}")
        if t["cuda"]:
            lines.append(f"{'gpu':<12}{t['gpu']}")
            lines.append(f"{'vram':<12}{t['vram_gb']:.2f} GB")
            lines.append(f"{'cudnn':<12}{t['cudnn']}")
    for label, ver in info["required"].items():
        lines.append(f"{label:<12}{ver if ver is not None else 'MISSING'}")
    for label, ver in info["optional"].items():
        lines.append(f"{label:<12}{ver if ver is not None else '-'}  (optional)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-cpu", action="store_true", help="do not fail when CUDA is unavailable")
    args = ap.parse_args(argv)
    info = collect()
    print(report(info))
    code, problems, warnings = evaluate(info, require_cuda=not args.allow_cpu)
    for w in warnings:
        print(f"WARNING: {w}")
    for p in problems:
        print(f"ERROR: {p}")
    print("OK" if code == 0 else f"FAILED (exit {code})")
    return code


if __name__ == "__main__":
    sys.exit(main())
