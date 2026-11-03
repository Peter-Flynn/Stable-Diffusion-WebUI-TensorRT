import os
import re
from importlib import metadata

import launch

MIN_TRT = (10, 1)
NVIDIA_INDEX = "--extra-index-url https://pypi.nvidia.com"


def installed_tensorrt():
    """(distribution, version) of the installed TensorRT, preferring the meta package, or (None, None)."""
    found = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name == "tensorrt" or re.fullmatch(r"tensorrt-cu\d+(-bindings)?", name):
            found.append((name, dist.version))
    found.sort(key=lambda x: (x[0] != "tensorrt", x[0]))
    return found[0] if found else (None, None)


def install():
    if os.environ.get("TRT_EXTENSION_SKIP_INSTALL", "").lower() in ("1", "true", "yes"):
        return
    name, version = installed_tensorrt()
    if name is None:
        print("[TensorRT] Installing TensorRT...")
        launch.run_pip(f"install tensorrt {NVIDIA_INDEX}", "tensorrt", live=True)
    elif tuple(map(int, re.findall(r"\d+", version)[:2])) < MIN_TRT:
        print(f"[TensorRT] {name} {version} is too old; this extension needs TensorRT {'.'.join(map(str, MIN_TRT))} or later (pip install -U tensorrt {NVIDIA_INDEX}).")
    if not launch.is_installed("onnx"):
        launch.run_pip("install onnx", "onnx", live=True)


try:
    install()
except Exception as e:
    print(f"[TensorRT] Installing dependencies failed: {e}")
