import os
import re
from importlib import metadata

import launch

MIN_TRT_VERSION = (10, 0)
NVIDIA_INDEX = "--extra-index-url https://pypi.nvidia.com"


def installed_tensorrt():
    """(distribution, version) of an installed TensorRT package (tensorrt, tensorrt-cu12, ...), preferring
    the meta package, or (None, None)."""
    found = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name == "tensorrt" or re.fullmatch(r"tensorrt-cu\d+(-bindings)?", name):
            found.append((name, dist.version))
    found.sort(key=lambda x: (x[0] != "tensorrt", x[0]))
    return found[0] if found else (None, None)


def install():
    if os.environ.get("TRT_EXTENSION_SKIP_INSTALL", "").lower() in ("1", "true", "yes"):
        print("[TensorRT Extension] TRT_EXTENSION_SKIP_INSTALL is set; not installing anything.")
        return

    # Never uninstall or replace a TensorRT the user installed.
    trt_pkg, trt_ver = installed_tensorrt()
    if trt_pkg is None:
        print("[TensorRT Extension] TensorRT is not installed. Installing the latest release...")
        launch.run_pip(f"install tensorrt {NVIDIA_INDEX}", "tensorrt", live=True)
    elif tuple(map(int, re.findall(r"\d+", trt_ver)[:2])) < MIN_TRT_VERSION:
        print(
            f"[TensorRT Extension] Warning: {trt_pkg}=={trt_ver} is older than {'.'.join(map(str, MIN_TRT_VERSION))} and "
            f"isn't supported. Please upgrade it (pip install -U tensorrt {NVIDIA_INDEX})."
        )

    for pkg in ("onnx", "onnxscript"):
        if not launch.is_installed(pkg):
            launch.run_pip(f"install {pkg}", pkg, live=True)
    for pkg in ("onnx-graphsurgeon", "polygraphy"):  # optional
        if not launch.is_installed(pkg):
            launch.run_pip(f"install {pkg} {NVIDIA_INDEX}", pkg, live=True)


try:
    install()
except Exception as e:
    print(f"[TensorRT Extension] Dependency installation failed: {e}")
