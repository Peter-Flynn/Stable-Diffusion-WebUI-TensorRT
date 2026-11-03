import os
import re
from importlib import metadata

import launch

MIN_TRT = (10, 1)


def torch_cuda() -> str:
    """PyTorch's CUDA major version ("12"), or None."""
    try:
        import torch

        return torch.version.cuda.split(".")[0] if torch.version.cuda else None
    except Exception:
        return None


def installed_tensorrt():
    """(version, {CUDA majors of the installed TensorRT libraries}), or (None, set())."""
    version, cudas = None, set()
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name == "tensorrt" or re.fullmatch(r"tensorrt-cu\d+(-bindings)?", name):
            version = version or dist.version
        match = re.fullmatch(r"tensorrt-cu(\d+)-libs", name)
        if match:
            cudas.add(match.group(1))
    return version, cudas


def install():
    if os.environ.get("TRT_EXTENSION_SKIP_INSTALL", "").lower() in ("1", "true", "yes"):
        return
    cuda = torch_cuda()
    # The plain `tensorrt` package is the newest CUDA's build (13.x), not necessarily PyTorch's.
    package = f"tensorrt-cu{cuda}" if cuda else "tensorrt"
    version, cudas = installed_tensorrt()
    if version is None:
        print(f"[TensorRT] Installing {package}...")
        launch.run_pip(f"install {package}", package, live=True)
    elif tuple(map(int, re.findall(r"\d+", version)[:2])) < MIN_TRT:
        print(f"[TensorRT] TensorRT {version} is too old; this extension needs {'.'.join(map(str, MIN_TRT))} or later (pip install -U {package}).")
    elif cuda and cudas and cuda not in cudas:
        others = " ".join(f"tensorrt-cu{c}{part}" for c in sorted(cudas) for part in ("", "-libs", "-bindings"))
        print(
            f"[TensorRT] The installed TensorRT is built for CUDA {'/'.join(sorted(cudas))}, PyTorch for CUDA {cuda}. "
            f"If engines fail to build or load: pip uninstall -y tensorrt {others}, then pip install {package}"
        )
    if not launch.is_installed("onnx"):
        launch.run_pip("install onnx", "onnx", live=True)


try:
    install()
except Exception as e:
    print(f"[TensorRT] Installing dependencies failed: {e}")
