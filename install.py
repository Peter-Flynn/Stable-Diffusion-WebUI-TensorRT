import logging
import os
import re
import sys
from importlib import metadata

import launch

MIN_TRT = (11, 0)
logger = logging.getLogger("TensorRT")


def installed_tensorrt():
    found = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        match = re.fullmatch(r"(tensorrt(?:-cu\d+)?)(-bindings)?", name)
        if match:
            version = tuple(map(int, re.findall(r"\d+", dist.version)))
            found.append((version, match.group(2) is None, match.group(1), dist.version))  # highest version first, then full package over -bindings
    return max(found)[2:] if found else (None, None)


def _installed_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _tensorrt_package(fallback: str = "tensorrt") -> str:
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None) or ""
        major = int(cuda.split(".")[0]) if cuda else None
    except Exception:
        major = None
    return f"tensorrt-cu{major}" if major in (12, 13) else fallback


def _install_onnx():
    # Constrain already-installed numpy and protobuf so pip picks a compatible onnx release (e.g. 1.17.x)
    # instead of pulling ml-dtypes / protobuf 7 and upgrading reForge's pinned numpy 1.x or protobuf 3.x.
    np_ver = _installed_version("numpy")
    pb_ver = _installed_version("protobuf")
    np_pin = [f'"numpy=={np_ver}"'] if np_ver else []
    pb_pin = []
    if pb_ver:
        major = int(re.findall(r"\d+", pb_ver)[0])
        if major >= 4 or sys.version_info < (3, 13):
            pb_pin = [f'"protobuf<{major + 1}"']
    try:
        launch.run_pip(" ".join(["install", "onnx", *np_pin, *pb_pin]), "onnx", live=True)
    except Exception:
        if pb_pin:
            launch.run_pip(" ".join(["install", "onnx", *np_pin]), "onnx", live=True)
        else:
            raise


def install():
    if os.environ.get("TRT_EXTENSION_SKIP_INSTALL", "").lower() in ("1", "true", "yes"):
        return
    package, version = installed_tensorrt()
    if package is None:
        target = _tensorrt_package()
        launch.run_pip(f"install {target}", target, live=True)
    elif tuple(map(int, re.findall(r"\d+", version)[:2])) < MIN_TRT:
        target = _tensorrt_package(fallback=package)
        logger.info(f"Updating {package} {version} to {target}: this extension needs TensorRT {MIN_TRT[0]} or later.")
        launch.run_pip(f"install -U {target}", target, live=True)
    if not launch.is_installed("onnx"):
        _install_onnx()


try:
    install()
except Exception as e:
    logger.error(f"Couldn't install TensorRT: {e}")
