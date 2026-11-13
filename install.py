import logging
import os
import re
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
            found.append((match.group(2) is not None, match.group(1), dist.version))  # False first: min() prefers a package over its -bindings
    return min(found)[1:] if found else (None, None)


def install():
    if os.environ.get("TRT_EXTENSION_SKIP_INSTALL", "").lower() in ("1", "true", "yes"):
        return
    package, version = installed_tensorrt()
    if package is None:
        launch.run_pip("install tensorrt", "tensorrt", live=True)
    elif tuple(map(int, re.findall(r"\d+", version)[:2])) < MIN_TRT:
        logger.info(f"Updating {package} {version}: this extension needs TensorRT {MIN_TRT[0]} or later.")
        launch.run_pip(f"install -U {package}", package, live=True)
    if not launch.is_installed("onnx"):
        launch.run_pip("install onnx", "onnx", live=True)


try:
    install()
except Exception as e:
    logger.error(f"Couldn't install TensorRT: {e}")
