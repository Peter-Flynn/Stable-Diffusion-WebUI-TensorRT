# reForge runs every extension's preload.py before any extension's scripts. Importing tensorrt here, first,
# binds its Python bindings to this extension's own libnvinfer. If another extension (e.g. sd-forge-nvidia-vfx,
# with TensorRT 10) loads a different libnvinfer into the process before that import, the bindings resolve to
# it instead, and no engine deserializes.
import importlib

try:
    importlib.import_module("tensorrt")
except Exception:
    pass
