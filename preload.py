"""reForge runs every extension's preload.py before any extension's scripts. TensorRT's Python bindings look up
TensorRT's functions once, when they're imported, and take a TensorRT that's visible to the whole process over
their own: another extension that loads one that way first (e.g. sd-forge-nvidia-vfx, with TensorRT 10) would
leave this extension calling it, and no engine would load. Importing TensorRT here binds the bindings to their
own TensorRT first. It stays private to them, so other extensions keep using theirs."""

import importlib

try:
    importlib.import_module("tensorrt")
except Exception:  # not installed yet: install.py installs it, and the tab says so
    pass
