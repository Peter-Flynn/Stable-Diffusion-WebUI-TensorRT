import os
import re
import sys
from importlib import metadata

python = sys.executable

# Provide fallback for standalone execution outside WebUI
try:
    import launch
except ImportError:
    import subprocess

    class MockLaunch:
        @staticmethod
        def is_installed(package):
            try:
                metadata.version(package)
                return True
            except Exception:
                return False

        @staticmethod
        def run_pip(args, desc="", live=True):
            print(f"[TensorRT Extension] Running pip {args} ({desc})")
            subprocess.run([sys.executable, "-m", "pip"] + args.split(), check=True)

        @staticmethod
        def run(cmd, desc=""):
            print(f"[TensorRT Extension] Running command: {cmd} ({desc})")
            subprocess.run(cmd, check=True)

    launch = MockLaunch()


MIN_TRT_VERSION = (10, 0)
NVIDIA_INDEX = "--extra-index-url https://pypi.nvidia.com"


def parse_version(v_str):
    parts = [int(p) for p in re.findall(r"\d+", v_str or "")]
    return tuple((parts + [0, 0, 0])[:3])


def installed_tensorrt():
    """Return (distribution, version) of any installed TensorRT Python package
    (tensorrt, tensorrt-cu12, tensorrt-cu13, ...), or (None, None)."""
    found = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name == "tensorrt" or re.fullmatch(r"tensorrt-cu\d+(-bindings)?", name):
            found.append((name, dist.version))
    if not found:
        return None, None
    # Prefer the meta package if present.
    found.sort(key=lambda x: (x[0] != "tensorrt", x[0]))
    return found[0]


def install():
    if os.environ.get("TRT_EXTENSION_SKIP_INSTALL", "").lower() in ("1", "true", "yes"):
        print("[TensorRT Extension] TRT_EXTENSION_SKIP_INSTALL is set; not installing anything.")
        return

    # TensorRT: never uninstall or replace a user-installed version.
    trt_pkg, trt_ver = installed_tensorrt()
    if trt_pkg is None:
        print("[TensorRT Extension] TensorRT is not installed. Installing the latest release...")
        launch.run_pip(f"install tensorrt {NVIDIA_INDEX}", "tensorrt", live=True)
    elif parse_version(trt_ver) < MIN_TRT_VERSION:
        print(
            f"[TensorRT Extension] Warning: {trt_pkg}=={trt_ver} is older than "
            f"{'.'.join(map(str, MIN_TRT_VERSION))} and is not supported. Please upgrade it "
            f"(e.g. pip install -U tensorrt {NVIDIA_INDEX})."
        )

    # Required for ONNX export.
    for pkg in ("onnx", "onnxscript"):
        if not launch.is_installed(pkg):
            launch.run_pip(f"install {pkg}", pkg, live=True)

    # Optional ONNX graph cleanup / constant folding before the TensorRT build.
    if not launch.is_installed("onnx-graphsurgeon"):
        launch.run_pip(f"install onnx-graphsurgeon {NVIDIA_INDEX}", "onnx-graphsurgeon", live=True)
    if not launch.is_installed("polygraphy"):
        launch.run_pip(f"install polygraphy {NVIDIA_INDEX}", "polygraphy", live=True)


if __name__ == "__main__" or "launch" in sys.modules:
    try:
        install()
    except Exception as e:
        print(f"[TensorRT Extension] Dependency installation failed: {e}")
