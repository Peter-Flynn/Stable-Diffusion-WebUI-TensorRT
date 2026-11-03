import functools

import torch


def log(message: str):
    print(f"[TensorRT] {message}", flush=True)


_notified = set()


def notify(message: str, once: bool = True):
    """Console and UI toast; `once` per generation."""
    if once and message in _notified:
        return
    _notified.add(message)
    log(message)
    try:
        import gradio as gr

        gr.Warning(f"TensorRT: {message}")
    except Exception:
        pass


def new_generation():
    _notified.clear()


def mm():
    import ldm_patched.modules.model_management as model_management

    return model_management


def inference_mode(fn):
    """reForge creates and moves its weights in inference mode; moving them outside it leaves weights
    restored after a LoRA unusable ("Inference tensors do not track version counter")."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapper


def gpu_tag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


MIN_TRT = 11


def trt_version() -> str:
    """The installed TensorRT's version; raises if it's missing or older than this extension needs."""
    import tensorrt

    if int(tensorrt.__version__.split(".")[0]) < MIN_TRT:
        raise RuntimeError(f"TensorRT {tensorrt.__version__} is installed; this extension needs {MIN_TRT} or later")
    return tensorrt.__version__


def trt_tag(version: str = None) -> str:
    return "trt" + ".".join((version or trt_version()).split(".")[:2])


def gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"
