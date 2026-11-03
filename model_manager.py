import json
import os

import torch

from modules import paths_internal

from datastructures import ModelConfig, ModelConfigEncoder

ONNX_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-onnx")
TRT_MODEL_DIR = os.path.join(paths_internal.models_path, "Unet-trt")
MODEL_FILE = os.path.join(TRT_MODEL_DIR, "model.json")
os.makedirs(ONNX_MODEL_DIR, exist_ok=True)
os.makedirs(TRT_MODEL_DIR, exist_ok=True)

try:
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
except Exception as e:
    print(f"[TensorRT] Could not determine the GPU's compute capability ({e}); assuming 12.0.", flush=True)
    cc_major, cc_minor = 12, 0
CC = f"cc{cc_major}{cc_minor}"


def onnx_path(model_name: str) -> str:
    return os.path.join(ONNX_MODEL_DIR, f"{model_name}.onnx")


def trt_filename(model_name: str, model_hash: str, profile: dict, static_shapes: bool) -> str:
    n = 1 if static_shapes else 3
    shapes = "-".join(f"{k}=" + "+".join("x".join(map(str, v[i])) for i in range(n)) for k, v in profile.items())
    return "_".join([model_name, model_hash, CC, shapes]) + ".trt"


def weights_map_path(model_name: str) -> str:
    return os.path.join(TRT_MODEL_DIR, f"{model_name}_weights_map_v2.json")


def timing_cache_path() -> str:
    platform = "win" if os.name == "nt" else "linux"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_caches", f"timing_cache_{platform}_{CC}.cache")


def _same_release(a: str, b: str) -> bool:
    return a.split(".")[:2] == b.split(".")[:2]


class ModelManager:
    def __init__(self, model_file=MODEL_FILE):
        self.model_file = model_file
        try:
            import tensorrt

            self.trt_version = tensorrt.__version__
        except Exception:
            self.trt_version = ""
        self._reported = set()
        self.all_models = self._read() if os.path.exists(model_file) else {}
        self._prune()

    def _read(self):
        with open(self.model_file, encoding="utf-8") as f:
            data = json.load(f)
        for models in data.values():
            for entries in models.values():
                for entry in entries:
                    entry["config"] = ModelConfig.from_dict(entry["config"])
        return data

    def _prune(self):
        """Drops entries whose engine file is gone, and duplicates."""
        engines = {f for f in os.listdir(TRT_MODEL_DIR) if f.endswith(".trt")}
        for models in self.all_models.values():
            for name in list(models):
                kept = {e["filepath"]: e for e in models[name] if e["filepath"] in engines}
                if kept:
                    models[name] = list(kept.values())
                else:
                    del models[name]
        self.write_json()

    def write_json(self):
        tmp = self.model_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.all_models, f, indent=4, cls=ModelConfigEncoder)
        os.replace(tmp, self.model_file)

    def add_entry(self, model_name: str, filepath: str, config: ModelConfig):
        entries = self.all_models.setdefault(CC, {}).setdefault(model_name, [])
        entry = {"filepath": filepath, "config": config}
        index = next((i for i, e in enumerate(entries) if e["filepath"] == filepath), None)
        if index is None:
            entries.append(entry)
        else:
            entries[index] = entry  # profile numbers are part of SD Unet labels
        self.write_json()

    def has_engine(self, filepath: str) -> bool:
        return any(m["filepath"] == filepath for _, _, m in self.all_engines())

    def get_entry(self, filepath: str):
        """This GPU's entry for an engine file, or None."""
        return next((e for entries in self.all_models.get(CC, {}).values() for e in entries if e["filepath"] == filepath), None)

    def _loadable(self, entry) -> bool:
        built_with = entry["config"].trt_version
        if entry["config"].lora:
            return False
        if not built_with or not self.trt_version or _same_release(built_with, self.trt_version):
            return True
        if entry["filepath"] not in self._reported:
            self._reported.add(entry["filepath"])
            print(
                f"[TensorRT] Ignoring {entry['filepath']}: built with TensorRT {built_with}, installed is "
                f"{self.trt_version}. Rebuild it from the TensorRT tab.",
                flush=True,
            )
        return False

    def available_models(self) -> dict:
        """{model name: [entries]} for the engines this GPU and TensorRT version can load."""
        available = {}
        for name, entries in self.all_models.get(CC, {}).items():
            usable = [e for e in entries if self._loadable(e)]
            if usable:
                available[name] = usable
        return available

    def all_engines(self):
        for cc, models in self.all_models.items():
            for name, entries in models.items():
                for entry in entries:
                    yield cc, name, entry

    def delete_engine(self, filepath: str) -> bool:
        found = False
        for models in self.all_models.values():
            for name in list(models):
                kept = [e for e in models[name] if e["filepath"] != filepath]
                if len(kept) != len(models[name]):
                    found = True
                    if kept:
                        models[name] = kept
                    else:
                        del models[name]
        path = os.path.join(TRT_MODEL_DIR, filepath)
        if os.path.basename(filepath) == filepath and os.path.isfile(path):
            os.remove(path)
            found = True
        self.write_json()
        return found

    @staticmethod
    def available_loras() -> dict:
        return {f[:-5]: os.path.join(TRT_MODEL_DIR, f) for f in os.listdir(TRT_MODEL_DIR) if f.endswith(".lora")}


modelmanager = ModelManager()
