"""Engines on disk: models/Unet-trt/<arch id>/<gpu>-<trt>.plan, with a .json describing it. Plans hold
no weights, so they're small; weights come from whatever checkpoint is loaded."""

import glob
import json
import os
import shutil
from dataclasses import dataclass, field

from . import common
from .arch import Arch
from .profiles import Profile

FORMAT = 1


def root() -> str:
    from modules import paths_internal

    path = os.path.join(paths_internal.models_path, "Unet-trt")
    os.makedirs(path, exist_ok=True)
    return path


def timing_cache_path() -> str:
    return os.path.join(root(), f"timing-{common.gpu_tag()}-{common.trt_tag()}.cache")


def work_dir(arch: Arch) -> str:
    return os.path.join(root(), arch.id, ".build")


@dataclass
class Record:
    plan: str
    meta: dict = field(repr=False)

    @property
    def arch(self) -> Arch:
        return Arch.from_dict(self.meta["arch"])

    @property
    def profiles(self) -> list:
        return [Profile.from_dict(p) for p in self.meta["profiles"]]

    @property
    def shapes(self) -> list:
        return self.meta["shapes"]

    @property
    def weight_bytes(self) -> int:
        return int(self.meta["weight_bytes"])

    @property
    def weight_names(self) -> list:
        return self.meta["weights"]

    @property
    def key(self) -> str:
        return self.plan

    @property
    def loadable(self) -> bool:
        try:
            return self.meta["gpu"] == common.gpu_tag() and common.trt_tag(self.meta["trt"]) == common.trt_tag()
        except Exception:
            return False

    @property
    def disk_bytes(self) -> int:
        try:
            return os.path.getsize(self.plan)
        except OSError:
            return 0

    def describe(self) -> str:
        return f"{self.arch.label} · {self.meta['gpu']}, TensorRT {self.meta['trt']} · {len(self.meta['profiles'])} profiles · {common.gib(self.disk_bytes)}"


def _read(meta_path: str):
    plan = meta_path[: -len(".json")] + ".plan"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") == FORMAT and os.path.isfile(plan):
            return Record(plan, meta)
    except Exception as e:
        common.log(f"Ignoring {meta_path}: {e}")
    return None


def records() -> list:
    found = (_read(p) for p in sorted(glob.glob(os.path.join(root(), "*", "*.json"))))
    return [r for r in found if r is not None]


def find(arch_id: str):
    """This GPU's and TensorRT version's engine for an architecture."""
    return next((r for r in records() if r.meta["arch"]["id"] == arch_id and r.loadable), None)


def save(arch: Arch, plan_tmp: str, meta: dict) -> Record:
    folder = os.path.join(root(), arch.id)
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, f"{common.gpu_tag()}-{common.trt_tag()}")
    meta = {"format": FORMAT, "arch": arch.to_dict(), **meta}
    with open(base + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(plan_tmp, base + ".plan")
    os.replace(base + ".json.tmp", base + ".json")
    return Record(base + ".plan", meta)


def delete(record: Record):
    for path in (record.plan, record.plan[: -len(".plan")] + ".json"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    folder = os.path.dirname(record.plan)
    if not glob.glob(os.path.join(folder, "*.plan")):
        shutil.rmtree(folder, ignore_errors=True)
