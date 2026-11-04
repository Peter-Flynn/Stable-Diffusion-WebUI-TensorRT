"""Engines on disk, one per profile: models/Unet-trt/<arch id>/<gpu>-<trt>-<profile key>.plan, with a .json
describing it. Plans hold no weights, so they're small; weights come from whatever checkpoint is loaded."""

import contextlib
import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field

from . import common
from .arch import Arch
from .profiles import Profile

FORMAT = 2  # one profile per engine


def root() -> str:
    from modules import paths_internal

    path = os.path.join(paths_internal.models_path, "Unet-trt")
    os.makedirs(path, exist_ok=True)
    return path


def timing_cache_path() -> str:
    return os.path.join(root(), f"timing-{common.gpu_tag()}-{common.trt_tag()}.cache")


def work_dir(arch: Arch) -> str:
    return os.path.join(root(), arch.id, ".build")


def profile_key(profile: Profile) -> str:
    return hashlib.sha1(json.dumps(profile.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


def _prefix(arch_id: str) -> str:
    """Where this GPU's and TensorRT version's engines for an architecture are, but for their profile key."""
    return os.path.join(root(), arch_id, f"{common.gpu_tag()}-{common.trt_tag()}")


@dataclass
class Record:
    """An engine: one profile of an architecture."""

    plan: str
    meta: dict = field(repr=False)

    @property
    def arch(self) -> Arch:
        return Arch.from_dict(self.meta["arch"])

    @property
    def profile(self) -> Profile:
        return Profile.from_dict(self.meta["profile"])

    @property
    def shapes(self) -> dict:
        """The profile's UNet input shapes."""
        return self.meta["shapes"]

    @property
    def index(self) -> int:
        """Where its profile is in the architecture's list."""
        return int(self.meta["index"])

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
    def group(self) -> tuple:
        """The engines built together: one architecture's, for one GPU and TensorRT version."""
        return self.meta["arch"]["id"], self.meta["gpu"], common.trt_tag(self.meta["trt"])

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
        return f"{self.arch.label} · {self.profile.describe()} · {self.meta['gpu']}, TensorRT {self.meta['trt']} · {common.gib(self.disk_bytes)}"


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
    """Every engine, each architecture's in its profiles' order."""
    found = [r for r in (_read(p) for p in glob.glob(os.path.join(root(), "*", "*.json"))) if r is not None]
    return sorted(found, key=lambda r: (r.group, r.index))


def engines(arch_id: str) -> list:
    """This GPU's and TensorRT version's engines for an architecture, in their profiles' order."""
    return [r for r in records() if r.meta["arch"]["id"] == arch_id and r.loadable]


def _write_meta(base: str, meta: dict):
    with open(base + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(base + ".json.tmp", base + ".json")


def save(arch: Arch, plan_tmp: str, meta: dict) -> Record:
    """meta has the profile (as a dict) and its index, among the rest."""
    base = f"{_prefix(arch.id)}-{profile_key(Profile.from_dict(meta['profile']))}"
    os.makedirs(os.path.dirname(base), exist_ok=True)
    meta = {"format": FORMAT, "arch": arch.to_dict(), **meta}
    os.replace(plan_tmp, base + ".plan")
    _write_meta(base, meta)
    return Record(base + ".plan", meta)


def keep_only(arch_id: str, profiles: list) -> int:
    """Deletes this GPU's and TensorRT version's engines for the architecture whose profile isn't in
    `profiles` (older formats' files too) and puts the others in the list's order. Returns how many
    engines it deleted."""
    order = {profile_key(p): i for i, p in enumerate(profiles)}
    prefix = _prefix(arch_id)
    deleted = 0
    for path in glob.glob(glob.escape(prefix) + ".plan") + glob.glob(glob.escape(prefix) + "-*.plan"):
        base = path[: -len(".plan")]
        key = base[len(prefix) + 1:] if base != prefix else None
        if key in order:
            record = _read(base + ".json")
            if record is not None and record.index != order[key]:
                _write_meta(base, {**record.meta, "index": order[key]})
            continue
        for stale in (path, base + ".json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        deleted += 1
    return deleted


def delete(group: list):
    """Deletes these engines (Records)."""
    for record in group:
        for path in (record.plan, record.plan[: -len(".plan")] + ".json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
    for folder in {os.path.dirname(r.plan) for r in group}:
        if not glob.glob(os.path.join(folder, "*.plan")):
            shutil.rmtree(folder, ignore_errors=True)
