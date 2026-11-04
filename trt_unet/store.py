"""Engines on disk: models/Unet-trt/<arch id>/<gpu>-<trt>-<key>.plan, with a .json describing it and, for
Speed Hacks (speed.py), the shallow UNet's engine as <...>.shallow.plan beside it. In
per-profile mode an architecture has an engine per profile (the key is the profile's); in merged mode, one
engine holding all of its profiles (the key is "merged-" and the profiles'). Plans hold no weights, so
they're small; weights come from whatever checkpoint is loaded. The mode is for every engine: it can only
change while there are none (models/Unet-trt/settings.json keeps the choice)."""

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

FORMAT = 2  # an engine holds one profile, or (with "mode": "merged") several
PER_PROFILE, MERGED = "per_profile", "merged"
SHALLOW = ".shallow.plan"
MODES = (PER_PROFILE, MERGED)


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


def merged_key(profiles: list) -> str:
    """A merged engine's key: its profiles', in any order (reordering them only renumbers them)."""
    return "merged-" + hashlib.sha1(json.dumps(sorted(profile_key(p) for p in profiles)).encode()).hexdigest()[:10]


def _settings_path() -> str:
    return os.path.join(root(), "settings.json")


def _saved_mode():
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            saved = json.load(f).get("mode")
        return saved if saved in MODES else None
    except (OSError, ValueError, AttributeError):
        return None


def mode() -> str:
    """The engine mode: that of the engines on disk if there are any, else the one chosen last, else per-profile."""
    found = {r.mode for r in records()}
    if len(found) == 1:
        return found.pop()
    return _saved_mode() or PER_PROFILE


def set_mode(new: str):
    """Only while there are no engines at all: the engines on disk decide the mode otherwise."""
    if new not in MODES:
        raise ValueError(f"Unknown engine mode {new!r}")
    if records():
        raise RuntimeError("The engine mode can only change while there are no engines: delete them all first.")
    with open(_settings_path() + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"mode": new}, f)
    os.replace(_settings_path() + ".tmp", _settings_path())


def _prefix(arch_id: str) -> str:
    """Where this GPU's and TensorRT version's engines for an architecture are, but for their profile key."""
    return os.path.join(root(), arch_id, f"{common.gpu_tag()}-{common.trt_tag()}")


@dataclass
class Record:
    """An engine: one profile of an architecture (per-profile mode), or all of them (merged mode). Its
    optimization profiles are in `profiles` order; `indexes` says where each is in the architecture's list."""

    plan: str
    meta: dict = field(repr=False)

    @property
    def arch(self) -> Arch:
        return Arch.from_dict(self.meta["arch"])

    @property
    def mode(self) -> str:
        return MERGED if self.meta.get("mode") == MERGED else PER_PROFILE

    @property
    def merged(self) -> bool:
        return self.mode == MERGED

    @property
    def profiles(self) -> list:
        return [Profile.from_dict(d) for d in (self.meta["profiles"] if self.merged else [self.meta["profile"]])]

    @property
    def profile_shapes(self) -> list:
        """Each profile's UNet input shapes."""
        return self.meta["shapes"] if self.merged else [self.meta["shapes"]]

    @property
    def indexes(self) -> list:
        return [int(i) for i in self.meta["indexes"]] if self.merged else [int(self.meta["index"])]

    @property
    def index(self) -> int:
        """Where its first profile is in the architecture's list."""
        return min(self.indexes)

    @property
    def engine_key(self) -> str:
        return merged_key(self.profiles) if self.merged else profile_key(self.profiles[0])

    @property
    def shallow_plan(self):
        name = (self.meta.get("speed_hacks") or {}).get("plan")
        return os.path.join(os.path.dirname(self.plan), name) if name else None

    @property
    def speed_hacks(self):
        """{"deep": the deep feature's channels and scale, "shapes": each profile's shallow UNet input shapes,
        "weights", "weight_bytes", ...}, or None if it was built without them (or before them)."""
        hacks = self.meta.get("speed_hacks")
        return hacks if hacks and os.path.isfile(self.shallow_plan) else None

    @property
    def outdated(self) -> bool:
        """Built before Speed Hacks: building again adds them. (One whose Speed Hacks part failed to build isn't.)"""
        return self.speed_hacks is None and "speed_hacks_error" not in self.meta

    @property
    def retry_speed_hacks(self) -> bool:
        """Its Speed Hacks part failed to build, and building again retries it, once: a failure that
        repeats likely always will."""
        return self.speed_hacks is None and "speed_hacks_error" in self.meta and not self.meta.get("speed_hacks_retried")

    @property
    def label(self) -> str:
        """For people: "engine for profile 2", "merged engine (3 profiles)"."""
        n = len(self.indexes)
        return f"merged engine ({n} profile{'' if n == 1 else 's'})" if self.merged else f"engine for profile {self.indexes[0] + 1}"

    def renumbered(self, indexes: list) -> dict:
        """Its meta with its profiles at these places in the architecture's list."""
        return {**self.meta, "indexes": list(indexes)} if self.merged else {**self.meta, "index": indexes[0]}

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
        total = 0
        for path in (self.plan, self.shallow_plan):
            with contextlib.suppress(OSError, TypeError):
                total += os.path.getsize(path)
        return total

    def describe(self) -> str:
        what = f"merged, {len(self.indexes)} profiles" if self.merged else self.profiles[0].describe()
        return f"{self.arch.label} · {what} · {self.meta['gpu']}, TensorRT {self.meta['trt']} · {common.gib(self.disk_bytes)}"


def _read(meta_path: str):
    plan = meta_path[: -len(".json")] + ".plan"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") == FORMAT and os.path.isfile(plan):
            record = Record(plan, meta)
            if len(record.profiles) != len(record.indexes) or len(record.profile_shapes) != len(record.indexes) or not record.indexes:
                raise ValueError("its profiles, their shapes and their indexes don't match")
            record.arch  # complete
            hacks = meta.get("speed_hacks")
            if hacks and (not isinstance(hacks.get("plan"), str) or len(hacks.get("shapes") or []) != len(record.indexes)):
                common.log(f"Ignoring the Speed Hacks part of {meta_path}: its plan or shapes are missing")
                meta.pop("speed_hacks")  # building again adds it
            return record
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


def slots(records: list) -> list:
    """(Record, optimization profile in it) for every profile these engines hold, in the architecture's
    profile order."""
    return sorted(((r, j) for r in records for j in range(len(r.indexes))), key=lambda s: s[0].indexes[s[1]])


def profiles_of(records: list) -> list:
    """The Profiles these engines hold, in the architecture's order."""
    return [r.profiles[j] for r, j in slots(records)]


def _write_meta(base: str, meta: dict):
    with open(base + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(base + ".json.tmp", base + ".json")


def save(arch: Arch, plan_tmp: str, meta: dict, shallow_tmp: str = None) -> Record:
    """meta has the mode and, per-profile, the profile (as a dict) and its index, or, merged, the profiles
    and their indexes, among the rest; with `shallow_tmp`, the shallow engine's "speed_hacks"."""
    record = Record("", {"format": FORMAT, "arch": arch.to_dict(), **meta})
    base = f"{_prefix(arch.id)}-{record.engine_key}"
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.remove(base + SHALLOW)  # an older build's
    if shallow_tmp:
        os.replace(shallow_tmp, base + SHALLOW)
        record.meta["speed_hacks"] = {**record.meta["speed_hacks"], "plan": os.path.basename(base) + SHALLOW}
    else:
        record.meta.pop("speed_hacks", None)
    os.replace(plan_tmp, base + ".plan")
    _write_meta(base, record.meta)
    return Record(base + ".plan", record.meta)


def wanted_keys(profiles: list, mode_: str) -> set:
    """The engine keys these profiles make in this mode."""
    return {merged_key(profiles)} if mode_ == MERGED else {profile_key(p) for p in profiles}


def keep_only(arch_id: str, profiles: list, mode_: str) -> int:
    """Deletes this GPU's and TensorRT version's engines for the architecture that these profiles don't
    make in this mode (older formats' files too) and puts the others' profiles in the list's order.
    Returns how many engines it deleted."""
    position = {profile_key(p): i for i, p in enumerate(profiles)}
    wanted = wanted_keys(profiles, mode_)
    prefix = _prefix(arch_id)
    deleted = 0
    for path in glob.glob(glob.escape(prefix) + ".plan") + glob.glob(glob.escape(prefix) + "-*.plan"):
        if path.endswith(SHALLOW):  # goes with its engine
            continue
        base = path[: -len(".plan")]
        key = base[len(prefix) + 1:] if base != prefix else None
        if key in wanted:
            record = _read(base + ".json")
            if record is not None:
                indexes = [position[profile_key(p)] for p in record.profiles]
                if indexes != record.indexes:
                    _write_meta(base, record.renumbered(indexes))
            continue
        for stale in (path, base + ".json", base + SHALLOW):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        deleted += 1
    return deleted


def delete(group: list):
    """Deletes these engines (Records)."""
    for record in group:
        for path in (record.plan, record.plan[: -len(".plan")] + ".json", record.plan[: -len(".plan")] + SHALLOW):
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
    for folder in {os.path.dirname(r.plan) for r in group}:
        if not glob.glob(os.path.join(folder, "*.plan")):
            shutil.rmtree(folder, ignore_errors=True)
