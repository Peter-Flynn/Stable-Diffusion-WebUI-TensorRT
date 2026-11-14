# Engines on disk: models/Unet-trt/<arch id>/<gpu>-<trt>-<key>.plan, a .json of its metadata beside it, and for Speed
# Hacks the shallow UNet's engine (with fresh.py's graph) as <...>.shallow.plan. Per-profile mode: one engine per
# profile, keyed by the profile. Merged mode: one engine holding all the architecture's profiles, keyed "merged-" + a
# hash of their keys. A record's "index"/"indexes" are its profiles' positions in the architecture's profile list. Plans
# hold no weights: they're refitted from the loaded checkpoint. The mode applies to every engine and can only change
# while none exist (models/Unet-trt/settings.json).
import contextlib
import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field

from . import common, fresh
from .arch import Arch
from .profiles import Profile

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
    found = {r.mode for r in records()}
    if len(found) == 1:
        return found.pop()
    return _saved_mode() or PER_PROFILE


def set_mode(new: str):
    if new not in MODES:
        raise ValueError(f"Unknown engine mode {new!r}")
    if records():
        raise RuntimeError("The engine mode can only change while there are no engines: delete them all first.")
    with open(_settings_path() + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"mode": new}, f)
    os.replace(_settings_path() + ".tmp", _settings_path())


def _prefix(arch_id: str) -> str:
    return os.path.join(root(), arch_id, f"{common.gpu_tag()}-{common.trt_tag()}")


@dataclass
class Record:
    plan: str
    meta: dict = field(repr=False)

    @property
    def arch(self) -> Arch:
        return Arch.from_dict(self.meta["arch"])

    @property
    def mode(self) -> str:
        return self.meta["mode"]

    @property
    def merged(self) -> bool:
        return self.mode == MERGED

    @property
    def profiles(self) -> list:
        return [Profile.from_dict(d) for d in (self.meta["profiles"] if self.merged else [self.meta["profile"]])]

    @property
    def profile_shapes(self) -> list:
        return self.meta["shapes"] if self.merged else [self.meta["shapes"]]

    @property
    def indexes(self) -> list:
        return [int(i) for i in self.meta["indexes"]] if self.merged else [int(self.meta["index"])]

    @property
    def index(self) -> int:
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
        hacks = self.meta.get("speed_hacks")
        return hacks if hacks and hacks.get("fresh") == fresh.SIGNATURE else None

    @property
    def stale_speed_hacks(self) -> bool:  # built with another fresh.SIGNATURE, so it takes other inputs
        return self.speed_hacks is None and "speed_hacks" in self.meta

    @property
    def retry_speed_hacks(self) -> bool:
        return self.speed_hacks is None and (self.stale_speed_hacks or not self.meta.get("speed_hacks_retried"))

    @property
    def label(self) -> str:
        n = len(self.indexes)
        return f"merged engine ({n} profile{'' if n == 1 else 's'})" if self.merged else f"engine for profile {self.indexes[0] + 1}"

    def renumbered(self, indexes: list) -> dict:
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


def _read(meta_path: str):
    plan = meta_path[: -len(".json")] + ".plan"
    try:
        record = Record(plan, common.read_json(meta_path))
        if os.path.isfile(plan) and (record.shallow_plan is None or os.path.isfile(record.shallow_plan)):
            record.arch
            return record
    except Exception as e:
        common.logger.debug(f"Ignoring {meta_path}, which couldn't be read: {e}")
    return None


def records() -> list:
    found = [r for r in (_read(p) for p in glob.glob(os.path.join(root(), "*", "*.json"))) if r is not None]
    return sorted(found, key=lambda r: (r.group, r.index))


def engines(arch_id: str) -> list:
    return [r for r in records() if r.meta["arch"]["id"] == arch_id and r.loadable]


def slots(records: list) -> list:
    return sorted(((r, j) for r in records for j in range(len(r.indexes))), key=lambda s: s[0].indexes[s[1]])


def profiles_of(records: list) -> list:
    return [r.profiles[j] for r, j in slots(records)]


def _write_meta(base: str, meta: dict):
    with open(base + ".json.tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(base + ".json.tmp", base + ".json")


def save(arch: Arch, plan_tmp: str, meta: dict, shallow_tmp: str = None) -> Record:
    record = Record("", {"arch": arch.to_dict(), **meta})
    base = f"{_prefix(arch.id)}-{record.engine_key}"
    os.makedirs(os.path.dirname(base), exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.remove(base + SHALLOW)
    if shallow_tmp:
        os.replace(shallow_tmp, base + SHALLOW)
        record.meta["speed_hacks"] = {**record.meta["speed_hacks"], "plan": os.path.basename(base) + SHALLOW}
    else:
        record.meta.pop("speed_hacks", None)
    os.replace(plan_tmp, base + ".plan")
    _write_meta(base, record.meta)
    return Record(base + ".plan", record.meta)


def wanted_keys(profiles: list, mode_: str) -> set:
    return {merged_key(profiles)} if mode_ == MERGED else {profile_key(p) for p in profiles}


def keep_only(arch_id: str, profiles: list, mode_: str) -> int:
    position = {profile_key(p): i for i, p in enumerate(profiles)}
    wanted = wanted_keys(profiles, mode_)
    prefix = _prefix(arch_id)
    deleted = 0
    for path in glob.glob(glob.escape(prefix) + "-*.plan"):
        if path.endswith(SHALLOW):
            continue
        base = path[: -len(".plan")]
        if base[len(prefix) + 1:] in wanted:
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
    for record in group:
        base = record.plan[: -len(".plan")]
        for path in (record.plan, base + ".json", base + SHALLOW):
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
    for folder in {os.path.dirname(r.plan) for r in group}:
        if not glob.glob(os.path.join(folder, "*.plan")):
            shutil.rmtree(folder, ignore_errors=True)
