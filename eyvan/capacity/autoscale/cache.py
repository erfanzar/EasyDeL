# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owner-only local persistence for autoscale defaults and generated fleets."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import typing as tp
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.autoscale.models import (
    DEFAULT_AUTOSCALE_CACHE_ROOT,
    AutoscaleDefaults,
    CachedAutoscaleConfig,
    GeneratedAutoscaleConfig,
    GeneratedZoneAutoscaleConfig,
    GenerationQuotaSnapshot,
    QueueAdmissionDenial,
    QueueAdmissionDenialKey,
    QuotaSnapshot,
)
from eyvan.capacity.fleet_config import EyvanFleetConfig, NodeType, TpuNodeType, validate_generated_fleet
from eyvan.records import CapacityClass

GeneratedAutoscale = GeneratedAutoscaleConfig | GeneratedZoneAutoscaleConfig


class AutoscaleDefaultsRepository:
    """Read and write the user-owned defaults YAML with stable permissions."""

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()

    def load(self) -> AutoscaleDefaults:
        """Load defaults, returning an empty editable profile when absent."""

        if not self.path.exists():
            return AutoscaleDefaults()
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"invalid autoscale defaults {self.path}: {exc}") from exc
        if not is_json_object(raw):
            raise ValueError(f"invalid autoscale defaults {self.path}: autoscale defaults YAML must be a mapping")
        try:
            return AutoscaleDefaults.from_document(raw)
        except ValueError as exc:
            raise ValueError(f"invalid autoscale defaults {self.path}: {exc}") from exc

    def save(self, defaults: AutoscaleDefaults) -> Path:
        """Atomically write one owner-only defaults document."""

        create_private_directory(self.path.parent)
        _atomic_write(self.path, yaml.safe_dump(defaults.as_document(), sort_keys=False), mode=0o600)
        return self.path


class GeneratedAutoscaleCache:
    """Persist generated fleet YAML files and their non-secret index metadata."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.configs_dir = self.root / "configs"
        self.index_path = self.root / "index.json"
        self.index_lock_path = self.root / ".index.lock"

    def store(
        self,
        generated: GeneratedAutoscale,
        *,
        refresh_generations: tuple[str, ...] = (),
        preserve_types: tuple[str, ...] = (),
    ) -> CachedAutoscaleConfig:
        """Persist a refresh, preserving out-of-scope pools under the index lock."""

        create_private_directory(self.configs_dir)
        try:
            os.chmod(self.root, 0o700)
            os.chmod(self.configs_dir, 0o700)
        except OSError:
            pass
        path = self.configs_dir / f"{generated.fleet_name}.yaml"
        with _exclusive_file_lock(self.index_lock_path):
            metadata = self._load_index()
            generated_at = datetime.now(UTC).isoformat()
            document = generated.document
            info = _metadata_for(generated, generated_at=generated_at, path=path)
            if isinstance(generated, GeneratedZoneAutoscaleConfig) and (refresh_generations or preserve_types):
                previous, previous_document = _load_fleet_config_document(path)
                if (
                    previous is not None
                    and previous_document is not None
                    and previous.provider == generated.config.provider
                ):
                    previous_types = {
                        name: pool.accelerator_type
                        for name, pool in previous.pools.items()
                        if isinstance(pool, TpuNodeType)
                    }
                    retained = {
                        name: pool
                        for name, pool in _document_pools(previous_document).items()
                        if name in previous_types
                        and (
                            previous_types[name] in preserve_types
                            or (
                                refresh_generations and previous_types[name].partition("-")[0] not in refresh_generations
                            )
                        )
                        and name not in _document_pools(document)
                    }
                    # A denial retains a flavor, not an independent historical
                    # quota budget. Reconcile its shared group against the fresh
                    # observation before validation or either atomic write. Groups
                    # absent from this refresh keep their existing declarations.
                    current_limits: dict[str, int] = {}
                    for pool in _document_pools(document).values():
                        group = pool.get("quota_group")
                        limit = pool.get("quota_limit")
                        if isinstance(group, str) and isinstance(limit, int) and not isinstance(limit, bool):
                            current_limits[group] = min(current_limits.get(group, limit), limit)
                    retained = {
                        name: {**pool, "quota_limit": current_limits[group]}
                        if isinstance(group := pool.get("quota_group"), str) and group in current_limits
                        else pool
                        for name, pool in retained.items()
                    }
                    document = {**document, "pools": {**retained, **_document_pools(document)}}
                    validate_generated_fleet(document)
                    info["pools"] = {
                        name: {"max_workers": _pool_max_workers(pool)}
                        for name, pool in _document_pools(document).items()
                    }
                    retained_generations = {previous_types[name].partition("-")[0] for name in retained}
                    old_info = metadata.get(generated.fleet_name)
                    old_quotas = old_info.get("generation_quotas") if isinstance(old_info, dict) else None
                    new_quotas = info["generation_quotas"]
                    if isinstance(old_quotas, list) and isinstance(new_quotas, list):
                        new_generations = {quota.get("generation") for quota in new_quotas if isinstance(quota, dict)}
                        info["generation_quotas"] = new_quotas + [
                            quota
                            for quota in old_quotas
                            if isinstance(quota, dict)
                            and quota.get("generation") in retained_generations
                            and quota.get("generation") not in new_generations
                        ]
            _atomic_write(path, yaml.safe_dump(document, sort_keys=False), mode=0o600)
            metadata[generated.fleet_name] = info
            _atomic_write(self.index_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n", mode=0o600)
        return _cached_from_generated(generated, path=path, generated_at=generated_at)

    def preview(self, generated: GeneratedAutoscale) -> CachedAutoscaleConfig:
        """Return a cache-shaped generated entry without writing local state."""

        return _cached_from_generated(
            generated,
            path=self.configs_dir / f"{generated.fleet_name}.yaml",
            generated_at=None,
        )

    def list(self) -> tuple[CachedAutoscaleConfig, ...]:
        """Load valid cached generated fleets in deterministic selector order."""

        if not self.configs_dir.exists():
            return ()
        with _exclusive_file_lock(self.index_lock_path):
            return self._list_locked()

    def invalidate_zone(self, *, project: str, zone: str) -> tuple[Path, ...]:
        """Remove generated configs for one GCP project and zone under the cache lock.

        This is used only after a successful provider refresh proves that a
        requested zone no longer has an eligible worker pool. Invalid cache
        files are deliberately left alone because their provider identity
        cannot be established safely.
        """

        if not self.configs_dir.exists():
            return ()
        with _exclusive_file_lock(self.index_lock_path):
            metadata = self._load_index()
            removed: list[Path] = []
            for path in sorted(self.configs_dir.glob("*.yaml")):
                config = _load_fleet_config(path)
                if config is None or config.provider.project != project or config.provider.zone != zone:
                    continue
                path.unlink(missing_ok=True)
                metadata.pop(config.name, None)
                removed.append(path)
            if removed:
                _atomic_write(self.index_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n", mode=0o600)
            return tuple(removed)

    def _list_locked(self) -> tuple[CachedAutoscaleConfig, ...]:
        """Load generated fleet YAML and its matching metadata under the cache lock."""

        metadata = self._load_index()
        values: list[tuple[CachedAutoscaleConfig, str]] = []
        for path in sorted(self.configs_dir.glob("*.yaml")):
            config, raw = _load_fleet_config_document(path)
            if config is None or raw is None:
                continue
            raw_info = metadata.get(config.name)
            info: JsonObject = raw_info if isinstance(raw_info, dict) else {}
            quota = _quota_from_dict(info.get("quota"))
            generated_at = info.get("generated_at")
            if not isinstance(generated_at, str):
                generated_at = None
            pool_max_workers = _public_worker_bounds_from_document(raw)
            pool_info = info.get("pools")
            if isinstance(pool_info, dict):
                for pool_name, pool_metadata in pool_info.items():
                    if pool_name in pool_max_workers or not isinstance(pool_metadata, dict):
                        continue
                    maximum = pool_metadata.get("max_workers")
                    if not isinstance(maximum, bool) and isinstance(maximum, int) and maximum >= 0:
                        pool_max_workers[pool_name] = maximum
            public_max_workers = next(iter(pool_max_workers.values())) if len(pool_max_workers) == 1 else None
            if public_max_workers is None:
                legacy_maximum = info.get("max_workers")
                if not isinstance(legacy_maximum, bool) and isinstance(legacy_maximum, int) and legacy_maximum >= 0:
                    public_max_workers = legacy_maximum
            generation_quotas = _generation_quotas_from_dict(info.get("generation_quotas"))
            if not generation_quotas and quota is not None:
                generation_quotas = (
                    GenerationQuotaSnapshot(
                        zone=quota.zone,
                        generation=quota.generation,
                        quota_tpus=quota.quota_tpus,
                        available_tpus=quota.available_tpus,
                        preemptible=quota.preemptible,
                        source_metric=quota.source_metric,
                        usage_observed_at=quota.usage_observed_at,
                    ),
                )
            entry = CachedAutoscaleConfig(
                path=path,
                config=config,
                quota=quota,
                generated_at=generated_at,
                public_max_workers=public_max_workers,
                pool_max_workers=tuple(sorted(pool_max_workers.items())),
                generation_quotas=generation_quotas,
            )
            layout = info.get("layout")
            values.append((entry, layout if layout in {"profile", "zone"} else "profile"))

        grouped_zones = {_cache_zone_key(entry) for entry, layout in values if layout == "zone"}
        visible = [entry for entry, layout in values if layout == "zone" or _cache_zone_key(entry) not in grouped_zones]
        return tuple(sorted(visible, key=lambda entry: ((entry.config.provider.zone or ""), entry.config.name)))

    def get(self, fleet_name: str) -> CachedAutoscaleConfig | None:
        """Return one cached fleet by its generated fleet name."""

        return next((entry for entry in self.list() if entry.config.name == fleet_name), None)

    def _load_index(self) -> JsonObject:
        if not self.index_path.exists():
            return {}
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if is_json_object(value) else {}


class QueueAdmissionDenialCache:
    """Persist exact Spot TPU queue-admission denials for automatic selection.

    This is intentionally separate from generated fleet metadata: a GCP
    queue-admission denial is a local eligibility fact, while generated YAML
    remains a reproducible output of quota discovery.
    """

    #: How long one denial keeps a shape out of automatic selection. A GCP
    #: stockout can last days, but so can the silence after a fixed one: an
    #: unexpired-forever denial turned one bad day into a permanently
    #: hand-cleared shape, so a denial ages out and the next attempt re-records
    #: it if the queue is still refusing.
    DEFAULT_TTL_S: tp.ClassVar[float] = 24 * 3600

    def __init__(self, root: Path | str, *, ttl_s: float = DEFAULT_TTL_S):
        self.root = Path(root).expanduser()
        self.ttl_s = max(float(ttl_s), 0.0)
        self.path = self.root / "queue-admission-denials.json"
        self.lock_path = self.root / ".queue-admission-denials.lock"

    def record(
        self,
        key: QueueAdmissionDenialKey,
        *,
        observed_at: datetime | None = None,
    ) -> QueueAdmissionDenial:
        """Record or refresh one exact queue-admission denial identity."""

        denial = QueueAdmissionDenial(
            key=key,
            observed_at=datetime.now(UTC) if observed_at is None else observed_at,
        )
        self._ensure_root()
        with _exclusive_file_lock(self.lock_path):
            entries = self._load_locked()
            previous = entries.get(key)
            if previous is not None and previous.observed_at >= denial.observed_at:
                return previous
            entries[key] = denial
            self._write_locked(entries)
        return denial

    def blocks(self, key: QueueAdmissionDenialKey) -> bool:
        """Return whether an unexpired prior denial still blocks this Spot worker."""

        if not self.path.exists():
            return False
        with _exclusive_file_lock(self.lock_path):
            entries = self._load_locked()
            denial = entries.get(key)
            if denial is None:
                return False
            if self._age_s(denial) >= self.ttl_s:
                # Expired: evict lazily so the file stops carrying stale
                # denials and the next attempt is genuinely free to retry.
                del entries[key]
                self._write_locked(entries)
                return False
            return True

    def _age_s(self, denial: QueueAdmissionDenial) -> float:
        return (datetime.now(UTC) - denial.observed_at).total_seconds()

    def list(self, *, include_expired: bool = False) -> tuple[QueueAdmissionDenial, ...]:
        """Return durable denials in stable identity order, freshest policy aside.

        Expired denials are hidden by default: the point of expiry is that a
        shape becomes selectable again, and reporting a stale denial would
        suggest otherwise. ``include_expired`` exists for inspection.
        """

        if not self.path.exists():
            return ()
        with _exclusive_file_lock(self.lock_path):
            values = list(self._load_locked().values())
        if not include_expired:
            values = [denial for denial in values if self._age_s(denial) < self.ttl_s]
        return tuple(sorted(values, key=_queue_admission_denial_sort_key))

    def clear(self, key: QueueAdmissionDenialKey | None = None) -> int:
        """Clear an exact denial, or every locally cached denial when ``key`` is absent."""

        if not self.path.exists():
            return 0
        self._ensure_root()
        with _exclusive_file_lock(self.lock_path):
            entries = self._load_locked()
            if key is None:
                removed = len(entries)
                entries.clear()
            elif key in entries:
                del entries[key]
                removed = 1
            else:
                removed = 0
            if removed:
                self._write_locked(entries)
            return removed

    def _ensure_root(self) -> None:
        """Create the owner-only cache directory before taking its file lock."""

        create_private_directory(self.root)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def _load_locked(self) -> dict[QueueAdmissionDenialKey, QueueAdmissionDenial]:
        """Load valid records under the cache lock, ignoring malformed local entries."""

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not is_json_object(raw) or raw.get("version") != 1:
            return {}
        values = raw.get("denials")
        if not isinstance(values, list):
            return {}
        entries: dict[QueueAdmissionDenialKey, QueueAdmissionDenial] = {}
        for value in values:
            if not is_json_object(value):
                continue
            denial = _queue_admission_denial_from_document(value)
            if denial is not None:
                entries[denial.key] = denial
        return entries

    def _write_locked(self, entries: dict[QueueAdmissionDenialKey, QueueAdmissionDenial]) -> None:
        """Write the complete deterministic denial document while holding its lock."""

        denials: list[JsonValue] = [
            denial.as_document() for denial in sorted(entries.values(), key=_queue_admission_denial_sort_key)
        ]
        document: JsonObject = {"denials": denials, "version": 1}
        _atomic_write(self.path, json.dumps(document, indent=2, sort_keys=True) + "\n", mode=0o600)


class AutoscaleConfigCache:
    """Compatibility facade over defaults and generated-fleet repositories."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or DEFAULT_AUTOSCALE_CACHE_ROOT).expanduser()
        self.configs_dir = self.root / "configs"
        self.defaults_path = self.root / "defaults.yaml"
        self.index_path = self.root / "index.json"
        self._generated = GeneratedAutoscaleCache(self.root)
        self._queue_admission_denials = QueueAdmissionDenialCache(self.root)

    def load_defaults(self, path: Path | str | None = None) -> AutoscaleDefaults:
        """Load a YAML autoscale profile, or return an empty profile when absent."""

        return AutoscaleDefaultsRepository(path or self.defaults_path).load()

    def save_defaults(self, defaults: AutoscaleDefaults, path: Path | str | None = None) -> Path:
        """Atomically save one owner-only YAML autoscale profile."""

        target = Path(path or self.defaults_path).expanduser()
        if target == self.defaults_path:
            try:
                os.chmod(self.root, 0o700)
            except OSError:
                pass
        return AutoscaleDefaultsRepository(target).save(defaults)

    def store(
        self,
        generated: GeneratedAutoscale,
        *,
        refresh_generations: tuple[str, ...] = (),
        preserve_types: tuple[str, ...] = (),
    ) -> CachedAutoscaleConfig:
        """Atomically persist one generated config and its non-secret metadata."""

        return self._generated.store(generated, refresh_generations=refresh_generations, preserve_types=preserve_types)

    def preview(self, generated: GeneratedAutoscale) -> CachedAutoscaleConfig:
        """Return a cache-shaped generated entry without writing local state."""

        return self._generated.preview(generated)

    def list(self) -> tuple[CachedAutoscaleConfig, ...]:
        """Load valid cached generated fleets in deterministic selector order."""

        return self._generated.list()

    def list_eligible(self) -> tuple[CachedAutoscaleConfig, ...]:
        """Load cached fleets with unsafe GCP Spot TPU pools removed.

        Cached YAML remains unchanged because it is the reproducible discovery
        output. The returned view excludes exact queue-admission denials and
        old finite-quota metadata without a live usable-balance observation.
        It is the only one suitable for a no-regeneration ``eyvan up`` selector.
        """

        blocked_keys = frozenset(denial.key for denial in self._queue_admission_denials.list())
        entries: list[CachedAutoscaleConfig] = []
        for entry in self._generated.list():
            eligible = _eligible_cached_config(entry, blocked_keys)
            if eligible is not None:
                entries.append(eligible)
        return tuple(entries)

    def invalidate_zone(self, *, project: str, zone: str) -> tuple[Path, ...]:
        """Remove generated cache entries for one GCP project and zone."""

        return self._generated.invalidate_zone(project=project, zone=zone)

    def get(self, fleet_name: str) -> CachedAutoscaleConfig | None:
        """Return one cached fleet by its generated fleet name."""

        return self._generated.get(fleet_name)

    def record_queue_admission_denial(
        self,
        *,
        project: str,
        zone: str,
        accelerator_type: str,
        capacity_class: CapacityClass,
        observed_at: datetime | None = None,
    ) -> QueueAdmissionDenial:
        """Remember that GCP denied one exact Spot TPU queued-resource request."""

        return self._queue_admission_denials.record(
            QueueAdmissionDenialKey(
                project=project,
                zone=zone,
                accelerator_type=accelerator_type,
                capacity_class=capacity_class,
            ),
            observed_at=observed_at,
        )

    def blocks_queue_admission_denial(
        self,
        *,
        project: str,
        zone: str,
        accelerator_type: str,
        capacity_class: CapacityClass,
    ) -> bool:
        """Return whether a cached exact GCP denial excludes one Spot TPU worker."""

        return self._queue_admission_denials.blocks(
            QueueAdmissionDenialKey(
                project=project,
                zone=zone,
                accelerator_type=accelerator_type,
                capacity_class=capacity_class,
            )
        )

    def list_queue_admission_denials(self) -> tuple[QueueAdmissionDenial, ...]:
        """Return durable GCP queue-admission denials for inspection by the CLI."""

        return self._queue_admission_denials.list()

    def clear_queue_admission_denials(
        self,
        key: QueueAdmissionDenialKey | None = None,
    ) -> int:
        """Clear one exact cached denial or every cached denial when ``key`` is absent."""

        return self._queue_admission_denials.clear(key)


def _queue_admission_denial_sort_key(denial: QueueAdmissionDenial) -> tuple[str, str, str, str]:
    """Return deterministic storage and display ordering for denial entries."""

    key = denial.key
    return key.project, key.zone, key.accelerator_type, key.capacity_class.value


def _queue_admission_denial_from_document(value: JsonObject) -> QueueAdmissionDenial | None:
    """Parse one cache entry without allowing malformed local state to block generation."""

    project = value.get("project")
    zone = value.get("zone")
    accelerator_type = value.get("accelerator_type")
    capacity_class_value = value.get("capacity_class")
    observed_at_value = value.get("observed_at")
    if not all(
        isinstance(item, str) for item in (project, zone, accelerator_type, capacity_class_value, observed_at_value)
    ):
        return None
    try:
        return QueueAdmissionDenial(
            key=QueueAdmissionDenialKey(
                project=project,
                zone=zone,
                accelerator_type=accelerator_type,
                capacity_class=CapacityClass(capacity_class_value),
            ),
            observed_at=datetime.fromisoformat(observed_at_value),
        )
    except ValueError:
        return None


def _eligible_cached_config(
    entry: CachedAutoscaleConfig,
    blocked_keys: frozenset[QueueAdmissionDenialKey],
) -> CachedAutoscaleConfig | None:
    """Return one selector-safe cache view after exact Spot TPU denials are removed."""

    eligible_pools = {
        name: pool for name, pool in entry.config.pools.items() if not _cached_pool_is_blocked(entry, pool, blocked_keys)
    }
    if not eligible_pools:
        return None
    if len(eligible_pools) == len(entry.config.pools):
        return entry

    eligible_accelerator_types = {
        pool.accelerator_type for pool in eligible_pools.values() if isinstance(pool, TpuNodeType)
    }
    eligible_generation_counts: dict[str, int] = {}
    for accelerator_type in eligible_accelerator_types:
        generation = accelerator_type.partition("-")[0]
        eligible_generation_counts[generation] = eligible_generation_counts.get(generation, 0) + 1
    pool_max_workers = tuple((name, maximum) for name, maximum in entry.pool_max_workers if name in eligible_pools)
    quota = (
        entry.quota if entry.quota is not None and entry.quota.accelerator_type in eligible_accelerator_types else None
    )
    return CachedAutoscaleConfig(
        path=entry.path,
        config=entry.config.model_copy(update={"pools": eligible_pools}),
        quota=quota,
        generated_at=entry.generated_at,
        public_max_workers=None,
        pool_max_workers=pool_max_workers,
        generation_quotas=tuple(
            replace(quota, shared=eligible_generation_counts[quota.generation] > 1)
            for quota in entry.generation_quotas
            if quota.generation in eligible_generation_counts
        ),
    )


def _cached_pool_is_blocked(
    entry: CachedAutoscaleConfig,
    pool: NodeType,
    blocked_keys: frozenset[QueueAdmissionDenialKey],
) -> bool:
    """Return whether a cached GCP Spot pool lacks a safe auto-selection basis."""

    config = entry.config
    if config.provider.type != "gcp" or not isinstance(pool, TpuNodeType):
        return False
    if (pool.capacity_class or config.capacity_class) is not CapacityClass.SPOT:
        return False
    if _cached_pool_has_unknown_live_quota(entry, pool):
        return True
    project = pool.provider_project or config.provider.project
    zone = pool.zone or config.provider.zone
    if project is None or zone is None:
        return False
    return (
        QueueAdmissionDenialKey(
            project=project,
            zone=zone,
            accelerator_type=pool.accelerator_type,
            capacity_class=CapacityClass.SPOT,
        )
        in blocked_keys
    )


def _cached_pool_has_unknown_live_quota(entry: CachedAutoscaleConfig, pool: TpuNodeType) -> bool:
    """Reject old finite-quota cache rows that predate Monitoring balance reads."""

    quota = next(
        (
            candidate
            for candidate in entry.generation_quotas
            if candidate.generation == pool.accelerator_type.partition("-")[0]
        ),
        None,
    )
    if quota is None and entry.quota is not None and entry.quota.accelerator_type == pool.accelerator_type:
        return entry.quota.quota_tpus is not None and entry.quota.available_tpus is None
    return quota is not None and quota.quota_tpus is not None and quota.available_tpus is None


def _load_fleet_config(path: Path) -> EyvanFleetConfig | None:
    """Load one valid generated fleet config without exposing its YAML document."""

    config, _document = _load_fleet_config_document(path)
    return config


def _load_fleet_config_document(path: Path) -> tuple[EyvanFleetConfig | None, JsonObject | None]:
    """Load one valid generated fleet config together with its YAML document."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not is_json_object(raw):
            return None, None
        return validate_generated_fleet(raw), raw
    except (OSError, ValueError, yaml.YAMLError):
        return None, None


def _metadata_for(
    generated: GeneratedAutoscale,
    *,
    generated_at: str,
    path: Path,
) -> JsonObject:
    pools: JsonObject = {}
    for name, pool in _document_pools(generated.document).items():
        pools[name] = {"max_workers": _pool_max_workers(pool)}
    if isinstance(generated, GeneratedAutoscaleConfig):
        profile_pool = _document_pools(generated.document)[generated.profile.accelerator_type]
        metadata: JsonObject = {
            "generated_at": generated_at,
            "layout": "profile",
            "max_workers": _pool_max_workers(profile_pool),
            "path": path.name,
            "pools": pools,
            "quota": generated.quota.as_dict() if generated.quota is not None else None,
        }
        return metadata
    generation_quotas: list[JsonValue] = [quota.as_dict() for quota in generated.generation_quotas]
    metadata: JsonObject = {
        "generated_at": generated_at,
        "generation_quotas": generation_quotas,
        "layout": "zone",
        "path": path.name,
        "pools": pools,
    }
    return metadata


def _cached_from_generated(
    generated: GeneratedAutoscale,
    *,
    path: Path,
    generated_at: str | None,
) -> CachedAutoscaleConfig:
    pool_max_workers = tuple(
        sorted((name, _pool_max_workers(pool)) for name, pool in _document_pools(generated.document).items())
    )
    if isinstance(generated, GeneratedAutoscaleConfig):
        quota = generated.quota
        generation_quotas = (
            ()
            if quota is None
            else (
                GenerationQuotaSnapshot(
                    zone=quota.zone,
                    generation=quota.generation,
                    quota_tpus=quota.quota_tpus,
                    available_tpus=quota.available_tpus,
                    preemptible=quota.preemptible,
                    source_metric=quota.source_metric,
                    usage_observed_at=quota.usage_observed_at,
                ),
            )
        )
        public_max_workers = _pool_max_workers(_document_pools(generated.document)[generated.profile.accelerator_type])
    else:
        quota = None
        generation_quotas = generated.generation_quotas
        public_max_workers = None
    return CachedAutoscaleConfig(
        path=path,
        config=generated.config,
        quota=quota,
        generated_at=generated_at,
        public_max_workers=public_max_workers,
        pool_max_workers=pool_max_workers,
        generation_quotas=generation_quotas,
    )


def _atomic_write(path: Path, content: str, *, mode: int) -> None:
    """Atomically write an owner-only cache artifact."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_file_lock(path: Path) -> tp.Iterator[None]:
    """Serialize cross-process read-modify-write updates for one cache file."""

    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _document_pools(document: JsonObject) -> dict[str, JsonObject]:
    """Return the concrete public pool documents produced by the compiler."""

    raw_pools = document.get("pools")
    if not isinstance(raw_pools, dict):
        raise AssertionError("generated autoscale document is missing pools")
    pools: dict[str, JsonObject] = {}
    for name, raw_pool in raw_pools.items():
        if not isinstance(raw_pool, dict):
            raise AssertionError("generated autoscale pool is not an object")
        pools[name] = raw_pool
    return pools


def _pool_max_workers(pool: JsonObject) -> int:
    """Return the public max worker value retained in a generated pool."""

    maximum = pool.get("max_workers")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise AssertionError("generated autoscale pool has an invalid max_workers value")
    return maximum


def _required_string(value: JsonObject, field_name: str) -> str:
    """Read one required string from tolerant legacy cache metadata."""

    item = value.get(field_name)
    if not isinstance(item, str):
        raise ValueError(f"cache metadata {field_name} must be a string")
    return item


def _optional_string(value: JsonValue | None) -> str | None:
    """Read an optional string from tolerant legacy cache metadata."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("cache metadata value must be a string")
    return value


def _optional_non_negative_integer(value: JsonValue | None) -> int | None:
    """Read an optional non-negative integer from cache metadata."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cache metadata value must be a non-negative integer")
    return value


def _optional_boolean(value: JsonValue | None, *, default: bool) -> bool:
    """Read an optional boolean from cache metadata without coercion."""

    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("cache metadata value must be a boolean")
    return value


def _optional_datetime(value: JsonValue | None) -> datetime | None:
    """Read an optional timezone-aware ISO timestamp from cache metadata."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("cache metadata timestamp must be a string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("cache metadata timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _quota_from_dict(value: JsonValue | None) -> QuotaSnapshot | None:
    if not is_json_object(value):
        return None
    try:
        zone = _required_string(value, "zone")
        accelerator_type = _required_string(value, "accelerator_type")
        generation = _required_string(value, "generation")
        quota_tpus = _optional_non_negative_integer(value.get("quota_tpus"))
        available_tpus = _optional_non_negative_integer(value.get("available_tpus"))
        preemptible = _optional_boolean(value.get("preemptible"), default=True)
        source_metric = _optional_string(value.get("source_metric"))
        usage_observed_at = _optional_datetime(value.get("usage_observed_at"))
        return QuotaSnapshot(
            zone=zone,
            accelerator_type=accelerator_type,
            generation=generation,
            quota_tpus=quota_tpus,
            available_tpus=available_tpus,
            preemptible=preemptible,
            source_metric=source_metric,
            usage_observed_at=usage_observed_at,
        )
    except ValueError:
        return None


def _generation_quotas_from_dict(value: JsonValue | None) -> tuple[GenerationQuotaSnapshot, ...]:
    """Load optional zone-generation quota provenance from cache metadata."""

    if not isinstance(value, list):
        return ()
    snapshots: list[GenerationQuotaSnapshot] = []
    for item in value:
        if not is_json_object(item):
            continue
        try:
            snapshots.append(
                GenerationQuotaSnapshot(
                    zone=_required_string(item, "zone"),
                    generation=_required_string(item, "generation"),
                    quota_tpus=_optional_non_negative_integer(item.get("quota_tpus")),
                    available_tpus=_optional_non_negative_integer(item.get("available_tpus")),
                    preemptible=_optional_boolean(item.get("preemptible"), default=True),
                    source_metric=_optional_string(item.get("source_metric")),
                    shared=_optional_boolean(item.get("shared"), default=False),
                    usage_observed_at=_optional_datetime(item.get("usage_observed_at")),
                )
            )
        except ValueError:
            continue
    return tuple(sorted(snapshots, key=lambda quota: quota.generation))


def _cache_zone_key(entry: CachedAutoscaleConfig) -> tuple[str | None, str | None]:
    """Return provider identity used to hide superseded profile cache rows."""

    return entry.config.provider.project, entry.config.provider.zone


def _public_worker_bounds_from_document(value: JsonObject) -> dict[str, int]:
    """Read public worker bounds from a generated fleet YAML document."""

    pools = value.get("pools")
    if not isinstance(pools, dict):
        return {}
    bounds: dict[str, int] = {}
    for name, pool in pools.items():
        if not isinstance(pool, dict):
            continue
        max_workers = pool.get("max_workers")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 0:
            continue
        bounds[name] = max_workers
    return bounds


def _public_max_workers_from_document(value: JsonObject) -> int | None:
    """Read the public worker bound from one generated single-pool YAML document."""

    bounds = _public_worker_bounds_from_document(value)
    return next(iter(bounds.values())) if len(bounds) == 1 else None


__all__ = (
    "AutoscaleConfigCache",
    "AutoscaleDefaultsRepository",
    "GeneratedAutoscaleCache",
    "QueueAdmissionDenialCache",
)
