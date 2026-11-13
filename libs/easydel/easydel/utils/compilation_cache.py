# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Where JAX's persistent compilation cache lives, resolved without user setup.

Order, first match wins:

1. An explicit directory passed by the caller (a serving YAML's ``compilation_cache_dir``).
2. A directory JAX already has (``JAX_COMPILATION_CACHE_DIR`` or ``jax.config``); it is left alone.
3. On Google Cloud: ``gs://easydel-caching-<region>-<project number>/jax-<jax version>/<model>``. The bucket is
   created in this VM's project and region on first use, with uniform access, public access prevention and a 90-day
   object lifetime. It is used only when it belongs to this project: JAX loads and runs whatever executables it finds
   in its cache, and bucket names are global, so a bucket owned by anyone else is refused.
4. Otherwise no persistent cache (JAX's default).

On a multi-host slice every process resolves the same directory. JAX writes entries only from distributed process 0
and every process reads them, so one shared directory keeps every host warm; a host-local directory does not.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import typing as tp
import urllib.request

from eformer.loggings import get_logger

logger = get_logger("CompilationCache")

BUCKET_PREFIX = "easydel-caching"
OBJECT_LIFETIME_DAYS = 90
_METADATA_URL = "http://169.254.169.254/computeMetadata/v1/"
_STORAGE_URL = "https://storage.googleapis.com/storage/v1/b"
_resolved: dict[str, str | None] = {}


@dataclasses.dataclass(frozen=True)
class GcpLocation:
    """The Google Cloud project and region this process runs in."""

    project_id: str
    project_number: str
    region: str


class BucketStore(tp.Protocol):
    """The two Cloud Storage calls the resolver needs."""

    def get(self, bucket: str) -> dict[str, tp.Any] | None:
        """Bucket metadata, or ``None`` when the bucket does not exist."""

    def create(self, bucket: str, *, project_id: str, region: str) -> None:
        """Create ``bucket``; succeeds silently when it already exists."""


def _metadata(path: str, timeout_s: float) -> str | None:
    request = urllib.request.Request(_METADATA_URL + path, headers={"Metadata-Flavor": "Google"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read().decode().strip() or None
    except Exception:
        return None


def detect_gcp(
    *, fetch: tp.Callable[[str, float], str | None] = _metadata, timeout_s: float = 1.0
) -> GcpLocation | None:
    """This VM's project and region from the GCE metadata server, or ``None`` off Google Cloud."""
    zone = fetch("instance/zone", timeout_s)
    if not zone:
        return None
    project_id = fetch("project/project-id", timeout_s)
    project_number = fetch("project/numeric-project-id", timeout_s)
    zone_name = zone.rsplit("/", 1)[-1]  # projects/123/zones/europe-west4-b
    region = zone_name.rsplit("-", 1)[0] if zone_name.count("-") >= 2 else ""
    if not (project_id and project_number and region):
        return None
    return GcpLocation(project_id=project_id, project_number=project_number, region=region)


class GcsBuckets:
    """Cloud Storage JSON API with the process's default Google credentials."""

    def __init__(self, timeout_s: float = 15.0):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_write"])
        self._session = AuthorizedSession(credentials)
        self._timeout_s = timeout_s

    def get(self, bucket: str) -> dict[str, tp.Any] | None:
        response = self._session.get(f"{_STORAGE_URL}/{bucket}", timeout=self._timeout_s)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def create(self, bucket: str, *, project_id: str, region: str) -> None:
        body = {
            "name": bucket,
            "location": region.upper(),
            "storageClass": "STANDARD",
            "labels": {"created-by": "easydel"},
            "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": True}, "publicAccessPrevention": "enforced"},
            "lifecycle": {"rule": [{"action": {"type": "Delete"}, "condition": {"age": OBJECT_LIFETIME_DAYS}}]},
        }
        response = self._session.post(_STORAGE_URL, params={"project": project_id}, json=body, timeout=self._timeout_s)
        if response.status_code != 409:  # 409: it exists (another host of the slice created it first)
            response.raise_for_status()


def cache_namespace(model: tp.Any) -> str:
    """A directory-safe name for ``model``: a checkpoint path's last component, else the config's name."""
    if isinstance(model, str):
        name = model.rstrip("/").rsplit("/", 1)[-1]
    else:
        config = getattr(model, "config", None)
        name = str(getattr(config, "_name_or_path", "") or "").rstrip("/").rsplit("/", 1)[-1]
        name = name or type(model).__name__
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:100] or "default"


def bucket_name(location: GcpLocation) -> str:
    """``easydel-caching-<region>-<project number>``; the project number keeps the global name ours."""
    return f"{BUCKET_PREFIX}-{location.region}-{location.project_number}"


def _owned_bucket(store: BucketStore, location: GcpLocation) -> str | None:
    name = bucket_name(location)
    metadata = store.get(name)
    if metadata is None:
        store.create(name, project_id=location.project_id, region=location.region)
        logger.info(f"created gs://{name} for the JAX compilation cache")
        metadata = store.get(name)
    owner = str((metadata or {}).get("projectNumber", ""))
    if owner != location.project_number:
        logger.warning(
            f"gs://{name} belongs to project {owner or '?'}, not {location.project_number}; JAX would run "
            "executables from it, so it is not used for the compilation cache"
        )
        return None
    return name


def resolve_gcs_cache_dir(
    namespace: str,
    *,
    locate: tp.Callable[[], GcpLocation | None] = detect_gcp,
    store_factory: tp.Callable[[], BucketStore] = GcsBuckets,
    jax_version: str | None = None,
) -> str | None:
    """The Google Cloud cache directory for ``namespace``, or ``None`` off Google Cloud or when unusable."""
    location = locate()
    if location is None:
        return None
    if importlib.util.find_spec("etils") is None:
        logger.warning("on Google Cloud but etils[epath] is not installed, so JAX cannot use a gs:// cache")
        return None
    try:
        bucket = _owned_bucket(store_factory(), location)
    except Exception as exc:  # no permission, no credentials, network: compile without a shared cache
        logger.warning(
            f"compilation cache bucket {bucket_name(location)} is unavailable ({type(exc).__name__}: {exc}); "
            "compiling without a shared cache"
        )
        return None
    if bucket is None:
        return None
    if jax_version is None:
        import jax

        jax_version = jax.__version__
    return f"gs://{bucket}/jax-{jax_version}/{namespace}"


def configure_compilation_cache(namespace: str, explicit: str | None = None) -> str | None:
    """Point JAX's persistent compilation cache at the resolved directory and return it (``None``: no cache).

    Call it before the first compile that should be cached. Resolution runs once per namespace per process.

    Args:
        namespace: Per-model directory name (see :func:`cache_namespace`).
        explicit: A directory the user chose; it always wins.
    """
    import jax
    from jax.experimental.compilation_cache import compilation_cache as jax_cache

    current = jax.config.jax_compilation_cache_dir
    if explicit:
        if current != explicit:
            if current:
                jax_cache.reset_cache()
            jax_cache.set_cache_dir(explicit)
            logger.info(f"JAX compilation cache: {explicit} (configured)")
        return explicit
    if current or not jax.config.jax_enable_compilation_cache:
        return current or None
    if namespace not in _resolved:
        _resolved[namespace] = resolve_gcs_cache_dir(namespace)
    path = _resolved[namespace]
    if path:
        jax_cache.set_cache_dir(path)
        logger.info(f"JAX compilation cache: {path} (Google Cloud, automatic)")
    return path
