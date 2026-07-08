# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""Ray cluster-launcher config generation for elastic TPU clusters.

The port of the repo's ``autoscale/generate-cluster-configs.py`` with its
known bugs fixed while moving:

- Node types now advertise the **eray-canonical resource labels**
  (:func:`eray.cli.utils.tpu_resource_labels`): ``TPU-{fam}-{size}-head`` with
  the exact casing ``SlicePoolManager`` schedules on (the old generator
  emitted lowercase ``tpu-...-head``, which the pool could never match), the
  ``TPU`` count is physical chips per host (the old ``TPU: 4`` hardcode
  overadvertised v5p hosts), plus ``TPU-{fam}`` and
  ``accelerator_type:TPU-{FAM}`` for parity with connect-mode clusters.
- Spot is a **flag** (``spot=True`` default), not a hardcode.
- GCP is queried through the gcloud CLI (``accelerator-types list``), not
  google-api-python-client; the YAML is assembled as a dict round-trip
  through pyyaml, not string concatenation; the Service Usage quota scraping
  and dead config paths were dropped.

Positioning: the launcher path is the elastic many-small-slices tool (Ray's
GCP provider itself notes multi-host TPU autoscaling is best-effort); the
QR + fleet path is the reliable big-slice tool. Both advertise the same
scheduling contract.
"""

from __future__ import annotations

import subprocess
from importlib import resources as importlib_resources
from pathlib import Path

import yaml

from ..gcp import gcloud_json
from .qr import default_runtime_version

DEFAULT_IMAGE = "ghcr.io/erfanzar/easydel:latest-tpu"
DEFAULT_FAMILIES = ("v4", "v5e", "v5p", "v6e")

#: Physical chips per host by family (drives node-type host math). v2-v5p
#: hosts carry 4 chips; v5e hosts carry 8; v6e ships in 4-chip host machines.
CHIPS_PER_HOST_BY_FAMILY: dict[str, int] = {
    "v2": 4,
    "v3": 4,
    "v4": 4,
    "v5p": 4,
    "v5e": 8,
    "v5litepod": 8,
    "v6e": 4,
}

#: Family name as it appears in accelerator types (v5e is "v5litepod-N").
ACCELERATOR_FAMILY_NAME = {"v5e": "v5litepod"}


def _total_chips(accelerator_type: str) -> int:
    """Physical chips in a slice (suffix counts cores on 2-core generations)."""
    family = accelerator_type.split("-")[0].lower()
    suffix = int(accelerator_type.split("-")[1])
    cores_per_chip = 2 if family in ("v2", "v3", "v4", "v5p") else 1
    return max(suffix // cores_per_chip, 1)


def slice_hosts(accelerator_type: str) -> int:
    """Worker hosts in a slice of the given accelerator type.

    Args:
        accelerator_type: e.g. ``"v5p-64"`` (8 hosts) or ``"v5litepod-16"``
            (2 hosts).

    Returns:
        Host count (minimum 1).
    """
    family = accelerator_type.split("-")[0].lower()
    per_host = CHIPS_PER_HOST_BY_FAMILY.get(family, 4)
    return max(_total_chips(accelerator_type) // per_host, 1)


def list_zone_accelerator_types(project: str, zone: str) -> list[str]:
    """Accelerator types offered in a zone.

    Args:
        project: GCP project id.
        zone: GCP zone.

    Returns:
        Type names (e.g. ``["v5p-8", "v5litepod-16", ...]``); empty when the
        zone offers none or the listing fails.
    """
    try:
        raw = gcloud_json(["compute", "tpus", "accelerator-types", "list", "--zone", zone, "--project", project])
    except subprocess.CalledProcessError:
        return []
    types = []
    for item in raw if isinstance(raw, list) else []:
        name = str(item.get("type") or item.get("name", "")).rsplit("/", 1)[-1]
        if name:
            types.append(name)
    return types


def list_tpu_zones(project: str) -> list[str]:
    """All TPU locations available to the project.

    Args:
        project: GCP project id.

    Returns:
        Zone names.
    """
    raw = gcloud_json(["compute", "tpus", "locations", "list", "--project", project])
    zones = []
    for item in raw if isinstance(raw, list) else []:
        name = str(item.get("locationId") or item.get("name", "")).rsplit("/", 1)[-1]
        if name:
            zones.append(name)
    return zones


def make_node_type(accelerator_type: str, *, spot: bool, min_workers: int = 0, max_workers: int = 1024) -> dict:
    """Build one launcher node type for a TPU slice size.

    Args:
        accelerator_type: e.g. ``"v5p-64"``.
        spot: Request preemptible capacity.
        min_workers: Slices kept warm.
        max_workers: Slice count ceiling.

    Returns:
        The ``available_node_types`` entry body.
    """
    from ..cli.utils import tpu_resource_labels

    hosts = slice_hosts(accelerator_type)
    resources = {"CPU": 120, **tpu_resource_labels(accelerator_type, hosts, is_head=True)}
    return {
        "min_workers": min_workers,
        "max_workers": max_workers,
        "resources": resources,
        "node_config": {
            "acceleratorType": accelerator_type,
            "runtimeVersion": default_runtime_version(accelerator_type),
            "schedulingConfig": {"preemptible": bool(spot)},
        },
    }


def _load_template() -> str:
    """The packaged cluster template text."""
    return (importlib_resources.files("eray.provision") / "templates" / "cluster-template.yaml").read_text()


def generate_zone_config(
    project: str,
    zone: str,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    image: str = DEFAULT_IMAGE,
    spot: bool = True,
    available_types: list[str] | None = None,
) -> dict | None:
    """Generate one zone's cluster-launcher config as a dict.

    Args:
        project: GCP project id.
        zone: GCP zone.
        families: TPU families to include.
        image: Docker image for head/workers.
        spot: Request preemptible capacity on every TPU node type.
        available_types: Override the zone's accelerator-type listing (tests).

    Returns:
        The config dict, or None when the zone offers none of the families.
    """
    types = list_zone_accelerator_types(project, zone) if available_types is None else available_types
    region = zone.rsplit("-", 1)[0]
    wanted_prefixes = tuple(ACCELERATOR_FAMILY_NAME.get(f, f) for f in families)

    selected: dict[str, dict] = {}
    for acc_type in sorted(types, key=lambda t: (t.split("-")[0], int(t.split("-")[1]) if "-" in t else 0)):
        family = acc_type.split("-")[0].lower()
        if family not in wanted_prefixes:
            continue
        key = f"tpu_slice_{family}_{acc_type.split('-')[1]}"
        selected[key] = make_node_type(acc_type, spot=spot)
    if not selected:
        return None

    text = _load_template()
    for placeholder, value in (
        ("{{NAME}}", f"easydel-{zone}"),
        ("{{REGION}}", region),
        ("{{ZONE}}", zone),
        ("{{PROJECT_ID}}", project),
        ("{{IMAGE}}", image),
    ):
        text = text.replace(placeholder, value)
    config = yaml.safe_load(text)
    config.setdefault("available_node_types", {}).update(selected)
    return config


def generate_configs(
    project: str,
    *,
    zones: list[str] | None = None,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    image: str = DEFAULT_IMAGE,
    spot: bool = True,
    output_dir: Path | str,
) -> list[Path]:
    """Generate per-zone launcher YAMLs.

    Args:
        project: GCP project id.
        zones: Zones to generate (default: every TPU location).
        families: TPU families to include.
        image: Docker image for head/workers.
        spot: Preemptible capacity (default) vs on-demand.
        output_dir: Directory for ``easydel-{zone}.yaml`` files.

    Returns:
        Paths written.
    """
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for zone in zones or list_tpu_zones(project):
        config = generate_zone_config(project, zone, families=families, image=image, spot=spot)
        if config is None:
            continue
        path = out / f"easydel-{zone}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written.append(path)
    return written


def ray_up(config_path: str | Path, *, yes: bool = True) -> None:
    """Bring a launcher cluster up (`ray up`).

    Args:
        config_path: Cluster YAML path.
        yes: Skip the interactive confirmation.

    Raises:
        subprocess.CalledProcessError: If `ray up` fails.
    """
    args = ["ray", "up", str(config_path), "--no-config-cache"]
    if yes:
        args.append("-y")
    subprocess.run(args, check=True)


def ray_down(config_path: str | Path, *, yes: bool = True) -> None:
    """Tear a launcher cluster down (`ray down`).

    Args:
        config_path: Cluster YAML path.
        yes: Skip the interactive confirmation.

    Raises:
        subprocess.CalledProcessError: If `ray down` fails.
    """
    args = ["ray", "down", str(config_path)]
    if yes:
        args.append("-y")
    subprocess.run(args, check=True)


def launcher_head_ip(config_path: str | Path) -> str:
    """The head IP of a launcher cluster.

    Args:
        config_path: Cluster YAML path.

    Returns:
        The head node IP as reported by `ray get-head-ip`.

    Raises:
        subprocess.CalledProcessError: If the cluster is not up.
    """
    result = subprocess.run(["ray", "get-head-ip", str(config_path)], check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()[-1].strip()


__all__ = [
    "CHIPS_PER_HOST_BY_FAMILY",
    "DEFAULT_FAMILIES",
    "DEFAULT_IMAGE",
    "generate_configs",
    "generate_zone_config",
    "launcher_head_ip",
    "list_tpu_zones",
    "list_zone_accelerator_types",
    "make_node_type",
    "ray_down",
    "ray_up",
    "slice_hosts",
]
