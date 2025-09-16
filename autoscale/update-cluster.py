# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""
Google Cloud TPU cluster configuration discovery and generation tool.

This module discovers available TPU resources across GCP zones and generates
cluster configuration YAML files for each zone with available TPU families.
It queries the TPU API for available accelerator types and optionally collects
quota information to determine TPU availability.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import google.auth
import jinja2
import yaml
from google.auth.transport.requests import AuthorizedSession
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

this_path = Path(__file__).parent.absolute()
cluster_template_path = this_path / "easydel-cluster-template.yaml"

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")


@dataclass
class TPUGenerationConfig:
    """Configuration for a specific TPU generation."""

    runtime_version: str
    base_worker: str
    slices: list[int]
    num_tpus: int
    tpus_worker: int | None = None


@dataclass
class ClusterConfig:
    """Configuration for a TPU cluster."""

    name: str
    region: str
    zone: str
    project_id: str
    bucket: str
    tpu_generation: str
    min_workers: int = 0
    worker_targets: dict[str, int] = field(default_factory=dict)


GENERATION_CONFIGS: dict[str, TPUGenerationConfig] = {
    "v4": TPUGenerationConfig(
        runtime_version="tpu-ubuntu2204-base",
        base_worker="8",
        slices=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        num_tpus=4,
        tpus_worker=4,
    ),
    "v5e": TPUGenerationConfig(
        runtime_version="v2-alpha-tpuv5-lite",
        base_worker="4",
        slices=[8, 16, 32, 64, 128, 256],
        num_tpus=4,
        tpus_worker=1,
    ),
    "v5p": TPUGenerationConfig(
        runtime_version="v2-alpha-tpuv5",
        base_worker="8",
        slices=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        num_tpus=4,
        tpus_worker=8,
    ),
    "v6e": TPUGenerationConfig(
        runtime_version="v2-alpha-tpuv6e",
        base_worker="4",
        slices=[8, 16, 32, 64, 128, 256],
        num_tpus=4,
    ),
    "v6e-serve": TPUGenerationConfig(
        runtime_version="v2-alpha-tpuv6e",
        base_worker="8",
        slices=[],
        num_tpus=8,
    ),
    "v4-serve": TPUGenerationConfig(
        runtime_version="tpu-ubuntu2204-base",
        base_worker="16",
        slices=[],
        num_tpus=4,
    ),
}


def get_default_project_id() -> str | None:
    pid = os.environ.get("GCP_PROJECT_ID")
    if pid:
        return pid
    _, default_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return default_project


def authorized_session() -> tuple[AuthorizedSession, str | None]:
    creds, default_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds), default_project


def get_project_number(project_id: str) -> str:
    """
    Robustly fetch project number via Cloud Resource Manager v3, then v1 fallback.
    """
    # Try v3
    try:
        crm_v3 = build("cloudresourcemanager", "v3", cache_discovery=False)
        resp_v3 = crm_v3.projects().get(name=f"projects/{project_id}").execute()
        name_field = resp_v3.get("name")
        if name_field and name_field.startswith("projects/"):
            candidate = name_field.split("/")[-1]
            if candidate.isdigit():
                return candidate
        pn = resp_v3.get("projectNumber") or resp_v3.get("project_number")
        if pn:
            return str(pn)
    except HttpError:
        pass

    # Fallback v1
    try:
        crm_v1 = build("cloudresourcemanager", "v1", cache_discovery=False)
        resp_v1 = crm_v1.projects().get(projectId=project_id).execute()
        pn = resp_v1.get("projectNumber")
        if pn:
            return str(pn)
    except HttpError as e:
        raise RuntimeError(f"Error calling Cloud Resource Manager: {e}") from e

    raise RuntimeError("Could not determine project number from Cloud Resource Manager (v3/v1).")


def su_list_enabled_services(session: AuthorizedSession, project_number: str) -> list[str]:
    url = f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services?filter=state:ENABLED"
    services = []
    while True:
        r = session.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"Service Usage services.list error {r.status_code}: {r.text}")
        data = r.json()
        for s in data.get("services", []):
            name = s.get("name", "")
            parts = name.split("/")
            if len(parts) >= 4:
                services.append(parts[-1])
        token = data.get("nextPageToken")
        if not token:
            break
        url = f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services?filter=state:ENABLED&pageToken={token}"
    return services


def su_list_consumer_quota_metrics(
    session: AuthorizedSession, project_number: str, service_name: str
) -> list[dict[str, Any]]:
    base = (
        f"https://serviceusage.googleapis.com/v1/projects/{project_number}/services/{service_name}/consumerQuotaMetrics"
    )
    url = f"{base}?view=FULL"
    out = []
    while True:
        r = session.get(url)
        if r.status_code in (404, 403):
            return []
        if r.status_code != 200:
            raise RuntimeError(
                f"Service Usage consumerQuotaMetrics.list error for {service_name} {r.status_code}: {r.text}"
            )
        data = r.json()
        out.extend(data.get("metrics", []))
        token = data.get("nextPageToken")
        if not token:
            break
        url = f"{base}?view=FULL&pageToken={token}"
    return out


def looks_tpu_metric(metric: dict[str, Any]) -> bool:
    hay = " ".join(
        [
            metric.get("metric", "") or "",
            metric.get("displayName", "") or "",
            json.dumps(metric, separators=(",", ":")).lower(),
        ]
    ).lower()
    patterns = [
        "tpu",
        "tensor processing unit",
        "v4",
        "v5e",
        "v5p",
        "v6e",
        "preemptible tpu",
        "tpu cores",
        "tpu v4",
        "tpu v5",
        "tpu v6",
    ]
    return any(p in hay for p in patterns)


def collect_tpu_quota_metrics(
    session: AuthorizedSession, project_number: str, try_all_services: bool = False, verbose: bool = False
) -> list[dict[str, Any]]:
    candidates = ["tpu.googleapis.com", "cloudtpu.googleapis.com", "compute.googleapis.com"]
    metrics = []
    for svc in candidates:
        try:
            m = su_list_consumer_quota_metrics(session, project_number, svc)
            if verbose:
                print(f"Checked {svc}: {len(m)} quota metrics")
            metrics.extend(m)
        except RuntimeError as e:
            if verbose:
                print(f"{svc} error: {e}")

    tpu_like = [m for m in metrics if looks_tpu_metric(m)]
    if tpu_like and not try_all_services:
        return tpu_like

    all_enabled = su_list_enabled_services(session, project_number)
    if verbose:
        print(f"Enabled services: {len(all_enabled)}")
    for svc in all_enabled:
        try:
            m = su_list_consumer_quota_metrics(session, project_number, svc)
            tpu_m = [x for x in m if looks_tpu_metric(x)]
            if tpu_m:
                if verbose:
                    print(f"Found {len(tpu_m)} TPU-like metrics under {svc}")
                metrics.extend(tpu_m)
        except RuntimeError as e:
            if verbose:
                print(f"{svc} error: {e}")
    return [m for m in metrics if looks_tpu_metric(m)]


def normalize_quota_buckets(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for m in metrics:
        metric = m.get("metric")
        display = m.get("displayName")
        for lim in m.get("consumerQuotaLimits", []) or []:
            unit = lim.get("unit", "")
            limit_name = lim.get("name", "")
            for b in lim.get("quotaBuckets", []) or []:
                dims = b.get("dimensions", {}) or {}
                eff = b.get("effectiveLimit", None)
                if eff is None:
                    continue
                try:
                    eff_int = int(eff)
                except Exception:
                    try:
                        eff_int = int(str(eff))
                    except Exception:
                        continue
                rows.append(
                    {
                        "metric": metric,
                        "displayName": display,
                        "unit": unit,
                        "limit_name": limit_name,
                        "dimensions": dims,
                        "effectiveLimit": eff_int,
                    }
                )
    return rows


def zone_from_dims(dims: dict[str, str]) -> str | None:
    if "zone" in dims:
        return dims["zone"]
    loc = dims.get("location")
    if loc and re.match(r"^[a-z]+-[a-z0-9]+[0-9]-[a-z]$", loc):
        return loc
    return None


def safe_classify_tpu_type(texts: list[str], dims: dict[str, str]) -> str | None:
    hay = " ".join([t for t in texts if t]).lower()
    for k in ("tpu_type", "accelerator_type", "family", "tpu_family", "type"):
        if dims.get(k):
            hay += " " + str(dims[k]).lower()
    for key in ["v6e", "v5p", "v5e", "v4"]:
        if re.search(rf"\b{key}\b", hay):
            return key
    if "v5-e" in hay or "v5 e" in hay:
        return "v5e"
    if "v5p" in hay:
        return "v5p"
    if "v4" in hay:
        return "v4"
    return None


def is_preemptible(texts: list[str], dims: dict[str, str]) -> bool | None:
    hay = " ".join([t for t in texts if t]).lower()
    if "preemptible" in hay or "preempt" in hay or "spot" in hay:
        return True
    if "on-demand" in hay or "ondemand" in hay:
        return False
    return None


def tpu_list_locations(session: AuthorizedSession, project_id: str) -> list[str]:
    url = f"https://tpu.googleapis.com/v2/projects/{project_id}/locations"
    locs = []
    while True:
        r = session.get(url)
        if r.status_code == 404:
            return []
        if r.status_code != 200:
            raise RuntimeError(f"TPU locations.list error {r.status_code}: {r.text}")
        data = r.json()
        for le in data.get("locations", []) or []:
            name = le.get("name", "")
            parts = name.split("/")
            if len(parts) >= 4:
                locs.append(parts[-1])
        token = data.get("nextPageToken")
        if not token:
            break
        url = f"https://tpu.googleapis.com/v2/projects/{project_id}/locations?pageToken={token}"
    return sorted(set(locs))


def tpu_list_accelerator_types(session: AuthorizedSession, project_id: str, zone: str) -> list[str]:
    url = f"https://tpu.googleapis.com/v2/projects/{project_id}/locations/{zone}/acceleratorTypes"
    types = []
    while True:
        r = session.get(url)
        if r.status_code in (404, 403):
            return []
        if r.status_code != 200:
            raise RuntimeError(f"TPU acceleratorTypes.list error for {zone} {r.status_code}: {r.text}")
        data = r.json()
        for t in data.get("acceleratorTypes", []) or []:
            tname = t.get("type") or t.get("name", "").split("/")[-1]
            if tname:
                types.append(tname)
        token = data.get("nextPageToken")
        if not token:
            break
        url = f"https://tpu.googleapis.com/v2/projects/{project_id}/locations/{zone}/acceleratorTypes?pageToken={token}"
    return sorted(set(types))


def region_from_zone(zone: str) -> str:
    return "-".join(zone.split("-")[:2])


def family_from_acc_type_name(acc_type: str) -> str | None:
    if acc_type.startswith("v5litepod-"):
        return "v5e"
    if acc_type.startswith("v5p-"):
        return "v5p"
    if acc_type.startswith("v6e-"):
        return "v6e"
    if acc_type.startswith("v4-"):
        return "v4"
    return None


def parse_sizes_for_family(acc_types: list[str], family: str) -> set[int]:
    prefix = "v5litepod-" if family == "v5e" else f"{family}-"
    out: set[int] = set()
    for t in acc_types:
        if t.startswith(prefix):
            try:
                size = int(t.split("-")[-1])
                out.add(size)
            except Exception:
                pass
    return out


def make_tpu_node_type(generation: str, count: int, target_count: int, key_name: str) -> dict[str, dict[str, Any]]:
    slice_gen_name = "v5litepod" if generation == "v5e" else generation
    if "serve" in generation:
        slice_gen_name = generation.replace("-serve", "")
    return {
        key_name: {
            "min_workers": target_count,
            "max_workers": 1024,
            "resources": {
                "CPU": 120,
                "TPU": GENERATION_CONFIGS[generation].num_tpus,
                f"tpu-{generation}-{count}-head": 1,
            },
            "node_config": {
                "acceleratorType": f"{slice_gen_name}-{count}",
                "runtimeVersion": GENERATION_CONFIGS[generation].runtime_version,
                "schedulingConfig": {"preemptible": True},
            },
        }
    }


def make_tpu_slice_config(generation: str, count: int, target_count: int) -> dict[str, dict[str, Any]]:
    key_name = f"tpu_slice_{generation}_{count}"
    return make_tpu_node_type(generation, count, target_count, key_name)


def make_tpu_base_config(generation: str, count: int, min_workers: int = 0) -> dict[str, dict[str, Any]]:
    key_name = f"tpu_base_{generation}_{count}"
    return make_tpu_node_type(generation, count, min_workers, key_name)


def render_template_base(name: str, region: str, zone: str, project_id: str, bucket: str) -> str:
    if not cluster_template_path.exists():
        raise FileNotFoundError(f"Template file not found: {cluster_template_path}")
    with open(cluster_template_path) as f_template:
        template = jinja2.Template(f_template.read())
    config_dict = {
        "NAME": name,
        "REGION": region,
        "ZONE": zone,
        "PROJECT_ID": project_id,
        "BUCKET": bucket,
        "tpu_generation": "mixed",
        "min_workers": 0,
    }
    return template.render(**config_dict)


def generate_zone_config(
    zone: str,
    families_sizes: dict[str, set[int]],
    output_dir: Path,
    project_id: str,
    bucket: str | None = None,
) -> Path:
    """
    Generate a single cluster YAML for a zone including all available TPU families in that zone.
    families_sizes: { 'v4': {sizes...}, 'v5e': {...}, ... } (families absent or empty are skipped)
    """
    region = region_from_zone(zone)
    bucket = bucket or f"scaling-computation-foundation-{region}"
    name = f"easydel-{zone}"
    yaml_string = render_template_base(name, region, zone, project_id, bucket)

    # Append node types under available_node_types
    for fam in sorted(families_sizes.keys()):
        sizes = families_sizes[fam]
        if not sizes:
            continue
        gen_cfg = GENERATION_CONFIGS.get(fam)
        if not gen_cfg:
            continue

        base_size = int(gen_cfg.base_worker)
        if base_size not in sizes:
            base_size = min(sizes)  # smallest available as fallback

        base_cfg = make_tpu_base_config(fam, base_size, min_workers=0)
        base_str = yaml.dump(base_cfg, default_flow_style=False, indent=2)
        base_str = "\n  " + base_str.replace("\n", "\n  ")
        yaml_string += base_str

        for s in [ss for ss in gen_cfg.slices if ss in sizes]:
            slice_cfg = make_tpu_slice_config(fam, s, target_count=0)
            slice_str = yaml.dump(slice_cfg, default_flow_style=False, indent=2)
            slice_str = "\n  " + slice_str.replace("\n", "\n  ")
            yaml_string += slice_str

    lines = yaml_string.splitlines()
    lines = [line.rstrip() for line in lines]
    yaml_string = "\n".join(lines)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"easydel-{zone}.yaml"
    with open(out_path, "w") as f:
        f.write(yaml_string)

    logger.info(f"Generated zone config at {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Discover Cloud TPU zones/types and generate one cluster config per zone with all TPU families."
    )
    parser.add_argument("--project-id", help="Override GCP project ID")
    parser.add_argument("--output-dir", type=Path, default=this_path, help="Output directory for generated configs")
    parser.add_argument("--print-summary", action="store_true", help="Print a human-readable summary")
    parser.add_argument("--try-all-services", action="store_true", help="Scrape TPU quotas from all enabled services")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--families",
        nargs="*",
        default=["v4", "v5e", "v5p", "v6e"],
        choices=["v4", "v5e", "v5p", "v6e"],
        help="TPU families to include in auto-generation",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    global PROJECT_ID
    PROJECT_ID = args.project_id or get_default_project_id()
    if not PROJECT_ID:
        print("Could not determine project ID. Set GCP_PROJECT_ID or pass --project-id.", file=sys.stderr)
        sys.exit(1)
    logger.info(f"Using project ID: {PROJECT_ID}")

    try:
        project_number = get_project_number(PROJECT_ID)
    except Exception as e:
        print(f"Failed to get project number: {e}", file=sys.stderr)
        sys.exit(1)

    session, _ = authorized_session()

    acc_types_by_zone: dict[str, list[str]] = {}
    try:
        zones = tpu_list_locations(session, PROJECT_ID)
        for z in zones:
            try:
                acc_types_by_zone[z] = tpu_list_accelerator_types(session, PROJECT_ID, z)
            except Exception as e:
                if args.verbose:
                    print(f"accel types error for {z}: {e}")
    except Exception as e:
        logger.error(f"Could not list TPU locations/types: {e}")
        acc_types_by_zone = {}

    availability_by_zone: dict[str, dict[str, set[int]]] = {}
    for zone, types in acc_types_by_zone.items():
        if not types:
            continue
        fam_map: dict[str, set[int]] = {}
        for fam in args.families:
            sizes = parse_sizes_for_family(types, fam)
            if sizes:
                fam_map[fam] = sizes
        if fam_map:
            availability_by_zone[zone] = fam_map

    tpu_quota_rows = []
    unclassified_quota_rows = []
    try:
        metrics = collect_tpu_quota_metrics(
            session, project_number, try_all_services=args.try_all_services, verbose=args.verbose
        )
        buckets = normalize_quota_buckets(metrics)

        for r in buckets:
            z = zone_from_dims(r.get("dimensions", {}))
            if not z:
                continue
            texts = [r.get("metric"), r.get("displayName"), r.get("limit_name"), r.get("unit")]
            fam = safe_classify_tpu_type(texts, r.get("dimensions", {}))
            pre = is_preemptible(texts, r.get("dimensions", {}))
            eff = int(r.get("effectiveLimit", 0) or 0)
            if fam in {"v4", "v5e", "v5p", "v6e"} and eff > 0:
                tpu_quota_rows.append(
                    {
                        "zone": z,
                        "tpu_family": fam,
                        "preemptible": False if pre is None else bool(pre),
                        "effective_limit_cores": eff,
                    }
                )
            else:
                unclassified_quota_rows.append(
                    {
                        "zone": z,
                        "displayName": r.get("displayName"),
                        "metric": r.get("metric"),
                        "limit": r.get("effectiveLimit"),
                        "unit": r.get("unit"),
                        "dims": r.get("dimensions", {}),
                    }
                )
    except Exception as e:
        if args.verbose:
            logger.warning(f"Quota collection failed or incomplete: {e}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    total = len(availability_by_zone)
    generated_files: dict[str, str] = {}

    for zone in sorted(availability_by_zone.keys()):
        region = region_from_zone(zone)
        bucket = f"scaling-computation-foundation-{region}"
        try:
            out_path = generate_zone_config(
                zone=zone,
                families_sizes=availability_by_zone[zone],
                output_dir=args.output_dir,
                project_id=PROJECT_ID,
                bucket=bucket,
            )
            generated_files[zone] = str(out_path)
            success += 1
        except Exception as e:
            logger.error(f"Failed to generate config for zone {zone}: {e}")

    if args.print_summary:
        print("\nZones with TPU availability (families -> sizes):")
        if not availability_by_zone:
            print("  None found (TPU API disabled or no access).")
        else:
            for z in sorted(availability_by_zone.keys()):
                fam_sizes = ", ".join(
                    [f"{fam}:{sorted(list(sizes))}" for fam, sizes in sorted(availability_by_zone[z].items())]
                )
                print(f"  {z:>15} -> {fam_sizes}")
        if tpu_quota_rows:
            print("\nTPU quotas (zone-specific, cores):")
            for row in sorted(tpu_quota_rows, key=lambda r: (r["zone"], r["tpu_family"], r["preemptible"])):
                print(
                    f"  zone={row['zone']:>15} family={row['tpu_family']:<4} "
                    f"preemptible={str(row['preemptible']).lower():<5} cores={row['effective_limit_cores']}"
                )
        print(f"\nGenerated {success}/{total} zone configs into {args.output_dir}")

    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
