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

"""`eray autoscale` — Ray cluster-launcher configs for elastic TPU clusters.

Generates per-zone launcher YAMLs (all-spot by default; every TPU node type
carries the eray-canonical resource labels so the pool scheduler works) and
wraps `ray up/down/get-head-ip`. For reliable large slices, prefer the
QR-based `eray fleet` path; this one is for elastic many-small-slices use.
"""

from __future__ import annotations

from pathlib import Path

import click

from .utils import detect_project, error, info, success

DEFAULT_OUTPUT_DIR = Path("~/.eray/autoscale").expanduser()


def register(cli: click.Group) -> None:
    """Register the `autoscale` command group on the root CLI.

    Args:
        cli: The root click group.
    """

    @cli.group()
    def autoscale() -> None:
        """Ray cluster-launcher configs for elastic (autoscaled) TPU clusters."""

    @autoscale.command()
    @click.option("--project", "-p", default=None, help="GCP project (auto-detected).")
    @click.option("--zones", "-z", default=None, help="Comma-separated zones (default: every TPU location).")
    @click.option("--families", default="v4,v5e,v5p,v6e", show_default=True, help="TPU families to include.")
    @click.option("--image", default=None, help="Docker image for head/workers (default: easydel latest-tpu).")
    @click.option(
        "--spot/--on-demand", "spot", default=True, show_default=True, help="Capacity tier for TPU node types."
    )
    @click.option(
        "--output-dir", "-o", default=str(DEFAULT_OUTPUT_DIR), show_default=True, help="Where to write the YAMLs."
    )
    def generate(project, zones, families, image, spot, output_dir):
        """Generate per-zone cluster-launcher YAMLs.

        \b
        Examples:
            eray autoscale generate --zones us-central1-a
            eray autoscale generate --families v5p --on-demand
        """
        from ..provision.launcher import DEFAULT_IMAGE, generate_configs

        project = project or detect_project()
        if not project:
            raise click.ClickException("Could not resolve a GCP project; pass --project.")
        zone_list = [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        family_tuple = tuple(f.strip() for f in families.split(",") if f.strip())
        try:
            written = generate_configs(
                project,
                zones=zone_list,
                families=family_tuple,
                image=image or DEFAULT_IMAGE,
                spot=spot,
                output_dir=output_dir,
            )
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        if not written:
            info("no zones offered the requested TPU families; nothing written.")
            return
        for path in written:
            success(f"wrote {path}")

    @autoscale.command()
    @click.argument("config", type=click.Path(exists=True))
    @click.option("--yes", "-y", is_flag=True, default=False, help="Skip the ray up confirmation.")
    def up(config, yes):
        """Bring a launcher cluster up (`ray up CONFIG`)."""
        from ..provision.launcher import ray_up

        if not yes:
            click.confirm(f"ray up {config} — create/update this cluster?", abort=True)
        try:
            ray_up(config, yes=True)
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        success(f"cluster up: {config}")

    @autoscale.command()
    @click.argument("config", type=click.Path(exists=True))
    @click.option("--yes", "-y", is_flag=True, default=False, help="Skip the ray down confirmation.")
    def down(config, yes):
        """Tear a launcher cluster down (`ray down CONFIG`)."""
        from ..provision.launcher import ray_down

        if not yes:
            click.confirm(f"ray down {config} — terminate this cluster's nodes?", abort=True)
        try:
            ray_down(config, yes=True)
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        success(f"cluster down: {config}")

    @autoscale.command()
    @click.argument("config", type=click.Path(exists=True))
    def status(config):
        """Show a launcher cluster's head and resource usage."""
        from ..provision.launcher import launcher_head_ip

        try:
            head_ip = launcher_head_ip(config)
        except Exception as exc:
            error(f"cluster not reachable via ray get-head-ip: {exc}")
            raise click.ClickException(str(exc)) from exc
        info(f"head: {head_ip}")
        from ..provision.fleet import head_reachable

        info(f"ray head port: {'up' if head_reachable(head_ip) else 'down'}")
        info(f"resources: eray resources -a {head_ip}:6379")
