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

"""``python -m easydel.inference.esurge.serving {serve,check} --config deployment.yaml``.

``serve`` runs on every host of the slice (start it on all hosts with ``eray tpu job start``). ``check`` only
validates the YAML; run it with ``ENABLE_DISTRIBUTED_INIT=0`` off-slice so importing easydel does not join a world.
"""

from __future__ import annotations

import argparse
import sys

from .config import ServingConfig, ServingConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m easydel.inference.esurge.serving", description=__doc__)
    parser.add_argument("command", choices=("serve", "check"))
    parser.add_argument("--config", required=True, help="Serving YAML (see easydel/docs/esurge_serving.md).")
    args = parser.parse_args(argv)
    try:
        config = ServingConfig.load(args.config)
    except (OSError, ServingConfigError) as exc:
        print(f"invalid serving config {args.config}: {exc}", file=sys.stderr)
        return 2
    if args.command == "check":
        print(f"ok: {config.model.checkpoint} on mesh {config.model.sharding_axis_dims}, port {config.server.port}")
        return 0
    from .app import serve

    serve(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
