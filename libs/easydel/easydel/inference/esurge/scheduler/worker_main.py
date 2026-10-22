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

"""Internal scheduler worker bootstrap, launched by file path with a child-only environment.

The parent supplies inherited pipe descriptors and an explicitly selected multiprocessing start method. No EasyDeL
module is imported until the subprocess has started with distributed initialization disabled and CPU-only placement.
"""

from __future__ import annotations

import multiprocessing
import sys
from multiprocessing.connection import Connection


def _run_scheduler(input_conn: Connection, output_conn: Connection, payload: bytes) -> None:
    """Deserialize scheduler configuration and run its host-only RPC loop.

    Args:
        input_conn: Pipe receiving commands from the coordinator.
        output_conn: Pipe returning responses to the coordinator.
        payload: Cloudpickled rank, scheduler class, and constructor arguments.
    """
    import cloudpickle

    from easydel.inference.esurge.scheduler.dp_scheduler import _scheduler_worker_process

    rank, scheduler_cls, scheduler_kwargs = cloudpickle.loads(payload)
    _scheduler_worker_process(rank, input_conn, output_conn, scheduler_cls, scheduler_kwargs)


def main() -> None:
    """Read inherited pipes and launch the selected scheduler process strategy."""
    input_conn = Connection(int(sys.argv[1]))
    output_conn = Connection(int(sys.argv[2]))
    start_method = sys.argv[3]
    try:
        payload = input_conn.recv_bytes()
        if start_method == "spawn":
            # This interpreter was already spawned with an explicit environment; no second process is needed.
            _run_scheduler(input_conn, output_conn, payload)
        else:
            process = multiprocessing.get_context(start_method).Process(
                target=_run_scheduler, args=(input_conn, output_conn, payload)
            )
            process.start()
            process.join()
            raise SystemExit(process.exitcode)
    finally:
        input_conn.close()
        output_conn.close()


if __name__ == "__main__":
    main()
