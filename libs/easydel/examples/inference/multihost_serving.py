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

"""Single-ingress multi-host eSurge serving (one model sharded across a pod).

On a multi-host pod the jitted decode step is a global-mesh collective: it
only runs when *every* process enters it together. But an HTTP request
lands on one host. Without coordination, rank 0 would enter the step and
block forever waiting for the other hosts — a silent deadlock.

The eSurge request plane fixes this. Set ``coordination="zmq"`` and rank 0
becomes the leader: it owns the scheduler and mirrors its runner-call
stream to every other host over ZeroMQ, so the whole pod steps in lockstep
even though requests only ever arrive at the leader's API server. Worker
ranks simply replay the leader's step stream.

Launch this same script on every host of the pod (one process per host)
with ``jax.distributed`` initialized — e.g. via eray, or the standard
``tpu_setup.sh`` multi-host bootstrap. Rank 0 serves the OpenAI API;
every other rank replays. The shared auth token must be identical on all
hosts (here it is derived from a literal; use a secret in production).

For data-parallel serving with N *independent* replicas of a model that
fits on a slice each (instead of one model sharded across the pod), see
``dp_replica_serving.py``.
"""

import easydel as ed


def main():
    import jax

    model_id = "Qwen/Qwen3.5-27B"
    max_length = 2**15
    max_concurrent_decodes = 32

    elm = ed.eLargeModel(
        {
            "model": {
                "name_or_path": model_id,
                "tokenizer": model_id,
                "task": "auto-bind",
            },
            "loader": {
                "dtype": "bfloat16",
                "param_dtype": "bfloat16",
                "precision": "fastest",
                "verbose": True,
            },
            "sharding": {
                # Canonical 6-axis mesh. Shard the single model across every
                # chip of the pod: tp fills all visible devices across all
                # hosts. Set pp>1 instead to pipeline-shard across stages.
                "axis_dims": (1, 1, 1, 1, -1, 1),
                "axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp"),
                "auto_shard_model": True,
            },
            "esurge": {
                "runtime": {
                    "max_model_len": max_length,
                    "max_num_seqs": max_concurrent_decodes,
                },
                "cache": {
                    "hbm_utilization": 0.85,
                    "page_size": 128,
                    "enable_prefix_caching": True,
                },
                "distributed": {
                    # Rank 0 owns the scheduler and mirrors its step stream to
                    # the other hosts; requests arriving at rank 0 no longer
                    # deadlock the pod. Every host must set the SAME token.
                    "coordination": "zmq",
                    "distributed_auth_token": "change-me-same-on-all-hosts",
                },
            },
            "eval": {"max_new_tokens": 8192},
        }
    )

    surge = elm.build_esurge()

    if jax.process_index() == 0:
        # Leader host: own the scheduler and expose the OpenAI-compatible API.
        ed.eSurgeApiServer(surge).fire()
    else:
        # Worker hosts: no server — replay the leader's step stream until the
        # engine is torn down. build_esurge already started the replay thread.
        surge._worker_replay_thread.join()


if __name__ == "__main__":
    main()
