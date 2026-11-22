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

"""Command-line interface for eray.

Provides commands for connecting TPU hosts into a Ray cluster and
managing cluster lifecycle.

Example:
    # Connect a TPU slice into a Ray cluster
    $ eray tpu connect --tpu-name my-tpu --project my-project --zone us-central2-b

    # Check cluster status
    $ eray tpu status --address 10.0.0.1:6379

    # Run a health check
    $ eray tpu health --address 10.0.0.1:6379

    # Disconnect (stop Ray on all hosts)
    $ eray tpu disconnect --tpu-name my-tpu --project my-project --zone us-central2-b
"""

from .main import cli

__all__ = ("cli",)
