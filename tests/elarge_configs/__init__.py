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

"""
eLargeModel YAML configuration tests.

This package contains YAML configurations that test the `python -m easydel.scripts.elarge`
unified runner. Each YAML file corresponds to a trainer type and mirrors the Python-based
tests in tests/trainers/.

Configs:
    - sft.yaml: Supervised Fine-Tuning
    - dpo.yaml: Direct Preference Optimization
    - orpo.yaml: Odds Ratio Preference Optimization
    - distillation.yaml: Knowledge Distillation
    - sdpo.yaml: Self-Distillation Policy Optimization

Usage:
    # Dry-run validation of all configs
    pytest tests/elarge_configs/test_elarge_runner.py::TestElargeDryRun -v

    # Full training tests (slow)
    pytest tests/elarge_configs/test_elarge_runner.py::TestElargeTraining -v

    # Run specific trainer
    pytest tests/elarge_configs/test_elarge_runner.py -k sft -v

    # Direct CLI usage
    python tests/elarge_configs/test_elarge_runner.py sft --dry-run
    python tests/elarge_configs/test_elarge_runner.py all
"""
