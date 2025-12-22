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
Test runner for eLargeModel YAML configurations.

This module provides pytest-based tests that run the elarge script with
various YAML configurations, mirroring the trainer tests in tests/trainers/.

Usage:
    pytest tests/elarge_configs/test_elarge_runner.py -v
    pytest tests/elarge_configs/test_elarge_runner.py::test_sft -v
    pytest tests/elarge_configs/test_elarge_runner.py -k "dpo or orpo" -v
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).parent
ELARGE_MODULE = "easydel.scripts.elarge"


def run_elarge_config(config_name: str, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run the elarge script with a given config file.

    Args:
        config_name: Name of the YAML config file (without directory).
        dry_run: If True, only parse and validate without executing.

    Returns:
        CompletedProcess with stdout/stderr captured.
    """
    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cmd = [sys.executable, "-m", ELARGE_MODULE, str(config_path)]
    if dry_run:
        cmd.append("--dry-run")

    env = os.environ.copy()

    # Ensure caches are writable in sandboxed/CI environments (mirrors tests/inference/*).
    # Only redirect when the default $HOME/.cache isn't writable.
    default_cache_root = Path.home() / ".cache"
    if not default_cache_root.exists() or not os.access(default_cache_root, os.W_OK):
        cache_root = (Path.cwd() / "tmp-files" / "elarge-tests" / "cache").resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
        env.setdefault("HF_HOME", str(cache_root / "huggingface"))
        env.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface-datasets"))
        env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
        env.setdefault("COMPILE_FUNC_DIR", str(cache_root / "ejkernel" / "ejit_compiled_functions"))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout for training
        env=env,
    )
    return result


class TestElargeDryRun:
    """Test that all configs can be parsed and validated (dry-run)."""

    @pytest.mark.parametrize(
        "config_name",
        [
            "sft.yaml",
            "dpo.yaml",
            "orpo.yaml",
            "distillation.yaml",
            "reward.yaml",
            "bco.yaml",
            "cpo.yaml",
            "kto.yaml",
            "gkd.yaml",
            "grpo.yaml",
            "gfpo.yaml",
            "gspo.yaml",
            "xpo.yaml",
            "nash_md.yaml",
        ],
    )
    def test_dry_run(self, config_name: str):
        """Validate config parsing without execution."""
        result = run_elarge_config(config_name, dry_run=True)
        assert result.returncode == 0, f"Dry-run failed for {config_name}:\n{result.stderr}"


class TestElargeTraining:
    """End-to-end training tests using YAML configs.

    These tests actually run training for a few steps to verify the full pipeline.
    They are marked as slow and can be skipped with: pytest -m "not slow"
    """

    @pytest.mark.slow
    def test_sft(self):
        """Test SFT training via YAML config."""
        result = run_elarge_config("sft.yaml")
        assert result.returncode == 0, f"SFT training failed:\n{result.stderr}"

    @pytest.mark.slow
    def test_dpo(self):
        """Test DPO training via YAML config."""
        result = run_elarge_config("dpo.yaml")
        assert result.returncode == 0, f"DPO training failed:\n{result.stderr}"

    @pytest.mark.slow
    def test_orpo(self):
        """Test ORPO training via YAML config."""
        result = run_elarge_config("orpo.yaml")
        assert result.returncode == 0, f"ORPO training failed:\n{result.stderr}"

    @pytest.mark.slow
    def test_distillation(self):
        """Test distillation training via YAML config."""
        result = run_elarge_config("distillation.yaml")
        assert result.returncode == 0, f"Distillation training failed:\n{result.stderr}"


if __name__ == "__main__":
    # Allow running directly: python test_elarge_runner.py [config_name] [--dry-run]
    import argparse

    parser = argparse.ArgumentParser(description="Run elarge YAML config tests")
    parser.add_argument(
        "config",
        nargs="?",
        default="all",
        help="Config name (sft, dpo, orpo, distillation) or 'all'",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only parse and validate")
    args = parser.parse_args()

    configs = {
        "sft": "sft.yaml",
        "dpo": "dpo.yaml",
        "orpo": "orpo.yaml",
        "distillation": "distillation.yaml",
        "reward": "reward.yaml",
        "bco": "bco.yaml",
        "cpo": "cpo.yaml",
        "kto": "kto.yaml",
        "gkd": "gkd.yaml",
        "grpo": "grpo.yaml",
        "gfpo": "gfpo.yaml",
        "gspo": "gspo.yaml",
        "xpo": "xpo.yaml",
        "nash_md": "nash_md.yaml",
    }

    if args.config == "all":
        to_run = list(configs.values())
    elif args.config in configs:
        to_run = [configs[args.config]]
    else:
        print(f"Unknown config: {args.config}")
        print(f"Available: {list(configs.keys())} or 'all'")
        sys.exit(1)

    for config_name in to_run:
        print(f"\n{'='*60}")
        print(f"Running: {config_name} (dry_run={args.dry_run})")
        print("=" * 60)

        try:
            result = run_elarge_config(config_name, dry_run=args.dry_run)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.returncode != 0:
                print(f"FAILED with return code {result.returncode}")
                sys.exit(result.returncode)
            print(f"SUCCESS: {config_name}")
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {config_name}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    print("\nAll tests passed!")
