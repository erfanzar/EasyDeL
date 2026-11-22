"""Registry lookup must discover built-ins without a prior trainer import."""

import subprocess
import sys

import pytest


@pytest.mark.cpu_ok
@pytest.mark.parametrize("operation", ["lookup", "category"])
def test_fresh_registry_discovers_trainers(operation):
    code = """
import sys
from easydel.utils.registery import Registry
assert 'easydel.trainers' not in sys.modules
if sys.argv[1] == 'lookup':
    trainer = Registry.get_or_raise('trainer', 'sft')
else:
    trainer = Registry.get_category_registry('trainer')['sft']
assert trainer.__name__ == 'SFTTrainer'
assert Registry.get_or_raise('trainer-arguments', 'sft', wakeup=False).__name__ == 'SFTConfig'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, operation],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
