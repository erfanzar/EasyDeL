"""Public packed-SSM exports and paged attention flag integration, without arrays."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_ok


def test_packed_ssm_package_exports_preserve_implementations():
    from easydel.operations import kernels
    from easydel.operations.kernels.ssm1 import packed_ssm1_fwd
    from easydel.operations.kernels.ssm2 import packed_ssm2_fwd

    assert kernels.packed_ssm1_fwd is packed_ssm1_fwd
    assert kernels.packed_ssm2_fwd is packed_ssm2_fwd
    assert {"packed_ssm1_fwd", "packed_ssm2_fwd"} <= set(kernels.__all__)


@pytest.mark.parametrize("setting,expected", [(None, True), ("0", False), ("false", False), ("yes", True)])
def test_paged_adapters_share_registered_flag_default_and_values(setting, expected):
    environment = os.environ.copy()
    name = "EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH"
    if setting is None:
        environment.pop(name, None)
    else:
        environment[name] = setting
    code = """
import sys
from easydel.operations import _operation_impl
from easydel.operations.kernels import multi_latent_ragged_page_attention, paged_flash_attention

expected = sys.argv[1] == 'True'
assert _operation_impl.ENABLE_DP_LOCAL_PAGE_PATH is expected
assert paged_flash_attention.ENABLE_DP_LOCAL_PAGE_PATH is expected
assert multi_latent_ragged_page_attention.ENABLE_DP_LOCAL_PAGE_PATH is expected
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(expected)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
