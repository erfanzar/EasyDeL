"""PageTable device mirror (lazy) stays consistent with the CPU table; LogprobsTensors helpers work."""

import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.outputs import LogprobsTensors
from easydel.inference.esurge.page_table import PageTable


def test_device_mirror_follows_commit_and_clear():
    pt = PageTable(max_num_reqs=4, max_num_pages_per_req=3, max_num_batched_tokens=16)
    assert np.all(np.asarray(pt.get_device_tensor()) == 0)
    pt.append_row([5, 6], row_idx=1)
    pt.commit(num_reqs=2)
    device = np.asarray(pt.page_table)
    assert device.shape == (4, 3)
    assert list(device[1, :2]) == [5, 6]
    pt.clear()
    assert np.all(np.asarray(pt.page_table) == 0)
    assert np.all(pt.get_cpu_tensor() == 0)


def test_logprobs_tensors_empty_and_tolists_roundtrip():
    empty = LogprobsTensors.empty(3, 2)
    assert empty.logprob_token_ids.shape == (3, 2)
    assert empty.logprobs.shape == (3, 2)
    assert empty.selected_token_ranks.shape == (3,)
    tensors = LogprobsTensors(
        logprob_token_ids=jnp.array([[1, 2], [3, 4]]),
        logprobs=jnp.array([[-0.5, -1.0], [-0.25, -2.0]]),
        selected_token_ranks=jnp.array([0, 1]),
    )
    lists = tensors.tolists()
    assert lists.logprob_token_ids == [[1, 2], [3, 4]]
    assert lists.sampled_token_ranks == [0, 1]
