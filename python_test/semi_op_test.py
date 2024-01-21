import flax.linen
import torch
import jax
from jax import numpy as jnp


def test_smi_op():
    tensor1 = torch.randn(
        (1, 8, 128, 32)
    )

    tensor2 = torch.randn(
        (1, 8, 128, 32)
    )

    res_t = tensor1 @ tensor2.transpose(-1, -2)

    array1 = tensor1.cpu().detach().numpy().transpose(0, 2, 1, 3)
    array2 = tensor2.cpu().detach().numpy().transpose(0, 2, 1, 3)
    res_j = jnp.einsum("...qhd,...khd->...hqk", array1, array2)

    print(res_j[..., -1, 0])
    print(res_t[..., -1, 0])
    assert jnp.allclose(res_t.cpu().detach().numpy(), res_j, atol=1e-5)
