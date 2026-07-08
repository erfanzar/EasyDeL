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

"""Regression: MoE routing-hook auto-configuration must not mutate module structure.

``BaseMoeModule._configure_hooks_for_routing_strategy`` used to write the default
``refine_weights_hook`` back onto ``self.moe_hooks``. Assigning a new attribute on
an ``spx.Module`` bumps the graph epoch, which is an *illegal structural mutation*
when the forward runs under a readonly transform (the jitted SFT training step).

Models that pre-install a ``select_hook`` in ``__init__`` (deepseek_v3, hy_v3,
mistral4) skipped the auto-config branch and never tripped this. ``qwen3_moe``
leaves ``moe_hooks`` at the empty default and uses TOP_K routing, so the first
``moe_call`` inside the readonly training-step trace tried to rebind
``self.moe_hooks`` -> ``spectrax.core.errors.IllegalMutationError``.

The fix makes ``_configure_hooks_for_routing_strategy`` pure: it *returns* the
effective ``MoeFusedHooks`` and callers thread it through, so no attribute is
rebound during the forward.
"""

from __future__ import annotations

import jax.numpy as jnp
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _build_qwen3_moe_block():
    """Single-device (ep=1) Qwen3MoeSparseBlock; ep=1 avoids ragged_all_to_all on CPU."""
    from easydel.modules.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseBlock
    from easydel.modules.qwen3_moe.qwen3_moe_configuration import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    block = Qwen3MoeSparseBlock(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    return config, block


def test_configure_hooks_does_not_mutate_module():
    """Auto-configuring the routing hooks must be pure (no epoch bump, no rebind)."""
    from spectrax.core.module import _graph_epoch

    _, block = _build_qwen3_moe_block()

    epoch_before = _graph_epoch()
    hooks = block._configure_hooks_for_routing_strategy()
    epoch_after = _graph_epoch()

    # The whole point of the fix: configuring hooks is a pure computation.
    assert epoch_after == epoch_before, "configuring MoE hooks bumped the graph epoch (structural mutation)"

    # TOP_K + norm_topk_prob and no user select_hook -> a default refine hook is returned...
    assert hooks is not None
    assert hooks.refine_weights_hook is not None

    # ...but it is NOT written back onto the module.
    assert block.moe_hooks.refine_weights_hook is None


def test_moe_block_forward_under_readonly_transform_does_not_raise():
    """Forward under spx.jit (readonly) reproduces the SFT training-step trace path.

    On the buggy code this raised ``IllegalMutationError`` while tracing; the fix
    lets the trace complete and return finite outputs.
    """
    config, block = _build_qwen3_moe_block()
    x = jnp.ones((1, 4, config.hidden_size), dtype=jnp.float32)

    def _fwd(blk, inp):
        out, _router_logits = blk(inp)
        return out

    with config.mesh:
        out = spx.jit(_fwd)(block, x)

    assert out.shape == (1, 4, config.hidden_size)
    assert bool(jnp.all(jnp.isfinite(out)))
    # The readonly transform must not have rebound the hooks on the live module.
    assert block.moe_hooks.refine_weights_hook is None
