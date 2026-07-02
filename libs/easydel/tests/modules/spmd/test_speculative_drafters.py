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

"""Tests for DeepSpec-style speculative drafter modules."""

import jax
import jax.numpy as jnp
import spectrax as spx

import easydel as ed
from easydel.infra.factory import TaskType, registry


def _input_ids(batch_size: int = 2, seq_len: int = 6, vocab_size: int = 32):
    """Create dummy input IDs for testing speculative drafters.

    Args:
        batch_size: Number of sequences in the batch. Defaults to 2.
        seq_len: Number of tokens per sequence. Defaults to 6.
        vocab_size: Vocabulary size (excluding the last reserved token). Defaults to 32.

    Returns:
        jax.Array: Integer input IDs of shape (batch_size, seq_len) with values
        in ``[0, vocab_size - 2]``.
    """
    values = jnp.arange(batch_size * seq_len, dtype=jnp.int32).reshape(batch_size, seq_len)
    return values % (vocab_size - 1)


def _target_hidden_states(num_states: int, batch_size: int = 2, seq_len: int = 6, hidden_size: int = 16):
    """Create dummy target hidden states for testing speculative drafters.

    Args:
        num_states: Number of hidden-state tensors to generate.
        batch_size: Number of sequences in the batch. Defaults to 2.
        seq_len: Number of tokens per sequence. Defaults to 6.
        hidden_size: Dimension of each hidden state. Defaults to 16.

    Returns:
        tuple[jax.Array, ...]: A tuple of ``num_states`` random normal arrays,
        each of shape (batch_size, seq_len, hidden_size).
    """
    key = jax.random.PRNGKey(0)
    return tuple(
        jax.random.normal(jax.random.fold_in(key, idx), (batch_size, seq_len, hidden_size), dtype=jnp.float32)
        for idx in range(num_states)
    )


def test_eagle3_forward_shapes_and_registry():
    """Validate Eagle3Model forward shapes and registry entries.

    Instantiates an ``Eagle3Model``, runs a forward pass, and asserts that
    ``hidden_states``, ``draft_logits``, and ``logits`` shapes match expected
    dimensions. Also verifies ``compute_mtp_chain`` returns the correct MTP
    chain shape, and that the config and base module are registered under
    ``eagle3``.
    """
    cfg = ed.Eagle3Config(
        vocab_size=32,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(0, 1, 2, 3, 4),
    )
    model = ed.Eagle3Model(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    input_ids = _input_ids()
    outputs = model(
        input_ids=input_ids,
        target_hidden_states=_target_hidden_states(6),
        attention_mask=jnp.ones_like(input_ids, dtype=bool),
        return_logits=True,
    )

    assert outputs.hidden_states.shape == (2, 6, 16)
    assert outputs.draft_logits.shape == (2, 6, 32)
    assert outputs.logits.shape == (2, 6, 32)
    assert model.compute_mtp_chain(outputs, input_ids, 3).shape == (3, 2, 6, 32)
    assert registry.get_config("eagle3") is ed.Eagle3Config
    assert registry.get_module_registration(TaskType.BASE_MODULE, "eagle3").module is ed.Eagle3Model


def test_dspark_block_forward_with_markov_and_confidence():
    """Validate DSparkModel block forward with Markov and confidence heads.

    Instantiates a ``DSparkModel`` with Markov and confidence heads enabled,
    runs a forward pass, and asserts all output tensor shapes including
    ``block_logits``, ``draft_logits``, ``target_ids``, ``eval_mask``,
    ``block_keep_mask``, ``confidence_logits``, and ``aligned_target_logits``.
    Also verifies registry entries for ``dspark``.
    """
    cfg = ed.DSparkConfig(
        vocab_size=32,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(-1, 0),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
        markov_rank=4,
        markov_head_type="gated",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    model = ed.DSparkModel(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    input_ids = _input_ids()
    target_hidden_states = _target_hidden_states(2)
    outputs = model(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states,
        loss_mask=jnp.ones_like(input_ids, dtype=jnp.float32),
        target_last_hidden_states=target_hidden_states[-1],
    )

    assert outputs.block_logits.shape == (2, 2, 3, 32)
    assert outputs.draft_logits.shape == (2, 2, 3, 32)
    assert outputs.target_ids.shape == (2, 2, 3)
    assert outputs.eval_mask.shape == (2, 2, 3)
    assert outputs.block_keep_mask.shape == (2, 2)
    assert outputs.confidence_logits.shape == (2, 2, 3)
    assert outputs.aligned_target_logits.shape == (2, 2, 3, 32)
    assert registry.get_config("dspark") is ed.DSparkConfig
    assert registry.get_module_registration(TaskType.BASE_MODULE, "dspark").module is ed.DSparkModel


def test_dspark_block_forward_uses_target_logits_for_smaller_drafter():
    """Validate DSparkModel uses target logits when the drafter is smaller.

    When the drafter ``hidden_size`` is smaller than ``target_hidden_size``,
    the model accepts external ``target_logits`` and produces aligned logits.
    Asserts that ``block_logits`` and ``aligned_target_logits`` shapes are
    correct.
    """
    cfg = ed.DSparkConfig(
        vocab_size=32,
        hidden_size=8,
        target_hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(-1, 0),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
        markov_rank=4,
        markov_head_type="gated",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    model = ed.DSparkModel(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    input_ids = _input_ids()
    target_hidden_states = _target_hidden_states(2)
    target_logits = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 32), dtype=jnp.float32)
    outputs = model(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states,
        loss_mask=jnp.ones_like(input_ids, dtype=jnp.float32),
        target_last_hidden_states=target_hidden_states[-1],
        target_logits=target_logits,
    )

    assert outputs.block_logits.shape == (2, 2, 3, 32)
    assert outputs.aligned_target_logits.shape == (2, 2, 3, 32)


def test_dspark_accepts_qwen3_6_target_layer_selection():
    """Validate DSparkModel can project Qwen3-6 style target layer selections.

    Configures ``DSparkModel`` with large ``target_layer_ids`` (e.g., 65 layers)
    and asserts that ``project_target_hidden_states`` returns the correct
    projected shape.
    """
    cfg = ed.DSparkConfig(
        vocab_size=32,
        hidden_size=8,
        target_hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(-1, 15, 31, 47, 63),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
        markov_rank=4,
        markov_head_type="gated",
    )
    model = ed.DSparkModel(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)

    projected = model.project_target_hidden_states(_target_hidden_states(65))

    assert projected.shape == (2, 6, 8)


def test_dflash_accepts_qwen3_6_target_layer_selection():
    """Validate DFlashModel can project Qwen3-6 style target layer selections.

    Configures ``DFlashModel`` with large ``target_layer_ids`` and asserts that
    ``project_target_hidden_states`` returns the correct projected shape.
    """
    cfg = ed.DFlashConfig(
        vocab_size=32,
        hidden_size=8,
        target_hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(-1, 15, 31, 47, 63),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
    )
    model = ed.DFlashModel(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)

    projected = model.project_target_hidden_states(_target_hidden_states(65))

    assert projected.shape == (2, 6, 8)


def test_eagle3_accepts_qwen3_6_target_layer_selection():
    """Validate Eagle3Model can project Qwen3-6 style target layer selections.

    Configures ``Eagle3Model`` with large ``target_layer_ids`` and asserts that
    ``project_target_hidden_states`` returns the correct projected shape.
    """
    cfg = ed.Eagle3Config(
        vocab_size=32,
        hidden_size=8,
        target_hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(15, 31, 47, 55, 63),
    )
    model = ed.Eagle3Model(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)

    projected = model.project_target_hidden_states(_target_hidden_states(65))

    assert projected.shape == (2, 6, 8)


def test_dflash_block_forward_disables_dspark_extra_heads():
    """Validate DFlashModel disables extra heads present in DSpark.

    Instantiates ``DFlashModel`` and asserts that ``markov_rank`` and
    ``enable_confidence_head`` are disabled, the corresponding head modules are
    ``None``, and block-forward output shapes are correct. Also verifies
    registry entries for ``dflash``.
    """
    cfg = ed.DFlashConfig(
        vocab_size=32,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        target_layer_ids=(-1, 0),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
    )
    model = ed.DFlashModel(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    input_ids = _input_ids()
    outputs = model(
        input_ids=input_ids,
        target_hidden_states=_target_hidden_states(2),
        loss_mask=jnp.ones_like(input_ids, dtype=jnp.float32),
    )

    assert cfg.markov_rank == 0
    assert not cfg.enable_confidence_head
    assert model.markov_head is None
    assert model.confidence_head is None
    assert outputs.block_logits.shape == (2, 2, 3, 32)
    assert outputs.confidence_logits is None
    assert registry.get_config("dflash") is ed.DFlashConfig
    assert registry.get_module_registration(TaskType.BASE_MODULE, "dflash").module is ed.DFlashModel
