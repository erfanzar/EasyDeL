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

"""Gradient accumulation over packed pre-embedded multimodal batches (accelerator: runs JAX numerics)."""

from __future__ import annotations

import dataclasses

import jax
import ml_dtypes
import numpy as np
import pytest
from easydel.data.transforms.collators import collate_packed_embeds
from easydel.trainers.training_utils import _slice_batch_for_scheduled_step, minibatch_call, slice_host_minibatch
from jax import numpy as jnp

IMAGE_TOKEN_ID = 99
EMBED_DIM = 4
SEQ_LEN = 16
KEYS = ("input_ids", "image_embeds", "image_embed_positions", "image_embed_mask")


@dataclasses.dataclass
class _State:
    graphstate: jax.Array


def _position_id_fn(input_ids, mm_token_type_ids=None, image_grid_thw=None, attention_mask=None):
    length = np.asarray(input_ids).shape[-1]
    return np.arange(length, dtype=np.int32).reshape(1, 1, -1).repeat(3, axis=0), None


def _row(n_images: int, tag: float) -> dict:
    ids = [1, 2] + [IMAGE_TOKEN_ID] * n_images
    embeds = (np.arange(n_images * EMBED_DIM, dtype=np.float32) + tag).reshape(n_images, EMBED_DIM)
    return {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "attention_mask": np.ones(len(ids), dtype=np.int32),
        "labels": np.asarray(ids, dtype=np.int32),
        "image_embeds": [embeds.astype(ml_dtypes.bfloat16).tobytes()] if n_images else [],
        "embed_n_tok": [n_images] if n_images else [],
        "image_grid_thw": np.ones((n_images, 3), dtype=np.int32),
    }


def _packed_batch(counts: list[int]) -> dict:
    packs = [[_row(c, 0.25 * (w + 1))] for w, c in enumerate(counts)]
    return collate_packed_embeds(
        packs,
        0,
        SEQ_LEN,
        sum(counts) + 2,
        len(counts),
        image_token_id=IMAGE_TOKEN_ID,
        embed_dim=EMBED_DIM,
        position_id_fn=_position_id_fn,
    )


def _device_batch(counts: list[int]) -> dict:
    batch = _packed_batch(counts)
    return {k: jnp.asarray(batch[k]) for k in (*KEYS, "n_real_embeds")}


def _merge(input_ids, image_embeds):
    """Model-side merge: k-th placeholder (row-major) takes the k-th embed row."""
    mask = input_ids == IMAGE_TOKEN_ID
    padded = jnp.concatenate([jnp.zeros((1, image_embeds.shape[-1]), image_embeds.dtype), image_embeds], axis=0)
    updates = padded[jnp.cumsum(mask.reshape(-1))].reshape(*input_ids.shape, image_embeds.shape[-1])
    return jnp.where(mask[..., None], updates, 0.0), mask


def _loss(param, minibatch):
    merged, mask = _merge(minibatch["input_ids"], minibatch["image_embeds"])
    per_token = jnp.sum((param * merged) ** 2, axis=-1)
    loss = jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1)
    return loss, loss


def _placeholder_weight(_tree, minibatch):
    return jnp.sum(minibatch["input_ids"] == IMAGE_TOKEN_ID).astype(jnp.float32)


@pytest.mark.parametrize(
    ("counts", "size"), [([2, 0, 3, 1], 2), ([2, 0, 3, 1], 1), ([1, 1, 1, 1], 2), ([0, 4, 0, 0], 1)]
)
def test_scan_accumulation_matches_full_batch(counts, size):
    state = _State(graphstate=jnp.asarray(1.5, dtype=jnp.float32))
    batch = _device_batch(counts)
    grad_fn = jax.value_and_grad(_loss, has_aux=True)
    (full_loss, _), full_grads = grad_fn(state.graphstate, batch)

    grads, metrics = jax.jit(
        lambda g, b: minibatch_call(_State(graphstate=g), b, size, grad_fn, microbatch_weight_fn=_placeholder_weight)
    )(state.graphstate, batch)

    np.testing.assert_allclose(metrics, full_loss, rtol=1e-5)
    np.testing.assert_allclose(grads, full_grads, rtol=1e-5)


@pytest.mark.parametrize(("counts", "size"), [([2, 0, 3, 1], 2), ([3, 0, 0, 2], 1)])
def test_traced_slicer_matches_host_slicer(counts, size):
    batch_size = len(counts)
    host_batch = _packed_batch(counts)
    device_batch = _device_batch(counts)
    traced = jax.jit(lambda b, start: _slice_batch_for_scheduled_step(b, batch_size, start, size))

    for start in range(0, batch_size, size):
        expected = slice_host_minibatch({k: host_batch[k] for k in KEYS}, batch_size, start, size)
        got = traced(device_batch, jnp.asarray(start, dtype=jnp.int32))
        for key in KEYS:
            np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(expected[key]))


def test_scan_slicer_compiles_once_across_microbatches():
    counts = [2, 0, 3, 1]
    device_batch = _device_batch(counts)
    calls = []

    @jax.jit
    def traced(b, start):
        calls.append(1)
        return _slice_batch_for_scheduled_step(b, 4, start, 2)

    first = traced(device_batch, jnp.asarray(0, dtype=jnp.int32))
    second = traced(device_batch, jnp.asarray(2, dtype=jnp.int32))

    assert len(calls) == 1
    assert first["image_embeds"].shape == second["image_embeds"].shape == device_batch["image_embeds"].shape
