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

"""Contrastive loss functions and training step for EmbeddingTrainer."""

from __future__ import annotations

import typing as tp

import jax
import optax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics

from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)


def infonce_loss(
    query_embeds: jax.Array,
    positive_embeds: jax.Array,
    negative_embeds: jax.Array | None = None,
    temperature: float = 0.05,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute InfoNCE contrastive loss with in-batch negatives.

    For each query, the corresponding positive is the target and all other
    positives in the batch serve as negatives. If explicit hard negatives
    are provided, they are appended to the candidate pool.

    Args:
        query_embeds: Query embeddings ``(batch, dim)``, L2-normalized.
        positive_embeds: Positive embeddings ``(batch, dim)``, L2-normalized.
        negative_embeds: Optional hard negatives ``(batch, dim)`` or
            ``(batch * k, dim)`` for k negatives per query.
        temperature: Temperature scaling for similarity logits.

    Returns:
        Tuple of (scalar loss, metrics dict with accuracy).
    """
    if negative_embeds is not None:
        candidates = jnp.concatenate([positive_embeds, negative_embeds], axis=0)
    else:
        candidates = positive_embeds

    similarity = query_embeds @ candidates.T / temperature
    batch_size = query_embeds.shape[0]
    labels = jnp.arange(batch_size)

    loss = optax.softmax_cross_entropy_with_integer_labels(similarity, labels)
    loss = jnp.mean(loss)

    predictions = jnp.argmax(similarity, axis=-1)
    accuracy = jnp.mean(predictions == labels)

    return loss, {"contrastive_accuracy": accuracy}


def triplet_loss(
    query_embeds: jax.Array,
    positive_embeds: jax.Array,
    negative_embeds: jax.Array,
    margin: float = 0.2,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute triplet margin loss.

    ``loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)``

    where ``d`` is the L2 distance (for normalized embeddings, this is
    equivalent to ``sqrt(2 - 2 * cosine_sim)``).

    Args:
        query_embeds: Anchor embeddings ``(batch, dim)``.
        positive_embeds: Positive embeddings ``(batch, dim)``.
        negative_embeds: Negative embeddings ``(batch, dim)``.
        margin: Margin for triplet loss.

    Returns:
        Tuple of (scalar loss, metrics dict).
    """
    pos_dist = jnp.sum((query_embeds - positive_embeds) ** 2, axis=-1)
    neg_dist = jnp.sum((query_embeds - negative_embeds) ** 2, axis=-1)
    loss = jnp.maximum(pos_dist - neg_dist + margin, 0.0)
    loss = jnp.mean(loss)

    fraction_active = jnp.mean((pos_dist - neg_dist + margin) > 0.0)
    return loss, {"fraction_active_triplets": fraction_active}


def mnrl_loss(
    query_embeds: jax.Array,
    positive_embeds: jax.Array,
    negative_embeds: jax.Array | None = None,
    temperature: float = 0.05,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute Multiple Negatives Ranking Loss (MNRL).

    Equivalent to InfoNCE with cosine similarity. This is the standard
    loss used by sentence-transformers' ``MultipleNegativesRankingLoss``.

    Args:
        query_embeds: Query embeddings ``(batch, dim)``, L2-normalized.
        positive_embeds: Positive embeddings ``(batch, dim)``, L2-normalized.
        negative_embeds: Optional hard negatives.
        temperature: Temperature scaling.

    Returns:
        Tuple of (scalar loss, metrics dict with accuracy).
    """
    return infonce_loss(query_embeds, positive_embeds, negative_embeds, temperature)


def matryoshka_loss(
    loss_fn: tp.Callable,
    query_embeds: jax.Array,
    positive_embeds: jax.Array,
    negative_embeds: jax.Array | None,
    dims: list[int],
    **loss_kwargs,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute Matryoshka Representation Learning loss.

    Evaluates the contrastive loss at multiple embedding dimensions and
    averages the results, encouraging the model to produce useful
    embeddings at every truncation level.

    Args:
        loss_fn: Base loss function (infonce_loss, triplet_loss, etc.).
        query_embeds: Full-dimension query embeddings ``(batch, full_dim)``.
        positive_embeds: Full-dimension positive embeddings.
        negative_embeds: Optional full-dimension negative embeddings.
        dims: List of truncation dimensions, e.g. ``[64, 128, 256, 768]``.
        **loss_kwargs: Additional kwargs for the loss function.

    Returns:
        Tuple of (averaged scalar loss, metrics dict).
    """
    total_loss = jnp.float32(0.0)
    all_metrics: dict[str, jax.Array] = {}

    for dim in dims:
        q = query_embeds[:, :dim]
        p = positive_embeds[:, :dim]
        n = negative_embeds[:, :dim] if negative_embeds is not None else None

        q = q / jnp.clip(jnp.linalg.norm(q, axis=-1, keepdims=True), min=1e-12)
        p = p / jnp.clip(jnp.linalg.norm(p, axis=-1, keepdims=True), min=1e-12)
        if n is not None:
            n = n / jnp.clip(jnp.linalg.norm(n, axis=-1, keepdims=True), min=1e-12)

        dim_loss, dim_metrics = loss_fn(q, p, n, **loss_kwargs) if n is not None else loss_fn(q, p, **loss_kwargs)
        total_loss = total_loss + dim_loss
        for k, v in dim_metrics.items():
            all_metrics[f"{k}_dim{dim}"] = v

    avg_loss = total_loss / len(dims)
    return avg_loss, all_metrics


def _embed_batch(
    module,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    normalize: bool = True,
) -> jax.Array:
    """Forward-pass a batch through the embedding model and return embeddings.

    Args:
        module: EasyDeL embedding module (merged state).
        input_ids: Token IDs ``(batch, seq_len)``.
        attention_mask: Attention mask ``(batch, seq_len)``.
        normalize: Whether to L2-normalize output embeddings.

    Returns:
        Embeddings array ``(batch, hidden_size)``.
    """
    outputs = module(input_ids=input_ids, attention_mask=attention_mask)
    embeds = outputs.embeddings
    if normalize:
        embeds = embeds / jnp.clip(jnp.linalg.norm(embeds, axis=-1, keepdims=True), min=1e-12)
    return embeds


def embedding_training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    loss_type: str = "infonce",
    temperature: float = 0.05,
    margin: float = 0.2,
    normalize: bool = True,
    matryoshka_dims: list[int] | None = None,
    learning_rate_fn: optax.Schedule | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[EasyDeLState, LossMetrics]:
    """Training step for contrastive embedding learning.

    Expects batch to contain:
    - ``query_input_ids``, ``query_attention_mask``
    - ``positive_input_ids``, ``positive_attention_mask``
    - Optionally: ``negative_input_ids``, ``negative_attention_mask``

    Args:
        state: Current model state.
        batch: Dictionary of batched inputs.
        loss_type: One of ``"infonce"``, ``"triplet"``, ``"mnrl"``.
        temperature: Temperature for InfoNCE/MNRL.
        margin: Margin for triplet loss.
        normalize: Whether to L2-normalize embeddings.
        matryoshka_dims: Optional MRL dimensions.
        learning_rate_fn: Learning rate schedule.
        partition_spec: Sharding spec for the batch.
        gradient_accumulation_steps: Gradient accumulation count.

    Returns:
        Tuple of (updated state, loss metrics).
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    loss_fns = {
        "infonce": infonce_loss,
        "mnrl": mnrl_loss,
        "triplet": triplet_loss,
    }
    base_loss_fn = loss_fns[loss_type]

    def loss_fn(tree, minibatch):
        module = state.merge(tree)
        module.eval()

        q_embeds = _embed_batch(
            module,
            minibatch["query_input_ids"],
            minibatch["query_attention_mask"],
            normalize=normalize,
        )
        p_embeds = _embed_batch(
            module,
            minibatch["positive_input_ids"],
            minibatch["positive_attention_mask"],
            normalize=normalize,
        )

        n_embeds = None
        if "negative_input_ids" in minibatch:
            n_embeds = _embed_batch(
                module,
                minibatch["negative_input_ids"],
                minibatch["negative_attention_mask"],
                normalize=normalize,
            )

        loss_kwargs = {}
        if loss_type in ("infonce", "mnrl"):
            loss_kwargs["temperature"] = temperature
        elif loss_type == "triplet":
            loss_kwargs["margin"] = margin

        if matryoshka_dims is not None:
            loss, extra_metrics = matryoshka_loss(
                base_loss_fn,
                q_embeds,
                p_embeds,
                n_embeds,
                matryoshka_dims,
                **loss_kwargs,
            )
        elif n_embeds is not None:
            loss, extra_metrics = base_loss_fn(q_embeds, p_embeds, n_embeds, **loss_kwargs)
        else:
            if loss_type == "triplet":
                loss = jnp.float32(0.0)
                extra_metrics = {"fraction_active_triplets": jnp.float32(0.0)}
            else:
                loss, extra_metrics = base_loss_fn(q_embeds, p_embeds, **loss_kwargs)

        metrics = LossMetrics(
            loss=loss,
            accuracy=extra_metrics.get("contrastive_accuracy", None),
            other_metrics=extra_metrics,
        )
        return loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
    )

    state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=None,
        metrics=update_metrics(
            metrics=metrics,
            learning_rate_fn=learning_rate_fn,
            step=state.step,
            gradients=gradients,
        ),
    )
    return state, metrics
