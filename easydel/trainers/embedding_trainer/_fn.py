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
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_scheduled_batch,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    scheduled_loss_cache_key,
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
    """Run one contrastive embedding training step.

    Encodes the ``query``, ``positive``, and optional ``negative``
    streams through the same encoder module, then computes the
    selected contrastive loss. With ``matryoshka_dims`` set, the loss
    is evaluated at every requested truncation dim and averaged
    (Matryoshka Representation Learning). The encoder is run in
    ``eval()`` mode inside the loss closure so dropout-style noise
    does not pollute the embeddings (it is, however, still
    differentiated against -- only the internal stochastic layers are
    disabled).

    The batch must carry tokenised ``query_*`` and ``positive_*``
    streams; ``negative_*`` is optional and triggers explicit hard
    negatives when present (in addition to in-batch negatives for
    InfoNCE/MNRL).

    Args:
        state: Encoder ``EasyDeLState`` being differentiated.
        batch: Dict with at minimum ``query_input_ids``,
            ``query_attention_mask``, ``positive_input_ids``,
            ``positive_attention_mask``. Optionally
            ``negative_input_ids`` / ``negative_attention_mask``.
        loss_type: One of ``"infonce"``, ``"triplet"``, ``"mnrl"``.
        temperature: Logit-scale temperature for InfoNCE/MNRL.
        margin: Triplet-loss margin.
        normalize: When ``True``, L2-normalises encoded embeddings
            before the similarity / distance computation.
        matryoshka_dims: Optional list of truncation dims for MRL.
            ``None`` disables MRL.
        learning_rate_fn: Schedule mapping step to learning rate.
        partition_spec: Sharding spec applied to the input batch.
        gradient_accumulation_steps: Gradient-accumulation factor;
            the batch must be evenly divisible.

    Returns:
        ``(new_state, metrics)`` where ``metrics`` is a ``LossMetrics``
        instance with ``loss``, ``accuracy`` (in-batch retrieval
        accuracy for InfoNCE/MNRL), and any per-dimension MRL
        diagnostics in ``other_metrics``.
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
        """Compute the embedding contrastive loss for one minibatch.

        Embeds the query, positive and (optionally) hard-negative
        sequences, dispatches to the configured base loss
        (``infonce`` / ``mnrl`` / ``triplet``), and applies the
        Matryoshka projection wrapper when requested.

        Args:
            tree: Encoder graphstate to differentiate against.
            minibatch: Minibatch with at least ``query_*`` and
                ``positive_*`` ``input_ids`` / ``attention_mask`` fields.

        Returns:
            ``(loss, metrics)`` with optional contrastive accuracy
            recorded in ``metrics.accuracy``.
        """
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


def _embedding_loss_values(
    *,
    module,
    batch: dict[str, jax.Array],
    loss_type: str,
    temperature: float,
    margin: float,
    normalize: bool,
    matryoshka_dims: list[int] | None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Run the encoder and compute the chosen contrastive loss.

    Computes embeddings for the concatenated query / positive / negative
    sequences in a single forward pass before splitting them again,
    which is friendlier to sharding than three separate forwards.

    Args:
        module: Encoder module.
        batch: Minibatch dict with ``query_*``, ``positive_*`` and
            optional ``negative_*`` ``input_ids`` / ``attention_mask``.
        loss_type: One of ``"infonce"``, ``"mnrl"``, ``"triplet"``.
        temperature: Temperature for InfoNCE / MNRL.
        margin: Margin for triplet loss.
        normalize: Whether embeddings are L2-normalised before the loss.
        matryoshka_dims: Optional list of nested dimensions for
            Matryoshka representation learning.

    Returns:
        ``(loss, extra_metrics)`` from the chosen base loss; for the
        triplet loss without negatives the loss is zero.
    """
    loss_fns = {
        "infonce": infonce_loss,
        "mnrl": mnrl_loss,
        "triplet": triplet_loss,
    }
    base_loss_fn = loss_fns[loss_type]

    input_ids = [batch["query_input_ids"], batch["positive_input_ids"]]
    attention_mask = [batch["query_attention_mask"], batch["positive_attention_mask"]]
    has_negatives = "negative_input_ids" in batch
    if has_negatives:
        input_ids.append(batch["negative_input_ids"])
        attention_mask.append(batch["negative_attention_mask"])

    q_size = batch["query_input_ids"].shape[0]
    p_size = batch["positive_input_ids"].shape[0]
    all_embeds = _embed_batch(
        module,
        jnp.concatenate(input_ids, axis=0),
        jnp.concatenate(attention_mask, axis=0),
        normalize=normalize,
    )
    q_embeds = all_embeds[:q_size]
    p_embeds = all_embeds[q_size : q_size + p_size]
    n_embeds = all_embeds[q_size + p_size :] if has_negatives else None

    loss_kwargs = {}
    if loss_type in ("infonce", "mnrl"):
        loss_kwargs["temperature"] = temperature
    elif loss_type == "triplet":
        loss_kwargs["margin"] = margin

    if matryoshka_dims is not None:
        return matryoshka_loss(
            base_loss_fn,
            q_embeds,
            p_embeds,
            n_embeds,
            matryoshka_dims,
            **loss_kwargs,
        )
    if n_embeds is not None:
        return base_loss_fn(q_embeds, p_embeds, n_embeds, **loss_kwargs)
    if loss_type == "triplet":
        return jnp.float32(0.0), {"fraction_active_triplets": jnp.float32(0.0)}
    return base_loss_fn(q_embeds, p_embeds, **loss_kwargs)


def _embedding_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the embedding scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering the loss type, temperature, margin, embedding
        normalisation flag, Matryoshka dim list, and partition spec.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("loss_type", "temperature", "margin", "normalize", "matryoshka_dims", "partition_spec"),
    )


def _make_embedding_scheduled_loss(call):
    """Build a SpectraX-scheduled embedding scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` carrying loss config.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready for
        :func:`spx.sxvalue_and_grad`.
    """
    loss_type = call.get("loss_type", "infonce")
    temperature = call.get("temperature", 0.05)
    margin = call.get("margin", 0.2)
    normalize = call.get("normalize", True)
    matryoshka_dims = call.get("matryoshka_dims")
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree, batch):
        """Compute the scalar contrastive loss inside the SpectraX scheduled VJP.

        Args:
            tree: Encoder graphstate to differentiate against.
            batch: Minibatch with query/positive/negative sequences.

        Returns:
            The scalar contrastive loss (extra metrics are dropped).
        """
        module = bind_scheduled_module(call, tree)
        batch = constrain_scheduled_batch(module, batch, partition_spec)
        loss, _extra_metrics = _embedding_loss_values(
            module=module,
            batch=batch,
            loss_type=loss_type,
            temperature=temperature,
            margin=margin,
            normalize=normalize,
            matryoshka_dims=matryoshka_dims,
        )
        return loss

    return scheduled_loss


register_scheduled_loss_adapter(
    embedding_training_step,
    ScheduledLossAdapter(
        name="embedding",
        make_loss=_make_embedding_scheduled_loss,
        make_cache_key=_embedding_scheduled_loss_cache_key,
    ),
)
