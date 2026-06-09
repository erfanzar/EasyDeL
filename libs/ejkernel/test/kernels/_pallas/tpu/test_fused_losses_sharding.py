# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shard-map coverage for TPU Pallas fused CE/KL losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from ejkernel.modules.operations import fused_cross_entropy, fused_kl_divergence


def _has_tpu_devices() -> bool:
    """Return whether at least four TPU devices are available for mesh tests."""
    try:
        return jax.default_backend() == "tpu" and jax.device_count() >= 4
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu_devices(), reason="requires at least 4 TPU devices")


def _mesh():
    """Build the 4-device mesh used for DP/FSDP, sequence, and vocab axes."""
    return Mesh(np.array(jax.devices()[:4], dtype=object).reshape(1, 1, 2, 2), ("dp", "fsdp", "sp", "tp"))


def _inputs():
    """Create deterministic small tensors with sparse row weights for TPU tests."""
    batch, seq, vocab = 1, 64, 1024
    logits = (jax.random.normal(jax.random.PRNGKey(0), (batch, seq, vocab)) * 0.25).astype(jnp.bfloat16)
    targets = jax.random.randint(jax.random.PRNGKey(1), (batch, seq), 0, vocab, dtype=jnp.int32)
    weights = (jnp.arange(seq)[None, :] < 16).astype(jnp.float32)
    student = (jax.random.normal(jax.random.PRNGKey(2), (batch, seq, vocab)) * 0.25).astype(jnp.bfloat16)
    teacher = (jax.random.normal(jax.random.PRNGKey(3), (batch, seq, vocab)) * 0.25).astype(jnp.bfloat16)
    return logits, targets, weights, student, teacher


def test_xla_vocab_parallel_mesh_matches_reference():
    """Check XLA shard_map vocab-parallel reductions match replicated reference."""
    logits, targets, weights, student, teacher = _inputs()
    logits_spec = P(("dp", "fsdp"), "sp", "tp")
    rows_spec = P(("dp", "fsdp"), "sp")

    with _mesh() as mesh:
        ce_ref = fused_cross_entropy(logits, targets, weights, platform="xla", reduction="mean").loss
        ce_mesh = fused_cross_entropy(
            logits,
            targets,
            weights,
            platform="xla",
            reduction="mean",
            mesh=mesh,
            in_specs=(logits_spec, rows_spec, rows_spec),
            out_specs=P(),
        ).loss
        kl_ref = fused_kl_divergence(student, teacher, weights, platform="xla", reduction="mean").loss
        kl_mesh = fused_kl_divergence(
            student,
            teacher,
            weights,
            platform="xla",
            reduction="mean",
            mesh=mesh,
            in_specs=(logits_spec, logits_spec, rows_spec),
            out_specs=P(),
        ).loss

    np.testing.assert_allclose(np.asarray(ce_mesh), np.asarray(ce_ref), rtol=0.0, atol=3e-3)
    np.testing.assert_allclose(np.asarray(kl_mesh), np.asarray(kl_ref), rtol=0.0, atol=1e-3)


def test_pallas_sp_mesh_matches_reference_with_replicated_vocab():
    """Check Pallas with sequence sharding and replicated vocab matches XLA."""
    logits, targets, weights, student, teacher = _inputs()
    logits_spec = P(("dp", "fsdp"), "sp", None)
    rows_spec = P(("dp", "fsdp"), "sp")

    with _mesh() as mesh:
        ce_ref = fused_cross_entropy(logits, targets, weights, platform="xla", reduction="mean").loss
        ce_mesh = fused_cross_entropy(
            logits,
            targets,
            weights,
            platform="pallas",
            reduction="mean",
            mesh=mesh,
            in_specs=(logits_spec, rows_spec, rows_spec),
            out_specs=P(),
        ).loss
        kl_ref = fused_kl_divergence(student, teacher, weights, platform="xla", reduction="mean").loss
        kl_mesh = fused_kl_divergence(
            student,
            teacher,
            weights,
            platform="pallas",
            reduction="mean",
            mesh=mesh,
            in_specs=(logits_spec, logits_spec, rows_spec),
            out_specs=P(),
        ).loss

    np.testing.assert_allclose(np.asarray(ce_mesh), np.asarray(ce_ref), rtol=0.0, atol=6e-3)
    np.testing.assert_allclose(np.asarray(kl_mesh), np.asarray(kl_ref), rtol=0.0, atol=7e-3)


def test_pallas_vocab_parallel_mesh_matches_reference_and_grad():
    """Check Pallas TP-vocab forward values and gradients against XLA."""
    logits, targets, weights, student, teacher = _inputs()
    logits_spec = P(("dp", "fsdp"), "sp", "tp")
    rows_spec = P(("dp", "fsdp"), "sp")

    with _mesh() as mesh:
        ce_ref = fused_cross_entropy(logits, targets, weights, platform="xla", reduction="mean").loss
        ce_mesh = fused_cross_entropy(
            logits,
            targets,
            weights,
            platform="pallas",
            reduction="mean",
            mesh=mesh,
            in_specs=(logits_spec, rows_spec, rows_spec),
            out_specs=P(),
        ).loss
        kl_ref = fused_kl_divergence(student, teacher, weights, platform="xla", reduction="mean").loss
        kl_mesh = fused_kl_divergence(
            student,
            teacher,
            weights,
            platform="pallas",
            reduction="mean",
            mesh=mesh,
            in_specs=(logits_spec, logits_spec, rows_spec),
            out_specs=P(),
        ).loss

        ce_grad_ref = jax.grad(
            lambda x: fused_cross_entropy(x, targets, weights, platform="xla", reduction="mean").loss
        )(logits)
        ce_grad_mesh = jax.grad(
            lambda x: (
                fused_cross_entropy(
                    x,
                    targets,
                    weights,
                    platform="pallas",
                    reduction="mean",
                    mesh=mesh,
                    in_specs=(logits_spec, rows_spec, rows_spec),
                    out_specs=P(),
                ).loss
            )
        )(logits)

        kl_grad_ref = jax.grad(
            lambda x: fused_kl_divergence(x, teacher, weights, platform="xla", reduction="mean").loss
        )(student)
        kl_grad_mesh = jax.grad(
            lambda x: (
                fused_kl_divergence(
                    x,
                    teacher,
                    weights,
                    platform="pallas",
                    reduction="mean",
                    mesh=mesh,
                    in_specs=(logits_spec, logits_spec, rows_spec),
                    out_specs=P(),
                ).loss
            )
        )(student)

    np.testing.assert_allclose(np.asarray(ce_mesh), np.asarray(ce_ref), rtol=0.0, atol=6e-3)
    np.testing.assert_allclose(np.asarray(kl_mesh), np.asarray(kl_ref), rtol=0.0, atol=7e-3)
    np.testing.assert_allclose(np.asarray(ce_grad_mesh), np.asarray(ce_grad_ref), rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(np.asarray(kl_grad_mesh), np.asarray(kl_grad_ref), rtol=0.0, atol=2e-5)
