import jax.numpy as jnp
import pytest

from easydel.trainers import _logprob_utils


@pytest.mark.parametrize("chunk_size", (None, 4, 16))
def test_compute_token_logps_uses_dense_fast_path_when_vocab_chunking_disabled(monkeypatch, chunk_size):
    logits = jnp.asarray(
        [
            [
                [1.0, 0.0, -1.0, 0.5],
                [0.1, 0.4, -0.2, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    targets = jnp.asarray([[0, 3]], dtype=jnp.int32)

    calls = {"log_softmax": 0}
    original_log_softmax = _logprob_utils.jax.nn.log_softmax

    def wrapped_log_softmax(*args, **kwargs):
        calls["log_softmax"] += 1
        return original_log_softmax(*args, **kwargs)

    def fail_fori_loop(*args, **kwargs):
        raise AssertionError("dense fast path should bypass the chunked loop")

    monkeypatch.setattr(_logprob_utils.jax.nn, "log_softmax", wrapped_log_softmax)
    monkeypatch.setattr(_logprob_utils.lax, "fori_loop", fail_fori_loop)

    token_logps, entropies = _logprob_utils.compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=True,
        chunk_size=chunk_size,
    )

    expected_log_probs = original_log_softmax(logits.astype(jnp.float32), axis=-1)
    expected_token_logps = jnp.take_along_axis(expected_log_probs, targets[..., None], axis=-1).squeeze(-1)
    expected_entropies = -jnp.sum(jnp.exp(expected_log_probs) * expected_log_probs, axis=-1)

    assert calls["log_softmax"] == 1
    assert jnp.allclose(token_logps, expected_token_logps)
    assert jnp.allclose(entropies, expected_entropies)
