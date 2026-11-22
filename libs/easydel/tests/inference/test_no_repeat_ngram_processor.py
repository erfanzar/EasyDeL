# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``NoRepeatNGramLogitsProcessor`` — banning already-seen n-gram completions.

The processor keeps generation from repeating any n-gram already present in
``input_ids``: given the last ``n - 1`` tokens, every token that would complete
a previously seen n-gram has its logit driven to ``-inf``.

The implementation builds a sparse ``BCOO`` index of shape
``(batch,) + (vocab,) * ngram_size`` and looks the prefix up in it, which is
compact but hard to read. These tests therefore check it against a direct
dictionary-based reference written independently in NumPy, rather than against
a restatement of the same sparse construction — comparing the production logic
with itself would prove nothing.

Sibling coverage for the other warpers/processors lives in
``test_logits_process_and_sampling.py``; this file exists because the n-gram
processor was the one member of that module with no tests at all.
"""

import jax
import numpy as np
import pytest
from easydel.inference.logits_process import NoRepeatNGramLogitsProcessor
from jax import numpy as jnp

VOCAB = 7


def _reference_banned(input_ids: np.ndarray, cur_len: int, ngram_size: int) -> np.ndarray:
    """Independent reference: which tokens are banned for each batch row.

    Collects every n-gram fully inside ``input_ids[:, :cur_len]``, then bans the
    final token of any whose first ``n - 1`` tokens match the current suffix.

    Args:
        input_ids: Token ids ``[batch, seq_len]``.
        cur_len: Number of valid tokens in each row.
        ngram_size: N-gram width.

    Returns:
        Boolean mask ``[batch, VOCAB]``; ``True`` where generation is banned.
    """
    batch = input_ids.shape[0]
    banned = np.zeros((batch, VOCAB), dtype=bool)
    if ngram_size <= 0 or cur_len < ngram_size:
        return banned
    for b in range(batch):
        seq = list(input_ids[b, :cur_len])
        seen: dict[tuple, set] = {}
        for i in range(len(seq) - ngram_size + 1):
            prefix = tuple(seq[i : i + ngram_size - 1])
            seen.setdefault(prefix, set()).add(seq[i + ngram_size - 1])
        suffix = tuple(seq[cur_len - (ngram_size - 1) : cur_len])
        for token in seen.get(suffix, ()):
            banned[b, token] = True
    return banned


def _apply(input_ids, scores, cur_len, ngram_size):
    proc = NoRepeatNGramLogitsProcessor(ngram_size=ngram_size)
    return np.asarray(proc(jnp.asarray(input_ids), jnp.asarray(scores), cur_len))


def _flat_scores(batch: int) -> np.ndarray:
    return np.zeros((batch, VOCAB), dtype=np.float32)


def test_ngram_size_zero_is_identity():
    """``ngram_size=0`` disables the processor entirely."""
    ids = np.array([[1, 2, 1, 2]], dtype=np.int32)
    scores = np.arange(VOCAB, dtype=np.float32)[None, :]
    out = _apply(ids, scores, cur_len=4, ngram_size=0)
    assert np.array_equal(out, scores)


def test_sequence_shorter_than_ngram_bans_nothing():
    """With fewer tokens than the n-gram width there is nothing to repeat."""
    ids = np.array([[3, 4, 0, 0]], dtype=np.int32)
    scores = _flat_scores(1)
    out = _apply(ids, scores, cur_len=2, ngram_size=3)
    assert np.all(np.isfinite(out))


def test_repeated_bigram_completion_is_banned():
    """``1 2 1`` with n=2 must ban ``2``: the bigram ``(1, 2)`` already occurred."""
    ids = np.array([[1, 2, 1, 0]], dtype=np.int32)
    out = _apply(ids, _flat_scores(1), cur_len=3, ngram_size=2)

    assert out[0, 2] == -np.inf
    others = [t for t in range(VOCAB) if t != 2]
    assert np.all(np.isfinite(out[0, others]))


def test_unrepeated_prefix_bans_nothing():
    """A suffix never seen before must leave every logit untouched."""
    ids = np.array([[1, 2, 3, 0]], dtype=np.int32)
    out = _apply(ids, _flat_scores(1), cur_len=3, ngram_size=2)
    assert np.all(np.isfinite(out))


def test_trigram_bans_only_the_completing_token():
    """``1 2 3 1 2`` with n=3 bans ``3`` — and nothing else."""
    ids = np.array([[1, 2, 3, 1, 2, 0]], dtype=np.int32)
    out = _apply(ids, _flat_scores(1), cur_len=5, ngram_size=3)

    assert out[0, 3] == -np.inf
    others = [t for t in range(VOCAB) if t != 3]
    assert np.all(np.isfinite(out[0, others]))


def test_multiple_continuations_of_one_prefix_are_all_banned():
    """``1 2 1 3 1`` with n=2: the prefix ``1`` was followed by both 2 and 3."""
    ids = np.array([[1, 2, 1, 3, 1, 0]], dtype=np.int32)
    out = _apply(ids, _flat_scores(1), cur_len=5, ngram_size=2)

    assert out[0, 2] == -np.inf
    assert out[0, 3] == -np.inf
    others = [t for t in range(VOCAB) if t not in (2, 3)]
    assert np.all(np.isfinite(out[0, others]))


def test_rows_are_independent():
    """One row's history must not ban tokens in another row."""
    ids = np.array([[1, 2, 1, 0], [4, 5, 4, 0]], dtype=np.int32)
    out = _apply(ids, _flat_scores(2), cur_len=3, ngram_size=2)

    assert out[0, 2] == -np.inf and np.isfinite(out[0, 5])
    assert out[1, 5] == -np.inf and np.isfinite(out[1, 2])


@pytest.mark.parametrize("ngram_size", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_independent_reference(ngram_size, seed):
    """Randomised agreement with the dictionary-based reference."""
    rng = np.random.default_rng(seed)
    batch, seq_len = 3, 8
    ids = rng.integers(0, VOCAB, (batch, seq_len)).astype(np.int32)
    cur_len = seq_len

    out = _apply(ids, _flat_scores(batch), cur_len, ngram_size)
    got_banned = ~np.isfinite(out)
    want_banned = _reference_banned(ids, cur_len, ngram_size)

    assert np.array_equal(got_banned, want_banned), (
        f"seed={seed} n={ngram_size}\nids={ids}\ngot={got_banned}\nwant={want_banned}"
    )


def test_unbanned_logits_keep_their_values():
    """Surviving logits must pass through untouched, not be renormalised."""
    ids = np.array([[1, 2, 1, 0]], dtype=np.int32)
    scores = np.arange(VOCAB, dtype=np.float32)[None, :] * 0.5 - 1.0
    out = _apply(ids, scores, cur_len=3, ngram_size=2)

    keep = [t for t in range(VOCAB) if t != 2]
    assert np.array_equal(out[0, keep], scores[0, keep])


def test_shape_and_dtype_are_preserved():
    ids = np.array([[1, 2, 1, 0]], dtype=np.int32)
    scores = _flat_scores(1)
    out = _apply(ids, scores, cur_len=3, ngram_size=2)
    assert out.shape == scores.shape
    assert out.dtype == scores.dtype


def test_works_under_jit():
    """The processor runs inside generation's compiled loop, so it must trace."""
    ids = jnp.asarray([[1, 2, 1, 0]], dtype=jnp.int32)
    scores = jnp.zeros((1, VOCAB), jnp.float32)
    proc = NoRepeatNGramLogitsProcessor(ngram_size=2)

    jitted = jax.jit(lambda i, s: proc(i, s, 3))
    out = np.asarray(jitted(ids, scores))

    assert out[0, 2] == -np.inf
    assert np.all(np.isfinite(out[0, [t for t in range(VOCAB) if t != 2]]))


# --------------------------------------------------------------------------
# Integration: the processor must actually reach a model's generation chain.
# --------------------------------------------------------------------------


MODELS_WITH_GENERATION = (
    "DeepseekV4ForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "LlamaForCausalLM",
)


@pytest.mark.parametrize("model_name", MODELS_WITH_GENERATION)
def test_model_families_inherit_the_generation_pipeline(model_name):
    """Every family reaches ``no_repeat_ngram_size`` through the shared mixin.

    The processor is model-agnostic — it lives on ``EasyGenerationMixin`` and is
    appended by ``_get_logits_processor``. This pins that the families that
    actually get served still inherit that path.
    """
    import easydel as ed
    from easydel.infra.mixins.generation import EasyGenerationMixin

    cls = getattr(ed, model_name, None)
    assert cls is not None, f"{model_name} is not exported"
    assert issubclass(cls, EasyGenerationMixin)


def test_generation_config_builds_the_processor_into_the_chain():
    """``no_repeat_ngram_size > 0`` must put the processor in the chain, and it
    must then actually ban the repeated completion end to end."""
    import easydel as ed
    import spectrax as spx
    from easydel.inference.logits_process import LogitsProcessorList
    from transformers import GenerationConfig

    config = ed.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        model = ed.LlamaForCausalLM.sequential_init(
            config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
        )
        model.eval()

    gen_config = GenerationConfig(no_repeat_ngram_size=3, max_new_tokens=4, do_sample=False)
    processors = model._get_logits_processor(gen_config, 4, LogitsProcessorList())
    assert any(isinstance(p, NoRepeatNGramLogitsProcessor) for p in processors)

    # "1 2 3 1 2" -> completing with 3 would repeat the trigram (1, 2, 3).
    ids = jnp.asarray([[1, 2, 3, 1, 2]], jnp.int32)
    scores = jnp.zeros((1, config.vocab_size), jnp.float32)
    for processor in processors:
        scores = processor(ids, scores, 5)

    banned = [i for i in range(config.vocab_size) if not bool(jnp.isfinite(scores[0, i]))]
    assert banned == [3]


def test_disabled_generation_config_leaves_the_chain_clean():
    """``no_repeat_ngram_size=0`` must not append the processor at all."""
    import easydel as ed
    import spectrax as spx
    from easydel.inference.logits_process import LogitsProcessorList
    from transformers import GenerationConfig

    config = ed.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        model = ed.LlamaForCausalLM.sequential_init(
            config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
        )
        model.eval()

    gen_config = GenerationConfig(no_repeat_ngram_size=0, max_new_tokens=4, do_sample=False)
    processors = model._get_logits_processor(gen_config, 4, LogitsProcessorList())
    assert not any(isinstance(p, NoRepeatNGramLogitsProcessor) for p in processors)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_unigram_bans_every_seen_token():
    """Unigram blocking forbids every token already present in the live prefix."""
    ids = np.array([[1, 3, 1, 4, 6]], dtype=np.int32)
    out = _apply(ids, _flat_scores(1), cur_len=4, ngram_size=1)

    assert np.array_equal(np.flatnonzero(~np.isfinite(out[0])), np.array([1, 3, 4]))


def test_unigram_works_under_jit():
    ids = jnp.asarray([[1, 3, 1, 4, 6]], dtype=jnp.int32)
    scores = jnp.zeros((1, VOCAB), jnp.float32)
    proc = NoRepeatNGramLogitsProcessor(ngram_size=1)

    out = np.asarray(jax.jit(lambda i, s: proc(i, s, 4))(ids, scores))
    assert np.array_equal(np.flatnonzero(~np.isfinite(out[0])), np.array([1, 3, 4]))
