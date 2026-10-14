"""Full synthetic checkpoint accounting, independent of the conversion test namespace.

Run in an environment containing the reviewed EasyDeL/Spectrax dependencies:
    JAX_PLATFORMS=cpu ENABLE_DISTRIBUTED_INIT=0 python -m pytest -q \
        .xerxes-scratch/test_qwen4_checkpoint_completeness.py

No claim is made that transform_fn consumes/mutates its input mapping: its
observed public contract returns a tree, not a consumed-key report. Instead an
independent value oracle accounts for every source exactly once (the flat PLE
table is read once and then sliced), and checks *every* runtime trainable leaf.
This catches ignored source tensors, missing destinations, bad fusions, expert
order, transposes, and silent retention of lazy-init values. TP=1, fp32, no MTP.
"""

import os
from collections import Counter

import pytest

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# These are host-derived metadata, NOT permitted missing trainable parameters.
# Exact names deliberately avoid a wildcard that could hide a new trainable.
DERIVED_ATTRIBUTES = (
    "head_vocab_sizes",
    "head_offsets",
    "layer_multipliers",
)
HF_ABSENT_DERIVED_ATTRIBUTES = ("ngram_heads_vocab_sizes", "ngram_heads_offsets")
PLE = "model.layers.0.ple.ple_embedding"
VOCAB = 512
ROWS = 16256
EXPERTS = 2


def _assert_source_accounting(source_keys, reads):
    counts = Counter(reads)
    assert set(counts) == set(source_keys), (
        f"Unaccounted sources: {sorted(set(source_keys) - set(counts))}; "
        f"unknown oracle sources: {sorted(set(counts) - set(source_keys))}"
    )
    assert all(n == 1 for n in counts.values()), f"Repeated source reads: {counts}"


def _assert_destinations(expected, actual):
    assert set(expected) == set(actual), (
        f"Missing trainables: {sorted(set(expected) - set(actual))}; "
        f"unexpected trainables: {sorted(set(actual) - set(expected))}"
    )


@pytest.mark.parametrize("keys,reads", [({"a", "b"}, ["a"]), ({"a"}, ["a", "a"]), ({"a"}, ["a", "b"])])
def test_source_accounting_rejects_incomplete_duplicate_or_unknown_sources(keys, reads):
    with pytest.raises(AssertionError):
        _assert_source_accounting(keys, reads)


@pytest.mark.parametrize("actual", [{}, {"a": 1, "extra": 2}])
def test_destination_accounting_rejects_missing_or_extra_leaves(actual):
    with pytest.raises(AssertionError):
        _assert_destinations({"a": 1}, actual)


def _config(config_type, jnp):
    # Standalone adaptation of final-review18's fixture; only two alternating
    # layers/two experts are needed to exercise every non-MTP trainable family.
    return config_type(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        linear_attention_separate_proj=True,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=2000,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        eos_token_id=0,
        bos_token_id=1,
        pad_token_id=2,
        norm_topk_prob=True,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        attn_dtype=jnp.float32,
        mtp=None,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


def _synthetic_sd(torch, layout):
    gen = torch.Generator().manual_seed(839)

    def t(*shape):
        # Distinct, non-symmetric values expose swaps and square transposes.
        return torch.randn(*shape, generator=gen) * 0.05

    sd = {
        "model.embed_tokens.weight": t(VOCAB, 64),
        "lm_head.weight": t(VOCAB, 64),
        "model.hyper_connection_mixer.hc_norm.weight": t(256),
        "model.hyper_connection_mixer.input_mix_weight_down.weight": t(8, 256),
        "model.hyper_connection_mixer.input_mix_weight_up.weight": t(256, 8),
    }
    for li in range(2):
        p = f"model.layers.{li}"
        for connection in ("attn_hyper_connection", "mlp_hyper_connection"):
            sd[f"{p}.{connection}.hc_norm.weight"] = t(256)
            sd[f"{p}.{connection}.input_mix_weight_down.weight"] = t(8, 256)
            sd[f"{p}.{connection}.input_mix_weight_up.weight"] = t(256, 8)
            sd[f"{p}.{connection}.block_inject_weight.weight"] = t(4, 256)
        sd[f"{p}.mlp.gate.weight"] = t(EXPERTS, 64)
        for e in range(EXPERTS):
            sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = t(16, 64)
            sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = t(16, 64)
            sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = t(64, 16)
        for proj, shape in (("gate_proj", (16, 64)), ("up_proj", (16, 64)), ("down_proj", (64, 16))):
            sd[f"{p}.mlp.shared_expert.{proj}.weight"] = t(*shape)
        sd[f"{p}.mlp.shared_expert_gate.weight"] = t(1, 64)
        if li == 0:
            for name, shape in {
                "in_proj_qkv.weight": (64, 64),
                "in_proj_z.weight": (32, 64),
                "in_proj_b.weight": (4, 64),
                "in_proj_a.weight": (4, 64),
                "conv1d.weight": (64, 1, 4),
                "A_log": (4,),
                "dt_bias": (4,),
                "norm.weight": (8,),
                "out_proj.weight": (64, 32),
            }.items():
                sd[f"{p}.linear_attn.{name}"] = t(*shape)
        else:
            for name, shape in {
                "q_proj.weight": (128, 64),
                "k_proj.weight": (32, 64),
                "v_proj.weight": (32, 64),
                "o_proj.weight": (64, 64),
                "q_norm.weight": (16,),
                "k_norm.weight": (16,),
                "indexer.index_qk_proj.weight": (24, 64),
                "indexer.q_layernorm.weight": (8,),
                "indexer.k_layernorm.weight": (8,),
            }.items():
                sd[f"{p}.self_attn.{name}"] = t(*shape)
    for name, shape in {
        "key_proj.weight": (256, 64),
        "value_proj.weight": (64, 64),
        "norm_key.weight": (256,),
        "norm_query.weight": (256,),
        "norm_conv.weight": (256,),
        "conv1d.weight": (256, 1, 4),
    }.items():
        sd[f"model.layers.0.ple.{name}"] = t(*shape)
    table = t(ROWS, 8)
    if layout == "flat":
        sd[f"{PLE}.ngram_embedding.weight"] = table
    else:
        for i in range(4):
            sd[f"{PLE}.ngram_embedding.shard_{i}.weight"] = table[i * (ROWS // 4) : (i + 1) * (ROWS // 4)].clone()
    return sd


def _expected_values(sd, np, layout):
    """Independent NumPy oracle; never invokes production reform callbacks.

    Keys denote single-leaf parameter owners, so weight/kernel/scale boxing
    conventions are resolved against the actual trainable registry below.
    Multi-leaf owners (conv) explicitly select weight versus bias.
    """
    reads = []
    expected = {}

    def take(key):
        reads.append(key)
        return sd[key].detach().cpu().numpy().copy()

    def direct(owner, suffix="weight", transpose=False):
        value = take(f"{owner}.{suffix}")
        expected[owner] = value.T if transpose else value

    direct("model.embed_tokens")
    direct("lm_head", transpose=True)
    direct("model.hyper_connection_mixer.hc_norm")
    direct("model.hyper_connection_mixer.input_mix_weight_down", transpose=True)
    direct("model.hyper_connection_mixer.input_mix_weight_up", transpose=True)
    for li in range(2):
        p = f"model.layers.{li}"
        for connection in ("attn_hyper_connection", "mlp_hyper_connection"):
            direct(f"{p}.{connection}.hc_norm")
            for proj in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
                direct(f"{p}.{connection}.{proj}", transpose=True)
        direct(f"{p}.mlp.gate", transpose=True)
        gate_up, down = [], []
        for e in range(EXPERTS):
            ep = f"{p}.mlp.experts.{e}"
            gate_up.append(np.concatenate((take(f"{ep}.gate_proj.weight").T, take(f"{ep}.up_proj.weight").T), axis=-1))
            down.append(take(f"{ep}.down_proj.weight").T)
        expected[f"{p}.mlp.experts.gate_up_proj"] = np.stack(gate_up)
        expected[f"{p}.mlp.experts.down_proj"] = np.stack(down)
        # Qwen4ExpMLP uses separate (not fused) shared-expert projections.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            direct(f"{p}.mlp.shared_expert.{proj}", transpose=True)
        direct(f"{p}.mlp.shared_expert_gate", transpose=True)
        if li == 0:
            a = f"{p}.linear_attn"
            for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                direct(f"{a}.{proj}", transpose=True)
            expected[f"{a}.conv1d.weight"] = take(f"{a}.conv1d.weight").transpose(2, 1, 0)
            for name in ("A_log", "dt_bias"):
                expected[f"{a}.{name}"] = take(f"{a}.{name}")
            direct(f"{a}.norm")
        else:
            a = f"{p}.self_attn"
            expected[f"{a}.qkv_proj"] = np.concatenate(
                [take(f"{a}.{proj}_proj.weight") for proj in ("q", "k", "v")], axis=0
            ).T
            direct(f"{a}.o_proj", transpose=True)
            for norm in ("q_norm", "k_norm"):
                direct(f"{a}.{norm}")
            direct(f"{a}.indexer.index_qk_proj", transpose=True)
            for norm in ("q_layernorm", "k_layernorm"):
                direct(f"{a}.indexer.{norm}")
    p = "model.layers.0.ple"
    for proj in ("key_proj", "value_proj"):
        direct(f"{p}.{proj}", transpose=True)
    for norm in ("norm_key", "norm_query", "norm_conv"):
        direct(f"{p}.{norm}")
    expected[f"{p}.conv1d"] = take(f"{p}.conv1d.weight").transpose(2, 1, 0)
    if layout == "flat":
        shards = np.split(take(f"{PLE}.ngram_embedding.weight"), 4, axis=0)
    else:
        shards = [take(f"{PLE}.ngram_embedding.shard_{i}.weight") for i in range(4)]
    for i, shard in enumerate(shards):
        expected[f"{PLE}.shards.{i}"] = shard
    _assert_source_accounting(sd, reads)
    return expected


def _resolve_owners(expected, trainables):
    resolved = {}
    for owner, value in expected.items():
        matches = [key for key in trainables if key == owner or key.startswith(owner + ".")]
        assert len(matches) == 1, f"Expected exactly one trainable for {owner}, found {matches}"
        key = matches[0]
        assert key not in resolved, f"Duplicate destination {key}"
        resolved[key] = value
    # No exemption for derived state here: it must not be trainable at all.
    _assert_destinations(resolved, trainables)
    return resolved


@pytest.mark.parametrize("layout", ["flat", "release_shards"])
def test_every_checkpoint_source_and_trainable_destination(layout):
    # Hard imports intentionally make unavailable runtime dependencies an ERROR,
    # not a misleading successful/skipped full-checkpoint verification.
    import easydel as ed
    import numpy as np
    import torch
    from easydel.modules.qwen4_exp import Qwen4ExpTextConfig
    from jax import numpy as jnp

    model = ed.AutoEasyDeLModelForCausalLM.from_config(
        _config(Qwen4ExpTextConfig, jnp), dtype=jnp.float32, param_dtype=jnp.float32
    )
    sd = _synthetic_sd(torch, layout)
    expected = _resolve_owners(_expected_values(sd, np, layout), model.parameter_values(remove_none=False))
    emb = model.model.layers[0].ple.ple_embedding
    derived = {name: np.asarray(getattr(emb, name)).copy() for name in DERIVED_ATTRIBUTES}
    for name in HF_ABSENT_DERIVED_ATTRIBUTES:
        assert not hasattr(emb, name)
    # Clone tensors: converter mutation cannot rewrite our source/value oracle.
    tree = model.transform_fn({key: value.clone() for key, value in sd.items()})
    merged = ed.traversals.merge_model_and_tree(model, tree=tree)
    actual = merged.parameter_values(remove_none=False)
    _assert_destinations(expected, actual)
    assert not merged.abstract_parameter_leaves(), "Conversion retained lazy trainables"
    for key, reference in expected.items():
        loaded = np.asarray(actual[key])
        assert loaded.shape == reference.shape, key
        assert loaded.dtype == np.dtype("float32"), key
        np.testing.assert_array_equal(loaded, reference, err_msg=key)
    merged_emb = merged.model.layers[0].ple.ple_embedding
    for name, reference in derived.items():
        np.testing.assert_array_equal(np.asarray(getattr(merged_emb, name)), reference, err_msg=name)
    for name in HF_ABSENT_DERIVED_ATTRIBUTES:
        assert not hasattr(merged_emb, name)
