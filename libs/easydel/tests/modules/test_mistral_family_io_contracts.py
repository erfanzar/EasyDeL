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

"""Public input/output and checkpoint contracts without array computation."""

from types import MethodType, SimpleNamespace

import pytest
from easydel.modules.minimax.modeling_minimax import MiniMaxDecoderLayer
from easydel.modules.mistral import MistralForSequenceClassification
from easydel.modules.mistral4 import Mistral4Config
from easydel.modules.mixtral import MixtralForSequenceClassification
from easydel.modules.mixtral.modeling_mixtral import MixtralDecoderLayer

pytestmark = pytest.mark.cpu_ok


class _TokenLogits:
    """Only support indexing token rows; no numerical array operations."""

    shape = (1, 3, 2)

    def __init__(self, final):
        self.final = final

    def __getitem__(self, key):
        assert key == (slice(None), -1)
        return self.final


@pytest.mark.parametrize("model_class", [MistralForSequenceClassification, MixtralForSequenceClassification])
@pytest.mark.parametrize("pad_token_id", [None, 0])
def test_embeddings_only_classifier_returns_final_token_and_backbone_cache(model_class, pad_token_id):
    final, hidden, pooled, cache, old_cache, auxiliary = (object() for _ in range(6))
    if model_class is MistralForSequenceClassification:
        # The shared classifier pools representations before applying its
        # score head, rather than scoring all tokens and pooling logits.
        hidden = _TokenLogits(pooled)
    backbone_output = SimpleNamespace(
        last_hidden_state=hidden,
        past_key_values=cache,
        hidden_states=(hidden,),
        attentions=None,
    )
    embeds = SimpleNamespace(shape=(1, 3, 4))
    observed = {}

    def backbone(**kwargs):
        observed.update(kwargs)
        return backbone_output

    model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=pad_token_id),
        _score_head_name="score",
        base_model=backbone,
        score=lambda x: (
            final if x is pooled else _TokenLogits(final) if x is hidden else pytest.fail("wrong backbone states")
        ),
        compute_router_aux_loss=lambda *args, **kwargs: auxiliary,
        compute_aux_loss=lambda *args, **kwargs: auxiliary,
    )
    # Mistral uses the base forward plus its own embedding-only pooling hook.
    # Mixtral keeps its explicit output_router_logits signature and forward.
    if model_class is MistralForSequenceClassification:
        model.pool_sequence = MethodType(model_class.pool_sequence, model)
    else:
        model.pool_sequence = lambda *args, **kwargs: pytest.fail("embedding-only pooling required token IDs")
    kwargs = {"output_router_logits": True} if model_class is MixtralForSequenceClassification else {}
    output = model_class.forward(model, inputs_embeds=embeds, past_key_values=old_cache, **kwargs)
    assert output.logits is final
    assert output.past_key_values is cache
    assert output.hidden_states == (hidden,)
    assert output.aux_loss is auxiliary
    assert observed["inputs_embeds"] is embeds
    assert observed["input_ids"] is None
    assert observed["attention_mask"] is None
    if kwargs:
        assert observed["output_router_logits"] is True


class _AxisTensor:
    """Track checkpoint axis labels, not numerical values."""

    def __init__(self, axes, shape):
        self.axes = axes
        self.shape = shape

    def swapaxes(self, first, second):
        axes, shape = list(self.axes), list(self.shape)
        axes[first], axes[second] = axes[second], axes[first]
        shape[first], shape[second] = shape[second], shape[first]
        return _AxisTensor(tuple(axes), tuple(shape))

    transpose = swapaxes

    def contiguous(self):
        return self


class _ShapeOnlyTorch:
    """Split and concatenate axis metadata without allocating arrays."""

    @staticmethod
    def split(tensor, sizes, dim):
        assert sum(sizes) == tensor.shape[dim]
        parts = []
        for size in sizes:
            shape = list(tensor.shape)
            shape[dim] = size
            parts.append(_AxisTensor(tensor.axes, tuple(shape)))
        return tuple(parts)

    @staticmethod
    def cat(tensors, dim):
        shape = list(tensors[0].shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        assert all(tensor.axes == tensors[0].axes for tensor in tensors)
        return _AxisTensor(tensors[0].axes, tuple(shape))


@pytest.mark.parametrize("decoder_class", [MiniMaxDecoderLayer, MixtralDecoderLayer])
def test_decoder_checkpoint_rules_preserve_router_down_and_fused_axes(decoder_class):
    # ExpertTensorMode intentionally stores contiguous gate/up channels (TP=1
    # for contraction packing), avoiding any mesh creation or backend work.
    layer = SimpleNamespace(config=SimpleNamespace(use_expert_tensor_mode=True))
    rules = decoder_class.reform_param.fget(layer)
    assert set(rules) == {
        "mlp.gate.weight$",
        "mlp.experts.down_proj$",
        "block_sparse_moe.experts.gate_up_proj.weight$",
    }
    router = rules["mlp.gate.weight$"]
    assert [item["name"] for item in router["splits"]] == ["block_sparse_moe.gate.weight"]
    hf_router = _AxisTensor(("experts", "hidden"), (2, 4))
    loaded_router = router["splits"][0]["spliter"](hf_router)
    assert loaded_router.axes == ("hidden", "experts")
    assert router["inverse_spliter"](loaded_router).axes == hf_router.axes

    down = rules["mlp.experts.down_proj$"]
    assert [item["name"] for item in down["splits"]] == ["block_sparse_moe.experts.w2.weight"]
    hf_down = _AxisTensor(("experts", "hidden", "intermediate"), (2, 4, 6))
    loaded_down = down["splits"][0]["spliter"](hf_down)
    assert loaded_down.axes == ("experts", "intermediate", "hidden")
    assert down["inverse_spliter"](loaded_down).axes == hf_down.axes

    fused = rules["block_sparse_moe.experts.gate_up_proj.weight$"]
    assert fused["sources"] == ("mlp.experts.gate_up_proj",)
    assert fused["already_converted"] is True
    hf_fused = _AxisTensor(("experts", "gate_up", "hidden"), (2, 12, 4))
    loaded_fused = fused["fuser"](_ShapeOnlyTorch, hf_fused)
    assert loaded_fused.axes == ("experts", "hidden", "gate_up")
    (exported_fused,) = fused["inverse_fuser"](_ShapeOnlyTorch, loaded_fused)
    assert exported_fused.axes == hf_fused.axes
    assert exported_fused.shape == hf_fused.shape


@pytest.mark.parametrize("k", [3, 4])
def test_mistral4_retains_configs_with_fewer_group_members_than_routing_slots(k):
    config = Mistral4Config(n_routed_experts=8, n_group=4, topk_group=1, num_experts_per_tok=k)
    assert config.num_experts_per_tok == k
    assert config.n_routed_experts // config.n_group * config.topk_group == 2
