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

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from easydel.data.transforms.pack import PackedShardedSource
from easydel.infra.loss_utils import LossConfig
from easydel.trainers.prompt_transforms import SFTPreprocessTransform
from easydel.trainers.supervised_fine_tuning_trainer.sft_config import SFTConfig
from easydel.trainers.supervised_fine_tuning_trainer.sft_trainer import SFTTrainer
from easydel.trainers.trainer._fn import _dft_causal_lm_metrics


class _TinySource:
    def __init__(self):
        self.shard_names = ["s0"]

    def open_shard(self, shard_name):
        del shard_name
        yield {
            "input_ids": np.asarray([10, 11], dtype=np.int32),
            "completion_mask": np.asarray([0, 1], dtype=np.int32),
        }
        yield {
            "input_ids": np.asarray([20], dtype=np.int32),
            "completion_mask": np.asarray([1], dtype=np.int32),
        }

    def __len__(self):
        return 2


def test_sft_completion_only_loss_overrides_assistant_only_loss():
    cfg = SFTConfig(completion_only_loss=True, assistant_only_loss=False)

    assert cfg.completion_only_loss is True
    assert cfg.assistant_only_loss is True


def test_sft_padding_free_opts_into_packing():
    cfg = SFTConfig(padding_free=True, packing=False)

    assert cfg.padding_free is True
    assert cfg.packing is True


def test_sft_pad_to_multiple_of_validates_positive_value():
    cfg = SFTConfig(pad_to_multiple_of=8)

    assert cfg.pad_to_multiple_of == 8
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        SFTConfig(pad_to_multiple_of=0)


def test_sft_truncation_mode_defaults_to_keep_start_and_validates_values():
    assert SFTConfig().truncation_mode == "keep_start"
    assert SFTConfig(truncation_mode="keep_end").truncation_mode == "keep_end"

    with pytest.raises(ValueError, match="truncation_mode"):
        SFTConfig(truncation_mode="middle")


def test_sft_activation_offloading_guard():
    assert SFTConfig(activation_offloading=True).activation_offloading is True


def test_sft_stale_compatibility_knobs_fail_loudly_when_non_default():
    with pytest.raises(ValueError, match="add_special_tokens"):
        SFTConfig(add_special_tokens=True)
    with pytest.raises(ValueError, match="dataset_batch_size"):
        SFTConfig(dataset_batch_size=32)
    with pytest.raises(ValueError, match="num_of_sequences"):
        SFTConfig(num_of_sequences=32)


def test_sft_dataset_kwargs_supports_skip_prepare_dataset_only():
    cfg = SFTConfig(dataset_kwargs={"skip_prepare_dataset": True})

    assert cfg.dataset_kwargs == {"skip_prepare_dataset": True}
    with pytest.raises(ValueError, match="skip_prepare_dataset"):
        SFTConfig(dataset_kwargs={"unknown": True})


def test_sft_skip_prepare_dataset_skips_preprocess_transform():
    trainer = object.__new__(SFTTrainer)
    trainer.arguments = SFTConfig(dataset_kwargs={"skip_prepare_dataset": True})

    assert trainer._get_preprocess_transform() is None


def test_sft_pad_token_and_shuffle_dataset_alias_have_behavior():
    cfg = SFTConfig(pad_token="<pad>", shuffle_dataset=False)

    assert cfg.pad_token == "<pad>"
    assert cfg.shuffle_dataset is False
    assert cfg.shuffle_train_dataset is False

    class Tokenizer:
        eos_token = None
        pad_token = None

    tokenizer = Tokenizer()
    SFTTrainer._apply_tokenizer_overrides(tokenizer, eos_token="<eos>", pad_token="<pad>")

    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.pad_token == "<pad>"


def test_sft_chat_template_path_loads_local_template(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages[0]['content'] }}", encoding="utf-8")

    class Tokenizer:
        chat_template = None

    tokenizer = Tokenizer()
    SFTTrainer._apply_chat_template_path(tokenizer, str(template))

    assert tokenizer.chat_template == "{{ messages[0]['content'] }}"


def test_sft_packing_can_preserve_completion_mask():
    source = PackedShardedSource(
        source=_TinySource(),
        seq_length=8,
        eos_token_id=2,
        pad_token_id=0,
        strategy="greedy",
        include_segment_ids=True,
        extra_field_pad_values={"completion_mask": 0},
        extra_field_separator_values={"completion_mask": 0},
        shuffle=False,
    )

    [row] = list(source.open_shard("packed_shard_0"))

    assert row["input_ids"].tolist() == [10, 11, 2, 20, 2, 0, 0, 0]
    assert row["completion_mask"].tolist() == [0, 1, 0, 1, 0, 0, 0, 0]


def test_sft_chunked_nll_sets_loss_config_chunk_size():
    cfg = SFTConfig(loss_type="chunked_nll", completion_loss_chunk_size=128)

    assert isinstance(cfg.loss_config, LossConfig)
    assert cfg.loss_config.chunk_token_size == 128


def test_sft_preprocess_honors_truncation_mode_and_restores_tokenizer_side():
    class Tokenizer:
        eos_token = "<eos>"
        pad_token_id = 0

        def __init__(self):
            self.truncation_side = "right"

        def __call__(
            self,
            text,
            *,
            truncation,
            max_length,
            padding,
            return_attention_mask,
            add_special_tokens=True,
            **kwargs,
        ):
            del padding, return_attention_mask, add_special_tokens, kwargs
            ids = [int(piece) for piece in text.split()]
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer = Tokenizer()
    keep_start = SFTPreprocessTransform(
        tokenizer=tokenizer,
        max_length=2,
        truncation_mode="keep_start",
        add_eos=False,
        padding=False,
    )
    keep_end = SFTPreprocessTransform(
        tokenizer=tokenizer,
        max_length=2,
        truncation_mode="keep_end",
        add_eos=False,
        padding=False,
    )

    assert keep_start({"text": "1 2 3 4"})["input_ids"] == [1, 2]
    assert tokenizer.truncation_side == "right"
    assert keep_end({"text": "1 2 3 4"})["input_ids"] == [3, 4]
    assert tokenizer.truncation_side == "right"


def test_sft_dft_loss_weights_tokens_by_detached_probability():
    logits = jnp.array(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    labels = jnp.array([[0, 1, -100]], dtype=jnp.int32)

    metrics = _dft_causal_lm_metrics(logits, labels)
    logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)[:, 0, 1]
    expected = -jax.lax.stop_gradient(jnp.exp(logp)) * logp

    assert jnp.allclose(metrics.loss, expected)
