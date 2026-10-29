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

"""Host-only trainer contracts: bf16 batch leaves, packed-embed microbatch slicing, terminal save reuse, chat payloads."""

from __future__ import annotations

import functools
import json
import os
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from easydel.data.transforms.collators import collate_packed_embeds
from easydel.infra.loss_utils import LossConfig
from easydel.trainers.base_trainer import BaseTrainer
from easydel.trainers.prompt_transforms import SFTPreprocessTransform
from easydel.trainers.prompt_utils import normalize_message_payload
from easydel.trainers.training_utils import _infer_batch_size, slice_host_minibatch
from easydel.trainers.utils import _prompt_aligned_padding_value, is_numeric_dtype

pytestmark = pytest.mark.cpu_ok

BF16 = ml_dtypes.bfloat16


@pytest.mark.parametrize("dtype", [BF16, ml_dtypes.float8_e4m3fn, np.float16, np.int32, np.bool_, np.complex64])
def test_is_numeric_dtype_accepts_low_precision_and_bool(dtype):
    assert is_numeric_dtype(np.dtype(dtype))


@pytest.mark.parametrize("dtype", [np.dtype("<U4"), np.dtype(object), "not-a-dtype"])
def test_is_numeric_dtype_rejects_text_and_object(dtype):
    assert not is_numeric_dtype(dtype)


def test_purify_batch_keeps_bf16_leaf_and_drops_text():
    embeds = np.ones((2, 3), dtype=BF16)
    batch = {
        "input_ids": np.arange(6, dtype=np.int64).reshape(2, 3),
        "image_embeds": embeds,
        "answer": np.asarray(["a", "b"]),
    }

    purified = BaseTrainer._purify_batch(SimpleNamespace(), batch)

    assert set(purified) == {"input_ids", "image_embeds"}
    assert purified["image_embeds"].dtype == BF16
    assert purified["input_ids"].dtype == np.int32


def test_purify_batch_list_branch_keeps_bf16_array():
    purified = BaseTrainer._purify_batch(SimpleNamespace(), {"w": [np.ones(2, dtype=BF16), np.ones(2, dtype=BF16)]})

    assert purified["w"].dtype == BF16
    assert purified["w"].shape == (2, 2)


def test_reward_sidechannels_treat_bf16_columns_as_numeric():
    batch = {"hidden": np.ones((2, 3), dtype=BF16), "answer": ["x", "y"]}

    side = BaseTrainer._extract_reward_batch_sidechannels(batch)

    assert "hidden" not in side
    assert side["answer"] == ["x", "y"]


def test_prompt_aligned_padding_is_zero_for_bf16():
    value = _prompt_aligned_padding_value("image_embeds", np.zeros((2, 2), dtype=BF16), pad_token_id=7)

    assert value == 0.0
    assert _prompt_aligned_padding_value("input_ids", np.zeros((2, 2), dtype=np.int32), pad_token_id=7) == 7


def test_loss_config_skip_nonfinite_updates_roundtrips():
    config = LossConfig(skip_nonfinite_updates=True)

    assert LossConfig().skip_nonfinite_updates is False
    assert LossConfig.from_dict(config.to_dict()).skip_nonfinite_updates is True
    assert LossConfig.from_json(config.to_json()).skip_nonfinite_updates is True
    assert "skip_nonfinite_updates" in repr(config)


IMAGE_TOKEN_ID = 99
EMBED_DIM = 4
SEQ_LEN = 16


def _position_id_fn(input_ids, mm_token_type_ids=None, image_grid_thw=None, attention_mask=None):
    length = np.asarray(input_ids).shape[-1]
    return np.arange(length, dtype=np.int32).reshape(1, 1, -1).repeat(3, axis=0), None


def _packed_row(n_images: int, tag: float) -> dict:
    ids = [1, 2] + [IMAGE_TOKEN_ID] * n_images
    embeds = (np.arange(n_images * EMBED_DIM, dtype=np.float32) + tag).reshape(n_images, EMBED_DIM)
    return {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "attention_mask": np.ones(len(ids), dtype=np.int32),
        "labels": np.asarray(ids, dtype=np.int32),
        "image_embeds": [embeds.astype(BF16).tobytes()] if n_images else [],
        "embed_n_tok": [n_images] if n_images else [],
        "image_grid_thw": np.ones((n_images, 3), dtype=np.int32),
    }


def _packed_batch(counts: list[int], spare_rows: int = 2) -> dict:
    """A real ``collate_packed_embeds`` batch with one document per window (window == batch row)."""
    packs = [[_packed_row(c, 10.0 * (w + 1))] for w, c in enumerate(counts)]
    return collate_packed_embeds(
        packs,
        0,
        SEQ_LEN,
        sum(counts) + spare_rows,
        len(counts),
        image_token_id=IMAGE_TOKEN_ID,
        embed_dim=EMBED_DIM,
        position_id_fn=_position_id_fn,
    )


def _scatter_by_coordinates(batch: dict) -> np.ndarray:
    """Reference merge: place each real embed row at its recorded (batch row, position)."""
    ids = np.asarray(batch["input_ids"])
    embeds = np.asarray(batch["image_embeds"], dtype=np.float32)
    out = np.zeros((*ids.shape, EMBED_DIM), dtype=np.float32)
    for idx in np.flatnonzero(np.asarray(batch["image_embed_mask"])):
        row, pos = batch["image_embed_positions"][idx]
        out[row, pos] = embeds[idx]
    return out


def _merge_by_placeholder_order(batch: dict) -> np.ndarray:
    """Reference of the model-side merge: k-th placeholder (row-major) takes the k-th embed row."""
    ids = np.asarray(batch["input_ids"])
    embeds = np.asarray(batch["image_embeds"], dtype=np.float32)
    mask = ids == IMAGE_TOKEN_ID
    padded = np.concatenate([np.zeros((1, EMBED_DIM), np.float32), embeds], axis=0)
    updates = padded[np.cumsum(mask.reshape(-1))].reshape(*ids.shape, EMBED_DIM)
    return np.where(mask[..., None], updates, 0.0)


@pytest.mark.parametrize(
    ("counts", "steps"),
    [([2, 0, 3, 1], 2), ([2, 0, 3, 1], 4), ([0, 0, 0, 0], 2), ([1, 1, 1, 1], 2), ([3, 0, 0, 2], 2), ([0, 4, 0, 0], 4)],
)
def test_host_microbatches_pair_each_row_with_its_own_embeds(counts, steps):
    batch_size = len(counts)
    size = batch_size // steps
    full = _packed_batch(counts)
    expected = _merge_by_placeholder_order(full)
    np.testing.assert_array_equal(_scatter_by_coordinates(full), expected)

    pieces = []
    for k in range(steps):
        rows = slice(k * size, (k + 1) * size)
        micro = slice_host_minibatch(full, batch_size, k * size, size)
        owned = sum(counts[rows])

        assert micro["image_embeds"].shape == full["image_embeds"].shape
        assert micro["image_embed_positions"].shape == full["image_embed_positions"].shape
        np.testing.assert_array_equal(micro["input_ids"], full["input_ids"][rows])
        np.testing.assert_array_equal(micro["position_ids"], full["position_ids"][:, rows])
        np.testing.assert_array_equal(_merge_by_placeholder_order(micro), expected[rows])
        np.testing.assert_array_equal(_scatter_by_coordinates(micro), expected[rows])
        assert int(micro["image_embed_mask"].sum()) == owned
        assert int(micro["n_real_embeds"]) == owned
        assert not np.any(micro["image_embeds"][owned:])
        if owned:
            assert micro["image_embed_positions"][:owned, 0].min() >= 0
            assert micro["image_embed_positions"][:owned, 0].max() < size
        pieces.append(np.asarray(micro["image_embeds"][:owned], dtype=np.float32))

    total = sum(counts)
    np.testing.assert_array_equal(
        np.concatenate(pieces, axis=0), np.asarray(full["image_embeds"][:total], dtype=np.float32)
    )


def test_host_slicing_uses_embed_n_tok_when_coordinates_are_absent():
    embeds = np.arange(8 * 5, dtype=np.float32).reshape(8, 5) + 1.0
    batch = {
        "input_ids": np.zeros((4, 8), dtype=np.int32),
        "image_embeds": embeds,
        "embed_n_tok": np.asarray([2, 0, 3, 1], dtype=np.int32),
    }

    first = slice_host_minibatch(batch, 4, 0, 2)
    second = slice_host_minibatch(batch, 4, 2, 2)

    assert first["image_embeds"].shape == second["image_embeds"].shape == embeds.shape
    np.testing.assert_array_equal(first["image_embeds"][:2], embeds[:2])
    assert not np.any(first["image_embeds"][2:])
    np.testing.assert_array_equal(second["image_embeds"][:4], embeds[2:6])
    assert not np.any(second["image_embeds"][4:])
    np.testing.assert_array_equal(second["embed_n_tok"], [3, 1])


def test_host_slicing_rejects_unusable_ownership_markers():
    batch = _packed_batch([1, 1, 1, 1])
    lone = {k: v for k, v in batch.items() if k != "image_embed_mask"}
    with pytest.raises(ValueError, match="provided together"):
        slice_host_minibatch(lone, 4, 0, 2)

    overflow = {
        "input_ids": np.zeros((4, 8), dtype=np.int32),
        "image_embeds": np.zeros((3, 5), dtype=np.float32),
        "embed_n_tok": np.asarray([2, 2, 2, 2], dtype=np.int32),
    }
    with pytest.raises(ValueError, match="sums to"):
        slice_host_minibatch(overflow, 4, 0, 2)

    short = {**overflow, "image_embeds": np.zeros((8, 5), dtype=np.float32), "embed_n_tok": np.asarray([2, 2, 2])}
    with pytest.raises(ValueError, match="one entry per batch row"):
        slice_host_minibatch(short, 4, 0, 2)


def test_full_slice_of_packed_batch_and_plain_batches_are_unchanged():
    full = _packed_batch([2, 0, 3, 1])
    whole = slice_host_minibatch(full, 4, 0, 4)
    for key in ("image_embeds", "image_embed_positions", "image_embed_mask", "input_ids"):
        np.testing.assert_array_equal(whole[key], full[key])

    plain = slice_host_minibatch({"input_ids": np.arange(8).reshape(4, 2)}, 4, 2, 2)
    np.testing.assert_array_equal(plain["input_ids"], np.arange(8).reshape(4, 2)[2:4])


class _DiskCheckpointer:
    """Writes one directory per save; periodic saves may be temporary, forced saves never are."""

    def __init__(self, root, periodic_is_temporary):
        self.root = root
        self.periodic_is_temporary = periodic_is_temporary

    def on_step(self, *, mesh, pytree, step, force, true_callbacks):
        del mesh, pytree
        dest = self.root / f"run-{step}-{len(list(self.root.iterdir()))}"
        meta = {"step": step, "is_temporary": self.periodic_is_temporary and not force}
        for callback in true_callbacks:
            callback(str(dest), None, meta)


def _disk_trainer(root, *, periodic_is_temporary=False):
    def save_state(*, state, save_directory, merge_lora_before_save):
        os.makedirs(save_directory)
        return save_directory

    trainer = SimpleNamespace(
        checkpointer=_DiskCheckpointer(root, periodic_is_temporary),
        mesh=None,
        arguments=SimpleNamespace(
            do_last_save=True, merge_lora_before_save=False, save_directory=None, _get_save_directory=lambda: None
        ),
        _preemption_checkpoint_path=None,
        _train_end_callbacks_dispatched=True,
        _last_saved_checkpoint=None,
        _save_state=save_state,
        _save_data_lane_state=lambda path, step: None,
        _cleanup_old_checkpoints=lambda: None,
        _sync_checkpointer_after_callback_save=lambda *, step: None,
        _dispatch_callbacks=lambda *args, **kwargs: None,
    )
    trainer._save_checkpoint_for_step = functools.partial(BaseTrainer._save_checkpoint_for_step, trainer)
    return trainer


def _train_then_finish(trainer, *, periodic_step, final_step):
    periodic = trainer._save_checkpoint_for_step(SimpleNamespace(step=periodic_step), step=periodic_step)
    out = BaseTrainer._prepare_training_output(trainer, SimpleNamespace(step=final_step))
    return periodic, out


def test_terminal_save_reuses_permanent_checkpoint_of_the_same_step(tmp_path):
    periodic, out = _train_then_finish(_disk_trainer(tmp_path), periodic_step=10, final_step=10)

    assert out.checkpoint_path == periodic
    assert out.last_save_file_name == periodic
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-10-0"]


def test_terminal_save_writes_when_last_save_was_an_earlier_step(tmp_path):
    periodic, out = _train_then_finish(_disk_trainer(tmp_path), periodic_step=5, final_step=10)

    assert out.checkpoint_path != periodic
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-10-1", "run-5-0"]


def test_temporary_checkpoint_does_not_stand_in_for_the_terminal_save(tmp_path):
    trainer = _disk_trainer(tmp_path, periodic_is_temporary=True)
    periodic, out = _train_then_finish(trainer, periodic_step=10, final_step=10)

    assert out.checkpoint_path != periodic
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run-10-0", "run-10-1"]


def test_normalize_message_payload_none_content_and_tool_calls():
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]

    plain = normalize_message_payload([{"role": "user", "content": None}])
    assert plain == [{"role": "user", "content": ""}]

    with_calls = normalize_message_payload([{"role": "assistant", "content": None, "tool_calls": calls}])
    assert with_calls[0]["content"] is None
    assert with_calls[0]["tool_calls"] == calls

    stringified = normalize_message_payload([{"role": "assistant", "content": None, "tool_calls": json.dumps(calls)}])
    assert stringified[0]["tool_calls"] == calls
    assert stringified[0]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'


_WORDS = ["[PAD]", "[UNK]", "<u>", "<a>", "<e>", *[f"w{i}" for i in range(20)]]
_TWO_TURNS = {
    "messages": [
        {"role": "user", "content": "w1 w2"},
        {"role": "assistant", "content": "w3 w4"},
        {"role": "user", "content": "w5 w6"},
        {"role": "assistant", "content": "w7 w8"},
    ]
}


def _tokenizer(template: str):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(_WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="<e>")
    tokenizer.chat_template = template
    return tokenizer


@pytest.mark.parametrize(
    ("open_tag", "close_tag"),
    [
        ("{% generation %}", "{% endgeneration %}"),
        ("{%- generation -%}", "{%- endgeneration %}"),
        ("{%-generation%}", "{%- endgeneration %}"),
    ],
)
def test_generation_tag_whitespace_variants_enable_assistant_mask(open_tag, close_tag):
    template = (
        "{% for m in messages %}{% if m['role'] == 'assistant' %}"
        f"{{{{ '<a> ' }}}}{open_tag}{{{{ m['content'] }}}} <e>{close_tag} "
        "{% else %}<u> {{ m['content'] }} {% endif %}{% endfor %}"
    )
    transform = SFTPreprocessTransform(_tokenizer(template), max_length=64, mask_prompt=True, padding=False)

    row = transform(_TWO_TURNS)

    assert row["assistant_masks"] == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]


def test_batch_size_ignores_embed_row_and_per_image_leaves():
    rows, embed_rows = 4, 8
    batch = {
        "input_ids": np.zeros((rows, 16), np.int32),
        "position_ids": np.zeros((3, rows, 16), np.int32),
        "image_embeds": np.zeros((embed_rows, 4), BF16),
        "image_embed_positions": np.zeros((embed_rows, 2), np.int32),
        "image_embed_mask": np.zeros((embed_rows,), np.int32),
        "image_grid_thw": np.zeros((embed_rows, 3), np.int32),
    }
    assert _infer_batch_size(batch) == rows
    assert _infer_batch_size({"image_embeds": batch["image_embeds"]}) == embed_rows
