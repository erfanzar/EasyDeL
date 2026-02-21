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

"""Tests for trainer preprocessing transforms."""

import pytest

# Skip if transforms are not available
try:
    from easydel.trainers.prompt_transforms import (
        BCOPreprocessTransform,
        CPOPreprocessTransform,
        DPOPreprocessTransform,
        GRPOPreprocessTransform,
        KTOPreprocessTransform,
        ORPOPreprocessTransform,
        PPOPreprocessTransform,
        RewardPreprocessTransform,
        SFTPreprocessTransform,
    )
except ImportError as exc:
    pytest.skip(f"Trainer transforms unavailable: {exc}", allow_module_level=True)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
    ):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
        return_attention_mask=True,
        **kwargs,
    ):
        if isinstance(text, list):
            results = []
            for t in text:
                tokens = [ord(c) % 100 for c in t]
                if max_length:
                    tokens = tokens[:max_length]
                results.append(tokens)
            return {
                "input_ids": results,
                "attention_mask": [[1] * len(r) for r in results],
            }
        tokens = [ord(c) % 100 for c in text]
        if max_length:
            tokens = tokens[:max_length]
        result = {"input_ids": tokens}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(tokens)
        return result

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
    ):
        if tokenize:
            text = "".join(m.get("content", "") for m in messages)
            return [ord(c) % 100 for c in text]
        return "".join(m.get("content", "") for m in messages)


class TestSFTPreprocessTransform:
    """Tests for SFTPreprocessTransform."""

    def test_init(self):
        """Test transform initialization."""
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
        )
        assert transform._max_length == 128
        assert transform._mask_prompt is False

    def test_text_example(self):
        """Test processing a simple text example."""
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
            text_field="text",
        )

        example = {"text": "Hello, world!"}
        result = transform(example)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) <= 128

    def test_conversational_example(self):
        """Test processing a conversational example."""
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
        )

        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = transform(example)

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_messages_in_text_field(self):
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
            text_field="text",
        )

        example = {
            "text": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = transform(example)

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_messages_in_text_field_from_value(self):
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
            text_field="text",
        )

        example = {
            "text": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
        result = transform(example)

        assert "input_ids" in result
        assert "attention_mask" in result

    def test_repr(self):
        """Test string representation."""
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=256,
            mask_prompt=True,
        )
        repr_str = repr(transform)
        assert "SFTPreprocessTransform" in repr_str
        assert "256" in repr_str


class TestDPOPreprocessTransform:
    """Tests for DPOPreprocessTransform."""

    def test_init(self):
        """Test transform initialization."""
        tokenizer = MockTokenizer()
        transform = DPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )
        assert transform._max_prompt_length == 256
        assert transform._max_completion_length == 128

    def test_preference_example(self):
        """Test processing a preference example."""
        tokenizer = MockTokenizer()
        transform = DPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        example = {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
        }
        result = transform(example)

        assert "prompt_input_ids" in result
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result

    def test_conversational_preference(self):
        """Test processing conversational preference data."""
        tokenizer = MockTokenizer()
        transform = DPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        example = {
            "chosen": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Good response"},
            ],
            "rejected": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Bad response"},
            ],
        }
        result = transform(example)

        assert "prompt_input_ids" in result
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result


class TestORPOPreprocessTransform:
    """Tests for ORPOPreprocessTransform."""

    def test_init(self):
        """Test transform initialization."""
        tokenizer = MockTokenizer()
        transform = ORPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )
        assert transform._max_prompt_length == 256

    def test_preference_example(self):
        """Test processing a preference example."""
        tokenizer = MockTokenizer()
        transform = ORPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        example = {
            "prompt": "Question?",
            "chosen": "Good answer",
            "rejected": "Bad answer",
        }
        result = transform(example)

        assert "prompt_input_ids" in result
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result


class TestCPOPreprocessTransform:
    """Tests for CPOPreprocessTransform (alias for DPOPreprocessTransform)."""

    def test_is_alias(self):
        """Test that CPO is an alias for DPO preprocessing."""
        assert CPOPreprocessTransform is DPOPreprocessTransform

    def test_preference_example(self):
        """Test processing a preference example."""
        tokenizer = MockTokenizer()
        transform = CPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        example = {
            "prompt": "Question?",
            "chosen": "Good answer",
            "rejected": "Bad answer",
        }
        result = transform(example)

        assert "prompt_input_ids" in result
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result


class TestKTOPreprocessTransform:
    """Tests for KTOPreprocessTransform."""

    def test_init(self):
        """Test transform initialization."""
        tokenizer = MockTokenizer()
        transform = KTOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )
        assert transform._max_prompt_length == 256
        assert transform._max_completion_length == 128

    def test_kto_example(self):
        """Test processing a KTO example with binary label."""
        tokenizer = MockTokenizer()
        transform = KTOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        example = {
            "prompt": "Question?",
            "completion": "Answer",
            "label": True,
        }
        result = transform(example)

        assert "prompt_input_ids" in result
        assert "completion_input_ids" in result
        assert "label" in result

    def test_repr(self):
        """Test string representation."""
        tokenizer = MockTokenizer()
        transform = KTOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )
        repr_str = repr(transform)
        assert "KTOPreprocessTransform" in repr_str


class TestBCOPreprocessTransform:
    """Tests for BCOPreprocessTransform."""

    def test_bco_example(self):
        """Test processing a BCO example."""
        tokenizer = MockTokenizer()
        transform = BCOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        example = {
            "prompt": "Question?",
            "completion": "Answer",
            "label": False,
        }
        results = list(transform(example))
        assert len(results) == 1
        result = results[0]

        assert "prompt_input_ids" in result
        assert "completion_input_ids" in result
        assert "label" in result


class TestGRPOPreprocessTransform:
    """Tests for GRPOPreprocessTransform."""

    def test_init(self):
        """Test transform initialization."""
        tokenizer = MockTokenizer()
        transform = GRPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=1024,
        )
        assert transform._max_prompt_length == 1024

    def test_prompt_example(self):
        """Test processing a prompt-only example."""
        tokenizer = MockTokenizer()
        transform = GRPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=1024,
        )

        example = {
            "prompt": "Generate a story about a cat.",
        }
        result = transform(example)

        assert "input_ids" in result

    def test_conversational_prompt(self):
        """Test processing a conversational prompt."""
        tokenizer = MockTokenizer()
        transform = GRPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=1024,
        )

        example = {
            "messages": [
                {"role": "user", "content": "Tell me a joke"},
            ]
        }
        result = transform(example)

        assert "input_ids" in result


class TestPPOPreprocessTransform:
    """Tests for PPOPreprocessTransform."""

    def test_init(self):
        tokenizer = MockTokenizer()
        transform = PPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=1024,
        )
        assert transform._max_prompt_length == 1024

    def test_prompt_example(self):
        tokenizer = MockTokenizer()
        transform = PPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=1024,
        )

        example = {"prompt": "Explain PPO in one sentence."}
        result = transform(example)

        assert "input_ids" in result
        assert "attention_mask" in result


class TestRewardPreprocessTransform:
    """Tests for RewardPreprocessTransform."""

    def test_init(self):
        """Test transform initialization."""
        tokenizer = MockTokenizer()
        transform = RewardPreprocessTransform(
            tokenizer=tokenizer,
            max_length=512,
        )
        assert transform._max_length == 512

    def test_reward_example(self):
        """Test processing a reward model example."""
        tokenizer = MockTokenizer()
        transform = RewardPreprocessTransform(
            tokenizer=tokenizer,
            max_length=512,
        )

        example = {
            "chosen": "This is a good response.",
            "rejected": "This is a bad response.",
        }
        result = transform(example)

        assert "input_ids_chosen" in result
        assert "input_ids_rejected" in result
        assert "attention_mask_chosen" in result
        assert "attention_mask_rejected" in result

    def test_repr(self):
        """Test string representation."""
        tokenizer = MockTokenizer()
        transform = RewardPreprocessTransform(
            tokenizer=tokenizer,
            max_length=256,
        )
        repr_str = repr(transform)
        assert "RewardPreprocessTransform" in repr_str
        assert "256" in repr_str


class TestTransformChaining:
    """Test transform chaining with >> operator."""

    def test_chain_transforms(self):
        """Test chaining transforms."""
        from easydel.data.transforms.base import Transform

        tokenizer = MockTokenizer()
        transform1 = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
        )

        # SFT transform is a Transform subclass
        assert isinstance(transform1, Transform)


class TestTransformIntegration:
    """Integration tests for transforms with ShardedDataSource."""

    def test_transform_with_sharded_source(self):
        """Test applying transform to a ShardedDataSource."""
        from easydel.data.sources.hf_wrapper import wrap_hf_dataset

        try:
            from datasets import Dataset
        except ImportError:
            pytest.skip("datasets library not available")

        # Create a small test dataset
        data = {
            "text": ["Hello world", "Test example", "Another sample"],
        }
        hf_dataset = Dataset.from_dict(data)

        # Wrap as ShardedDataSource
        source = wrap_hf_dataset(hf_dataset)

        # Create transform
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
            text_field="text",
        )

        # Apply transform
        transformed = source.transform(transform)

        # Iterate and check
        for example in transformed.open_shard(transformed.shard_names[0]):
            assert "input_ids" in example
            assert "attention_mask" in example
            break  # Just check first example
