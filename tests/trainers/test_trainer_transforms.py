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


class StrictChatTokenizer(MockTokenizer):
    """Tokenizer that rejects assistant-only chat templating."""

    def __init__(self):
        super().__init__()
        self.chat_template = "strict-chat-template"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        if not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user query found in messages.")

        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "assistant":
                parts.append(f"<assistant>{content}</assistant>")
            else:
                parts.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")

        text = "".join(parts)
        if tokenize:
            return [ord(c) % 100 for c in text]
        return text


class PromptMaskTokenizer(MockTokenizer):
    """Tokenizer that exposes chat-template-only prompt prefix tokens."""

    def __init__(self):
        super().__init__()
        self.chat_template = "prompt-mask-template"
        self.prompt_text = "[S]You are helpful.[U]Say hi.[A]"
        self.completion_text = "Hello!"
        self.full_text = f"{self.prompt_text}{self.completion_text}{self.eos_token}"

    def _render(self, messages, add_generation_prompt=False):
        parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                parts.append(f"[S]{content}")
            elif role == "user":
                parts.append(f"[U]{content}")
            elif role == "assistant":
                parts.append(f"[A]{content}")
            else:
                raise AssertionError(f"Unexpected role: {role!r}")
        if add_generation_prompt:
            parts.append("[A]")
        return "".join(parts)

    def _tokenize_text(self, text):
        if text == self.prompt_text:
            return [11, 12, 13]
        if text == self.full_text:
            return [99, 11, 12, 13, 14, self.eos_token_id]
        raise AssertionError(f"Unexpected text: {text!r}")

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
        tokens = self._tokenize_text(text)
        if max_length:
            tokens = tokens[:max_length]

        attention_mask = [1] * len(tokens)
        if padding == "max_length" and max_length:
            pad_len = max_length - len(tokens)
            if pad_len > 0:
                tokens = tokens + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        result = {"input_ids": tokens}
        if return_attention_mask:
            result["attention_mask"] = attention_mask
        return result

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        text = self._render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        if text == self.prompt_text:
            return [99, 11, 12, 13]
        if text == self.prompt_text + self.completion_text:
            return [99, 11, 12, 13, 14]
        raise AssertionError(f"Unexpected templated text: {text!r}")


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

    def test_prompt_completion_plain_strings(self):
        tokenizer = MockTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=128,
            mask_prompt=True,
        )

        result = transform(
            {
                "prompt": "Question: ",
                "completion": "Answer",
            }
        )

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "completion_mask" in result

    def test_prompt_completion_conversational_suffix_uses_template_token_ids(self):
        tokenizer = PromptMaskTokenizer()
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=16,
            mask_prompt=True,
        )

        result = transform(
            {
                "prompt": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hi."},
                ],
                "completion": [
                    {"role": "assistant", "content": "Hello!"},
                ],
            }
        )

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"] == [99, 11, 12, 13, 14, tokenizer.eos_token_id] + [0] * 10
        assert result["completion_mask"] == [0, 0, 0, 0, 1, 1] + [0] * 10

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

    def test_conversational_preference_drops_auxiliary_messages_column(self):
        tokenizer = MockTokenizer()
        transform = DPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "messages": [
                    {"role": "user", "content": "Side-channel prompt that should be ignored."},
                ],
                "chosen": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Good response"},
                ],
                "rejected": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Bad response"},
                ],
            }
        )

        assert "prompt_input_ids" in result
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result

    def test_conversational_preference_shared_prefix_with_strict_tokenizer(self):
        tokenizer = StrictChatTokenizer()
        transform = DPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "chosen": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity attracts bodies with mass."},
                ],
                "rejected": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity is a color."},
                ],
            }
        )

        assert "prompt_input_ids" in result
        assert "chosen_input_ids" in result
        assert "rejected_input_ids" in result

    def test_conversational_preference_preserves_multimodal_fields(self):
        tokenizer = StrictChatTokenizer()
        transform = DPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "chosen": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Describe the picture."},
                    {"role": "assistant", "content": "A cat is sleeping."},
                ],
                "rejected": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Describe the picture."},
                    {"role": "assistant", "content": "The image is unreadable."},
                ],
                "pixel_values": [[0.1, 0.2], [0.3, 0.4]],
                "pixel_attention_mask": [1, 1],
                "image_sizes": [224, 224],
            }
        )

        assert result["pixel_values"] == [[0.1, 0.2], [0.3, 0.4]]
        assert result["pixel_attention_mask"] == [1, 1]
        assert result["image_sizes"] == [224, 224]


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

    def test_conversational_preference_shared_prefix_with_strict_tokenizer(self):
        tokenizer = StrictChatTokenizer()
        transform = ORPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "chosen": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity attracts bodies with mass."},
                ],
                "rejected": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity is a color."},
                ],
            }
        )

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

    def test_conversational_preference_shared_prefix_with_strict_tokenizer(self):
        tokenizer = StrictChatTokenizer()
        transform = CPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "chosen": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity attracts bodies with mass."},
                ],
                "rejected": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity is a color."},
                ],
            }
        )

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

    def test_conversational_completion_suffix(self):
        tokenizer = StrictChatTokenizer()
        transform = KTOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "prompt": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hi."},
                ],
                "completion": [
                    {"role": "assistant", "content": "Hello!"},
                ],
                "label": True,
            }
        )

        assert "prompt_input_ids" in result
        assert "completion_input_ids" in result
        assert result["label"] is True

    def test_conversational_completion_preserves_multimodal_fields(self):
        tokenizer = StrictChatTokenizer()
        transform = KTOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = transform(
            {
                "prompt": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Describe the picture."},
                ],
                "completion": [
                    {"role": "assistant", "content": "A cat is sleeping."},
                ],
                "label": True,
                "pixel_values": [[0.5, 0.6], [0.7, 0.8]],
                "pixel_attention_mask": [1, 1],
                "image_sizes": [112, 112],
            }
        )

        assert result["pixel_values"] == [[0.5, 0.6], [0.7, 0.8]]
        assert result["pixel_attention_mask"] == [1, 1]
        assert result["image_sizes"] == [112, 112]

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

    def test_conversational_completion_suffix(self):
        tokenizer = StrictChatTokenizer()
        transform = BCOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        result = next(
            iter(
                transform(
                    {
                        "prompt": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "Say hi."},
                        ],
                        "completion": [
                            {"role": "assistant", "content": "Hello!"},
                        ],
                        "label": False,
                    }
                )
            )
        )

        assert "prompt_input_ids" in result
        assert "completion_input_ids" in result
        assert result["label"] is False

    def test_paired_conversational_preference_preserves_multimodal_fields(self):
        tokenizer = StrictChatTokenizer()
        transform = BCOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=256,
            max_completion_length=128,
        )

        results = list(
            transform(
                {
                    "chosen": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Describe the picture."},
                        {"role": "assistant", "content": "A cat is sleeping."},
                    ],
                    "rejected": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Describe the picture."},
                        {"role": "assistant", "content": "The image is unreadable."},
                    ],
                    "pixel_values": [[0.1, 0.2], [0.3, 0.4]],
                    "pixel_attention_mask": [1, 1],
                    "image_sizes": [224, 224],
                }
            )
        )

        assert len(results) == 2
        assert results[0]["label"] is True
        assert results[1]["label"] is False
        for result in results:
            assert result["pixel_values"] == [[0.1, 0.2], [0.3, 0.4]]
            assert result["pixel_attention_mask"] == [1, 1]
            assert result["image_sizes"] == [224, 224]


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

    def test_mixed_preference_format_keeps_explicit_prompt(self):
        """Explicit prompt should not be replaced by empty extracted prefix."""
        tokenizer = MockTokenizer()
        transform = GRPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=32,
        )

        example = {
            "prompt": "Solve 2+2",
            "chosen": [{"role": "assistant", "content": "4"}],
            "rejected": [{"role": "assistant", "content": "5"}],
        }

        result = transform(example)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert sum(result["attention_mask"]) > 0

    def test_empty_prompt_gets_fallback_token(self):
        tokenizer = MockTokenizer()
        transform = GRPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=16,
        )

        result = transform({"prompt": ""})
        assert "input_ids" in result
        assert "attention_mask" in result
        assert sum(result["attention_mask"]) > 0

    def test_empty_extracted_prompt_gets_fallback_token(self):
        tokenizer = MockTokenizer()
        transform = GRPOPreprocessTransform(
            tokenizer=tokenizer,
            max_prompt_length=16,
        )

        example = {
            "chosen": [{"role": "assistant", "content": "Good"}],
            "rejected": [{"role": "assistant", "content": "Bad"}],
        }

        result = transform(example)
        assert "attention_mask" in result
        assert sum(result["attention_mask"]) > 0


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

    def test_reward_conversational_preference_shared_prefix(self):
        tokenizer = StrictChatTokenizer()
        transform = RewardPreprocessTransform(
            tokenizer=tokenizer,
            max_length=512,
        )

        result = transform(
            {
                "chosen": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity attracts bodies with mass."},
                ],
                "rejected": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity is a color."},
                ],
            }
        )

        assert "input_ids_chosen" in result
        assert "input_ids_rejected" in result
        assert "attention_mask_chosen" in result
        assert "attention_mask_rejected" in result

    def test_reward_prompt_side_column_is_not_duplicated(self):
        tokenizer = MockTokenizer()
        transform = RewardPreprocessTransform(
            tokenizer=tokenizer,
            max_length=64,
        )

        result = transform(
            {
                "prompt": "Question: ",
                "chosen": "Question: Good answer",
                "rejected": "Question: Bad answer",
            }
        )

        expected_chosen = tokenizer(
            "Question: Good answer",
            truncation=True,
            max_length=64,
            padding="max_length",
            return_attention_mask=True,
        )
        expected_rejected = tokenizer(
            "Question: Bad answer",
            truncation=True,
            max_length=64,
            padding="max_length",
            return_attention_mask=True,
        )

        assert result["input_ids_chosen"] == expected_chosen["input_ids"]
        assert result["attention_mask_chosen"] == expected_chosen["attention_mask"]
        assert result["input_ids_rejected"] == expected_rejected["input_ids"]
        assert result["attention_mask_rejected"] == expected_rejected["attention_mask"]

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
            from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
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
