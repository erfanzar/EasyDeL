"""GRPO preprocessing keeps per-example reward metadata and drops consumed text (host-only)."""

import pytest
from easydel.trainers.prompt_transforms import GRPOPreprocessTransform

from .test_preprocessing import TextTokenizer

pytestmark = pytest.mark.cpu_ok


def _transform():
    return GRPOPreprocessTransform(tokenizer=TextTokenizer(), max_prompt_length=None, skip_apply_chat_template=True)


def test_gold_answer_and_structured_metadata_survive_preprocessing():
    row = {
        "prompt": "what is 2+2",
        "answer": "4",
        "verification_info": {"tests": ["assert f() == 4"]},
        "tools": [{"name": "calc"}],
        "difficulty": 0.5,
    }
    out = _transform()(row)
    assert out["answer"] == "4"
    assert out["verification_info"] == {"tests": ["assert f() == 4"]}
    assert out["tools"] == [{"name": "calc"}]
    assert "prompt" not in out
    assert "difficulty" not in out
    assert len(out["input_ids"]) == len(out["attention_mask"]) > 0


def test_grpo_collators_keep_text_columns_as_python_lists():
    from easydel.trainers.utils import GRPODataCollatorGrain, GRPODataCollatorTFDS

    t = _transform()
    rows = [t({"prompt": "a b", "answer": "4"}), t({"prompt": "c", "answer": "10"})]
    tfds = GRPODataCollatorTFDS(max_prompt_length=4, pad_token_id=0)(rows)
    assert tfds["answer"] == ["4", "10"]
    assert tfds["input_ids"].shape == (2, 4)
    grain = GRPODataCollatorGrain(max_prompt_length=4, pad_token_id=0)(rows[0])
    assert grain["answer"] == "4"
