"""TPO user collators are captured intact during dataloader setup."""

import pytest
from easydel.trainers.direct_preference_optimization_trainer.dpo_trainer import DPOTrainer
from easydel.trainers.tpo_trainer import TPOConfig, TPOTrainer

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("backend", ["grain", "tfds"])
def test_user_collator_is_used_while_base_initializes(monkeypatch, backend):
    captured = {}

    def initialize(self, *, arguments, **kwargs):
        self.arguments = arguments
        self.padding_value = 0
        self.input_data_collator_grain = object()
        self.input_data_collator_tfds = object()
        captured["collate"] = getattr(self, f"create_{backend}_collect_function")(arguments.max_length)

    monkeypatch.setattr(DPOTrainer, "__init__", initialize)

    def collator(rows):
        return {"reference_input_ids": rows, "custom": True}

    TPOTrainer(TPOConfig(), model=object(), data_collator=collator)
    output = captured["collate"](["gold"])
    assert output == {"reference_input_ids": ["gold"], "custom": True}
