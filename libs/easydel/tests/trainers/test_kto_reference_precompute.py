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

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from easydel.trainers.kto_trainer._fn import _prepare_kto_scheduled_batch
from easydel.trainers.kto_trainer.kto_config import KTOConfig
from easydel.trainers.kto_trainer.kto_trainer import KTOTrainer


class _FakeCall:
    def __init__(self, batch, **values):
        self.batch = batch
        self.schedule = SimpleNamespace(microbatches=1)
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeSource:
    shard_names = ("shard",)

    def __init__(self, sample):
        self.sample = sample

    def open_shard(self, shard_name):
        del shard_name
        return iter((self.sample,))


def test_kto_scheduled_batch_reuses_precomputed_reference_kl_logps():
    reference_kl = jnp.array([-0.1, -0.2], dtype=jnp.float32)
    batch = {
        "reference_logps": jnp.array([-1.0, -2.0], dtype=jnp.float32),
        "reference_KL_logps": reference_kl,
        "_policy_kl_logps": jnp.array([-0.3, -0.4], dtype=jnp.float32),
    }

    prepared = _prepare_kto_scheduled_batch(_FakeCall(batch, calculate_kl=True))

    assert prepared["_reference_kl_logps"] is reference_kl


def test_kto_source_reference_logps_check_requires_kl_when_needed():
    source_without_kl = _FakeSource({"reference_logps": 0.0})
    source_with_kl = _FakeSource({"reference_logps": 0.0, "reference_KL_logps": 0.0})

    assert KTOTrainer._source_has_reference_logps(source_without_kl)
    assert not KTOTrainer._source_has_reference_logps(source_without_kl, require_kl=True)
    assert KTOTrainer._source_has_reference_logps(source_with_kl, require_kl=True)


def test_kto_config_accepts_reference_precompute_batch_size():
    cfg = KTOConfig(precompute_ref_batch_size=7)

    assert cfg.precompute_ref_batch_size == 7


def test_kto_precompute_ref_batch_size_validates_positive():
    with pytest.raises(ValueError, match="precompute_ref_batch_size"):
        KTOConfig(precompute_ref_batch_size=0)


def test_kto_precompute_ref_batch_size_is_used_for_train_and_eval(monkeypatch):
    calls = []
    trainer = object.__new__(KTOTrainer)
    trainer.arguments = KTOConfig(precompute_ref_log_probs=True, precompute_ref_batch_size=7)
    trainer.dataset_train = object()
    trainer.dataset_eval = object()
    trainer._train_source = _FakeSource({"input_ids": [1]})
    trainer._eval_source = _FakeSource({"input_ids": [1]})
    trainer.calculate_kl = False
    monkeypatch.setattr(KTOTrainer, "training_batch_size", property(lambda self: 2))
    monkeypatch.setattr(KTOTrainer, "evaluation_batch_size", property(lambda self: 3))

    def fake_precompute(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(
        KTOTrainer,
        "_precompute_reference_log_probs_for_split",
        lambda self, **kwargs: fake_precompute(**kwargs),
    )
    from easydel.trainers.trainer.trainer import Trainer

    monkeypatch.setattr(Trainer, "configure_dataloaders", lambda self: "configured")

    assert KTOTrainer.configure_dataloaders(trainer) == "configured"
    assert [call["batch_size"] for call in calls] == [7, 7]
