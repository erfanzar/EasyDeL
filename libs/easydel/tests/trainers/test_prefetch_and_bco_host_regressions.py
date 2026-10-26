"""Host-only data, generation, prefetch and step-binding contract regressions."""

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import spectrax as spx
from datasets import Dataset
from easydel.trainers.base_trainer import BaseTrainer
from easydel.trainers.batch_prefetcher import TrainBatchPrefetcher
from easydel.trainers.binary_classifier_optimization_trainer import _fn
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer
from easydel.trainers.trainer.trainer import Trainer
from easydel.trainers.utils import BCODataCollatorGrain
from transformers import ProcessorMixin

from .test_bco_host_state import TextTokenizer

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("workers", [1, 3])
def test_prefetch_resume_across_epochs_and_inactive_buckets(workers):
    rows = [{"value": i} for i in range(30)]

    def fetch(iterator, loader):
        return next(iterator), iterator

    def collate(raw):
        # A mutating collator must not corrupt the raw replay copy.
        return raw.pop("value")

    active_iter, inactive_iter = iter(rows), iter([{"value": i} for i in range(100, 130)])
    delivered = []
    for _ in range(3):
        active = TrainBatchPrefetcher(
            fetch_fn=fetch,
            data_iter=active_iter,
            dataloader=rows,
            data_collator=collate,
            worker_count=workers,
            buffer_size=5,
        )
        inactive = TrainBatchPrefetcher(
            fetch_fn=fetch,
            data_iter=inactive_iter,
            dataloader=rows,
            data_collator=collate,
            worker_count=workers,
            buffer_size=5,
        )
        # Same lifecycle as Trainer: save the iterator before closing producers.
        try:
            delivered.extend(active.next(schedule_next=True)[0] for _ in range(2))
        finally:
            active_iter, inactive_iter = active.data_iter, inactive.data_iter
            active.close()
            inactive.close()
        active.close()  # Idempotent, including after handing out its iterator.
    assert delivered == list(range(6))
    assert [row["value"] for row in active_iter] == list(range(6, 30))
    assert [row["value"] for row in inactive_iter] == list(range(100, 130))


def bco_host_trainer(*, encoder_decoder=False):
    trainer = object.__new__(BCOTrainer)
    trainer.arguments = SimpleNamespace(
        max_prompt_length=8,
        max_completion_length=8,
        label_pad_token_id=-100,
        is_encoder_decoder=encoder_decoder,
        ids_to_pop_from_dataset=[],
    )
    trainer.processing_class = TextTokenizer()
    trainer.embedding_tokenizer = None
    return trainer


def test_reference_precompute_preserves_tokenization_and_collation():
    trainer = bco_host_trainer()
    trainer.dataset_train = Dataset.from_dict(
        {"prompt": ["a", "b"], "completion": ["yes", "no"], "label": [True, False]}
    )
    trainer._train_source = trainer._build_source_from_dataset(trainer.dataset_train)
    # The score boundary is mocked, not the lazy tokenizer, source, or collator.
    grain = BCODataCollatorGrain(8, 8, 0, -100, False)
    trainer.data_collator = lambda rows: {key: np.stack([grain(row)[key] for row in rows]) for key in grain(rows[0])}
    trainer.compute_reference_log_probs = lambda batch: np.asarray([-float(i) for i in batch["prompt_input_ids"][:, -1]])
    assert trainer._precompute_reference_log_probs_for_split(
        dataset_attr="dataset_train",
        source_attr="_train_source",
        batch_size=2,
        is_train=True,
        desc="host reference score fixture",
    )
    rows = [row for shard in trainer._train_source.shard_names for row in trainer._train_source.open_shard(shard)]
    assert rows[0]["prompt_input_ids"] == [ord("a")]
    assert rows[1]["completion_labels"] == [-100, ord("n"), ord("o"), 2]
    batch = trainer.data_collator(rows)
    np.testing.assert_array_equal(batch["reference_logps"].reshape(-1), [-97.0, -98.0])
    assert batch["completion_input_ids"].shape == (2, 16)


def test_encoder_decoder_transform_separates_prompt_from_targets():
    trainer = bco_host_trainer(encoder_decoder=True)
    transform = trainer._get_preprocess_transform()
    raw = {"prompt": "ab", "completion": "c", "label": True, "reference_logps": -3.0}
    row = next(transform(raw))
    assert row["prompt_input_ids"] == [ord("a"), ord("b")]
    assert row["completion_input_ids"] == [ord("c"), 2]
    assert row["completion_labels"] == [ord("c"), 2]
    assert row["completion_attention_mask"] == [1, 1]
    assert row["reference_logps"] == -3.0
    assert next(transform(row)) == row


@pytest.mark.parametrize("pooled_output", [False, True])
def test_udm_fitting_and_batches_share_batched_masked_pooling(pooled_output):
    trainer = bco_host_trainer()
    trainer.embedding_tokenizer = TextTokenizer()

    def embed(input_ids, attention_mask):
        assert input_ids.ndim == 2
        token_embeddings = np.stack([input_ids, input_ids * 2], axis=-1).astype(np.float32)
        if pooled_output:
            return token_embeddings[:, 0, :]
        return (token_embeddings, None)

    trainer.embedding_func = embed
    trainer.arguments.prompt_sample_size = 2
    trainer._rng = np.random.default_rng(0)
    dataset = Dataset.from_dict({"prompt": ["a", "b"], "completion": ["yes", "no"], "label": [True, False]})
    trainer._train_source = trainer._build_source_from_dataset(dataset)
    trainer._train_density_ratio_classifier()
    assert trainer.clf_weights[0].shape == (2,)
    batched = trainer._vectorize_prompt(np.asarray([[97, 0], [98, 0]]), np.asarray([[1, 0], [1, 0]]))
    np.testing.assert_array_equal(batched, [[97.0, 194.0], [98.0, 196.0]])
    single = trainer._vectorize_prompt(np.asarray([97, 0]), np.asarray([1, 0]))
    np.testing.assert_array_equal(single, batched[:1])


def generation_args(**kwargs):
    return SimpleNamespace(
        **{
            "generation_max_new_tokens": 4,
            "generation_temperature": 0.0,
            "generation_top_p": 1.0,
            "generation_top_k": 0,
            "generation_num_return_sequences": 1,
            "generation_do_sample": True,
            **kwargs,
        }
    )


@pytest.mark.parametrize(
    "overrides,do_sample,temperature",
    [
        ({}, True, 0.0),
        ({}, False, 0.0),
        ({"temperature": 0.6}, False, 0.0),
        ({"do_sample": True, "temperature": 0.6}, False, 0.6),
    ],
)
def test_sampling_preserves_zeros_and_greedy_backend_parity(overrides, do_sample, temperature):
    trainer = object.__new__(Trainer)
    params = trainer._build_generation_sampling_params(generation_args(generation_do_sample=do_sample), overrides)
    assert params.temperature == temperature
    assert params.top_k == 0
    assert params.top_p == 1.0


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("eos,model_eos,expected", [(2, None, 2), ([3, 2], [2, 4], 3), (None, [5], 5)])
def test_padding_falls_back_to_ordered_eos_ids(nested, eos, model_eos, expected):
    trainer = object.__new__(Trainer)
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=eos)
    if nested:
        processor = object.__new__(ProcessorMixin)
        processor.tokenizer = tokenizer
        trainer.processing_class = processor
    else:
        trainer.processing_class = tokenizer
    trainer._model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=model_eos))
    assert trainer._pad_token_id == expected


def test_padding_without_pad_or_eos_raises_explicit_error():
    trainer = object.__new__(Trainer)
    trainer.processing_class = SimpleNamespace(pad_token_id=None, eos_token_id=None)
    trainer._model = SimpleNamespace(generation_config=None)
    with pytest.raises(ValueError, match="pad_token_id or an eos_token_id"):
        _ = trainer._pad_token_id


@dataclass
class ModeState:
    model: object

    def replace(self, *, graphdef):
        return ModeState(spx.bind(graphdef, spx.State({})))


@pytest.mark.parametrize("disable_dropout", [False, True])
def test_base_initial_policy_mode_respects_disable_dropout(disable_dropout):
    class ModeModule(spx.Module):
        pass

    class SetupComplete(Exception):
        pass

    class ModeProbeTrainer(Trainer):
        def _apply_runtime_model_config_overrides(self):
            # Stop before shape inference, device setup and dataloader construction.
            raise SetupComplete

    trainer = object.__new__(ModeProbeTrainer)
    args = SimpleNamespace(model_name="mode_probe", resume_if_possible=False, disable_dropout=disable_dropout)
    state = ModeState(ModeModule().eval())
    with pytest.raises(SetupComplete):
        BaseTrainer.__init__(trainer, arguments=args, model_state=state)
    assert trainer.model_state.model.training is (not disable_dropout)


def test_compile_aot_binds_real_bco_signatures_after_processing():
    trainer = object.__new__(Trainer)
    trainer.model_state = "train_state"
    trainer.arguments = SimpleNamespace(ids_to_pop_from_dataset=["metadata"])
    trainer.dataloader_train = [[{"raw": "train", "metadata": "drop"}]]
    trainer.dataloader_eval = [[{"raw": "eval", "metadata": "drop"}]]

    def collate(rows):
        assert "metadata" not in rows[0]
        return {"collated": rows[0]["raw"]}

    trainer.data_collator = collate
    trainer._preprocess_batch_input = lambda state, batch, is_train: ({**batch, "mode": is_train}, {})
    trainer._eval_state_for_compiled_step = lambda state: "eval_state"
    reference = object()
    forward = object()
    trainer._train_shared_fn_extra_args = (reference,)
    trainer._train_shared_fn_static_args = (None, forward, 0.5, None, None, 1, None)
    trainer._eval_shared_fn_extra_args = (reference,)
    trainer._eval_shared_fn_static_args = (forward, 0.5)
    bound_calls = []

    class SignatureLowerer:
        def __init__(self, target):
            self.signature = inspect.signature(target)

        def lower(self, *args):
            binding = self.signature.bind(*args)
            bound_calls.append(binding.arguments)
            return SimpleNamespace(compile=lambda: object())

    train_fn, eval_fn = SignatureLowerer(_fn.training_step), SignatureLowerer(_fn.evaluation_step)
    trainer.sharded_training_step_function = train_fn
    trainer.sharded_evaluation_step_function = eval_fn
    assert trainer.compile_aot()
    assert [call["state"] for call in bound_calls] == ["train_state", "eval_state"]
    assert [call["batch"] for call in bound_calls] == [
        {"collated": "train", "mode": True},
        {"collated": "eval", "mode": False},
    ]
    assert all(call["reference_state"] is reference for call in bound_calls)
    assert all(call["concatenated_forward_fn"] is forward for call in bound_calls)
    # Normal execution still supplies static args, so the public wrapper must survive AOT warming.
    assert trainer.sharded_training_step_function is train_fn
    assert trainer.sharded_evaluation_step_function is eval_fn
