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

"""DeepSeek-V4.1 trains through the EasyDeL ``Trainer`` on a 4-device accelerator.

Computation tests (TPU/GPU only, never ``cpu_ok``):

* ``test_text_memorizes_through_trainer[ep|tp|fsdp]``: a real ``ed.Trainer(...).train()`` run on a tiny repeated
  synthetic dataset (4 fixed sequences) with the *engram* config. The host-computed ``engram_hash_ids`` travel through
  the Trainer as a dataset column (grain/source path: identity collator, then ``_purify_batch`` stacks them). Asserts
  every per-step loss is finite, no step raised, and the final loss is below half of the initial one.
* ``test_checkpoint_resume_continues_training``: train, save mid-run, build a new Trainer with
  ``resume_if_possible``; asserts the step counter resumes, the loaded parameters equal the live ones at the save step,
  and the resumed losses equal the uninterrupted run's losses at those steps.
* ``test_vl_trains_through_trainer_step``: ``DeepseekV41ForConditionalGeneration`` on one fixed image+text batch through
  the Trainer's own compiled step function (``trainers.trainer._fn.training_step``), driven directly because the
  Trainer data path cannot carry the per-image ``pixel_values`` (see the test docstring). Asserts the loss decreases
  and every vision-side parameter moves.

Tunables: ``STEPS``, ``LEARNING_RATE`` (memorization speed) and the ``MESHES`` table.
"""

import typing as tp

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]
from easydel.infra.base_state import EasyDeLState
from easydel.modules.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41ForConditionalGeneration,
)
from easydel.modules.deepseek_v41._engram_hash import EngramHasher
from easydel.trainers.trainer._fn import training_step
from spectrax.serialization import find_latest_checkpoint

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

VOCAB = ref.VOCAB
SEQ_LEN = 32
BATCH = 4  # divisible by every 4-way batch axis (fsdp, or ep with moe_ep_carries_batch)
NUM_SEQUENCES = 4  # distinct fixed sequences; each batch holds all of them, so every step sees identical data
STEPS = 30
LEARNING_RATE = 8e-3
RESUME_TOTAL_STEPS = 6
RESUME_SAVE_STEP = 4
MODEL_NAME = "deepseek-v41-train"

#: ``sharding_axis_dims`` order is ``(pp, dp, fsdp, ep, tp, sp)``.
MESHES: dict[str, dict[str, tp.Any]] = {
    "ep": dict(
        sharding_axis_dims=(1, 1, 1, 4, 1, 1),
        use_expert_tensor_mode=False,  # DeepSeek MoE on ep > 1
        fsdp_is_ep_bound=False,  # moe_ep_carries_batch must be given to the constructor together with these two
        sp_is_ep_bound=False,
        moe_ep_carries_batch=True,
    ),
    "tp": dict(sharding_axis_dims=(1, 1, 1, 1, 4, 1)),
    "fsdp": dict(sharding_axis_dims=(1, 1, 4, 1, 1, 1)),
}


class _Tokenizer:
    """Minimal tokenizer stand-in for pretokenized datasets (no network)."""

    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __len__(self):
        return VOCAB


def _finalize(config: DeepseekV41Config) -> DeepseekV41Config:
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


def _text_config(mesh_name: str) -> DeepseekV41Config:
    """Tiny engram-enabled DeepSeek-V4.1 config on one of the ``MESHES``."""
    kwargs = ref.case_config_kwargs("engram")
    kwargs.update(MESHES[mesh_name])
    kwargs["moe_force_xla_gmm"] = True
    return _finalize(DeepseekV41Config(**kwargs))


def _build_model(model_cls, config):
    with config.mesh:
        model = model_cls.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        return model.shard_model()


def _fixed_sequences() -> np.ndarray:
    """``int64[NUM_SEQUENCES, SEQ_LEN]``: distinct tokens, so every token determines its successor within the data."""
    perm = np.random.default_rng(1234).permutation(np.arange(3, VOCAB))
    return perm[: NUM_SEQUENCES * SEQ_LEN].reshape(NUM_SEQUENCES, SEQ_LEN).astype(np.int64)


def _text_dataset(config: DeepseekV41Config, num_rows: int) -> Dataset:
    """Rows are the fixed sequences repeated, with their host-computed engram hash ids as an extra column."""
    base = _fixed_sequences()
    hashes = EngramHasher.from_config(config, np.arange(VOCAB, dtype=np.int64)).hash(base)  # [4, S, n_layers, cols]
    repeats = -(-num_rows // NUM_SEQUENCES)
    ids = np.tile(base, (repeats, 1))[:num_rows]
    hash_rows = np.tile(hashes, (repeats, 1, 1, 1))[:num_rows]
    return Dataset.from_dict(
        {
            "input_ids": ids.tolist(),
            "attention_mask": np.ones_like(ids).tolist(),
            "engram_hash_ids": hash_rows.tolist(),
        }
    )


def _make_trainer(
    config: DeepseekV41Config,
    save_directory,
    *,
    steps: int,
    save_steps: int | None = None,
    do_last_save: bool = False,
    resume: bool = False,
    model=None,
):
    arguments = ed.TrainingArguments(
        model_name=MODEL_NAME,
        save_directory=str(save_directory),
        max_training_steps=steps,
        num_train_epochs=1,
        total_batch_size=BATCH,
        gradient_accumulation_steps=1,
        max_length=SEQ_LEN,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.NONE,
        learning_rate=LEARNING_RATE,
        warmup_steps=0,
        weight_decay=0.0,
        clip_grad=1.0,
        use_wandb=False,
        do_last_save=do_last_save,
        save_steps=save_steps,
        save_optimizer_state=True,
        save_tpu_preemption_checkpoints=False,
        resume_if_possible=resume,
        log_steps=1_000_000,
        report_steps=1_000_000,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )
    return ed.Trainer(
        arguments=arguments,
        model=model if model is not None else _build_model(DeepseekV41ForCausalLM, config),
        dataset_train=_text_dataset(config, steps * BATCH),
        processing_class=_Tokenizer(),
    )


def _param_arrays(graphstate) -> dict[str, np.ndarray]:
    """Host copies of every floating ``parameters`` leaf of a graphstate, keyed by ``collection:path``."""
    out: dict[str, np.ndarray] = {}
    for collection, path, leaf in graphstate.items():
        if collection != "parameters":
            continue
        arr = leaf.value if hasattr(leaf, "value") else leaf
        if hasattr(arr, "dtype") and getattr(arr, "size", 0) and jnp.issubdtype(arr.dtype, jnp.floating):
            out[f"{collection}:{path}"] = np.asarray(arr)
    assert out, "graphstate exposes no floating parameters"
    return out


class _StepRecorder:
    """Wraps ``Trainer._execute_train_step`` to record per-step losses, swallowed exceptions and param snapshots."""

    def __init__(self, trainer, snapshot_steps: tp.Iterable[int] = ()):
        self.losses: list[float] = []
        self.steps: list[int] = []
        self.errors: list[BaseException] = []
        self.snapshots: dict[int, dict[str, np.ndarray]] = {}
        self._snapshot_steps = set(snapshot_steps)
        inner = trainer._execute_train_step

        def spy(state, batch):
            new_state, metrics, exc = inner(state=state, batch=batch)
            if exc is not None:  # the trainer returns (does not raise) TypeError / OOM; surface it
                self.errors.append(exc)
            else:
                self.losses.append(float(jax.device_get(metrics.loss)))
                step = int(jax.device_get(new_state.step))
                self.steps.append(step)
                if step in self._snapshot_steps:
                    self.snapshots[step] = _param_arrays(new_state.graphstate)
            return new_state, metrics, exc

        trainer._execute_train_step = spy


@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_text_memorizes_through_trainer(mesh_name, tmp_path):
    """``Trainer.train()`` on the engram config memorizes 4 fixed sequences: finite losses, final < 0.5 * initial."""
    config = _text_config(mesh_name)
    trainer = _make_trainer(config, tmp_path, steps=STEPS)
    recorder = _StepRecorder(trainer)
    output = trainer.train()

    assert not recorder.errors, f"train step returned an exception: {recorder.errors[0]!r}"
    assert recorder.steps == list(range(1, STEPS + 1))
    assert int(jax.device_get(output.state.step)) == STEPS
    losses = np.asarray(recorder.losses)
    assert np.isfinite(losses).all(), f"non-finite losses: {losses}"
    assert losses[-1] < 0.5 * losses[0], f"{mesh_name}: loss did not halve: first {losses[0]:.4f} last {losses[-1]:.4f}"


def test_checkpoint_resume_continues_training(tmp_path):
    """Save at step 4 of 6, resume in a fresh Trainer: step, parameters and the following losses all match."""
    mesh_name = "fsdp"
    config = _text_config(mesh_name)
    first = _make_trainer(config, tmp_path, steps=RESUME_TOTAL_STEPS, save_steps=RESUME_SAVE_STEP)
    reference = _StepRecorder(first, snapshot_steps=[RESUME_SAVE_STEP])
    first.train()
    assert not reference.errors, f"uninterrupted run raised: {reference.errors[0]!r}"
    assert reference.steps == list(range(1, RESUME_TOTAL_STEPS + 1))
    assert find_latest_checkpoint(str(tmp_path / MODEL_NAME)) is not None, "no checkpoint was written"

    resumed = _make_trainer(_text_config(mesh_name), tmp_path, steps=RESUME_TOTAL_STEPS, resume=True)
    assert int(jax.device_get(resumed.model_state.step)) == RESUME_SAVE_STEP

    saved = reference.snapshots[RESUME_SAVE_STEP]
    loaded = _param_arrays(resumed.model_state.graphstate)
    assert loaded.keys() == saved.keys()
    for name, value in saved.items():
        np.testing.assert_allclose(loaded[name], value, rtol=0, atol=1e-6, err_msg=name)

    after = _StepRecorder(resumed)
    output = resumed.train()
    assert not after.errors, f"resumed run raised: {after.errors[0]!r}"
    assert after.steps == [RESUME_SAVE_STEP + 1, RESUME_SAVE_STEP + 2]
    assert int(jax.device_get(output.state.step)) == RESUME_TOTAL_STEPS
    # first resumed loss uses the loaded params; the second also exercises the restored optimizer moments
    np.testing.assert_allclose(after.losses, reference.losses[RESUME_SAVE_STEP:], rtol=1e-3, atol=1e-5)


def _host_table_model(mesh_name: str):
    """Engram config with fp8 tables in host RAM (the released-checkpoint layout), tables filled with random bytes."""
    import ml_dtypes

    device = jax.devices()[0]
    if "pinned_host" not in {memory.kind for memory in device.addressable_memories()}:
        pytest.skip(f"backend {device.platform!r} exposes no pinned_host memory")
    kwargs = ref.case_config_kwargs("engram")
    kwargs.update(MESHES[mesh_name])
    kwargs.update(engram_table_dtype="fp8", engram_table_memory_kind="pinned_host")
    config = _finalize(DeepseekV41Config(**kwargs))
    model = _build_model(DeepseekV41ForCausalLM, config)
    rng = np.random.default_rng(7)
    head_dim = int(config.engram_head_dim)
    installed = 0
    for layer in model.model.layers:
        if layer.engram is None:
            continue
        rows = int(layer.engram.table_rows)
        weight = (rng.standard_normal((rows, head_dim)) * 0.5).astype(ml_dtypes.float8_e4m3fn)
        scale = rng.integers(125, 130, size=(rows, head_dim // 32), dtype=np.uint8)
        layer.engram.set_table(weight, scale)
        assert layer.engram.uses_host_table
        installed += 1
    assert installed == len(config.engram_layer_ids)
    return config, model


def _release_host_tables(model) -> None:
    for layer in model.model.layers:
        if layer.engram is not None:
            layer.engram.release_host_table()


@pytest.mark.parametrize("mesh_name", ["ep", "tp"])
def test_host_batch_transform_matches_in_program_engram_lookup(mesh_name):
    """Host-table engram: the trainer hook's host-gathered rows give the same loss and gradients as the in-program
    ``pure_callback`` lookup on ``engram_hash_ids``, and the transformed batch's step never traces that lookup.

    Not on the ``fsdp`` mesh: there ``fsdp_is_ep_bound`` folds fsdp into a 3-axis ``(dp, ep, tp)`` expert mesh, and
    JAX rejects that second abstract mesh in any program holding a host callback, so the callback *reference* cannot
    compile (the hook's callback-free path trains on fsdp; see the trainer test below).
    """
    from easydel.layers.embeddings import GatedEngram

    config, model = _host_table_model(mesh_name)
    try:
        ids = _fixed_sequences()
        hashes = EngramHasher.from_config(config, np.arange(VOCAB, dtype=np.int64)).hash(ids)
        host_batch = {"input_ids": ids.astype(np.int32), "engram_hash_ids": np.asarray(hashes, np.int32)}
        transformed = type(model).host_batch_transform(model, dict(host_batch))
        assert "engram_hash_ids" not in transformed
        assert transformed["engram_rows"].dtype == np.uint8 and transformed["engram_scales"].dtype == np.uint8
        assert transformed["engram_rows"].shape == (*hashes.shape, int(config.engram_head_dim))
        graphdef, state = spx.export(model)

        def loss(gs, batch):
            batch = dict(batch)
            tokens = batch.pop("input_ids")
            logits = spx.bind(graphdef, gs)(input_ids=tokens, **batch).logits.astype(jnp.float32)
            return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], tokens[:, 1:]).mean()

        def run(batch):
            device_batch = {k: jnp.asarray(v) for k, v in batch.items()}
            # allow_int: the host-table placeholder buffers are uint8 (float0 cotangents, skipped below).
            return jax.jit(jax.value_and_grad(loss, allow_int=True))(state, device_batch)

        with config.mesh:
            lookup_loss, lookup_grads = run(host_batch)
            original = GatedEngram._host_gather_rows

            def _forbidden(self, ids):
                raise AssertionError("in-program host engram lookup traced for a host-gathered batch")

            GatedEngram._host_gather_rows = _forbidden
            try:
                gathered_loss, gathered_grads = run(transformed)
            finally:
                GatedEngram._host_gather_rows = original

        assert np.isfinite(float(lookup_loss))
        np.testing.assert_allclose(float(gathered_loss), float(lookup_loss), rtol=1e-6, atol=1e-7)
        checked = 0
        for (path, a), b in zip(
            jax.tree_util.tree_leaves_with_path(lookup_grads), jax.tree_util.tree_leaves(gathered_grads), strict=True
        ):
            a, b = np.asarray(a), np.asarray(b)
            if not a.size or not np.issubdtype(a.dtype, np.floating):
                continue
            scale = max(float(np.abs(a).max()), 1e-12)
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-5 * scale, err_msg=jax.tree_util.keystr(path))
            checked += 1
        assert checked > 0
    finally:
        _release_host_tables(model)


def test_host_table_engram_trains_through_trainer_without_callback(tmp_path, monkeypatch):
    """``Trainer.train()`` on host-table engram: batches are host-gathered by the model hook, so the compiled step
    never traces the in-program lookup (patched to raise); losses stay finite and fall."""
    from easydel.layers.embeddings import GatedEngram

    steps = 8
    config, model = _host_table_model("fsdp")
    try:

        def _forbidden(self, ids):
            raise AssertionError("in-program host engram lookup traced in the train step")

        monkeypatch.setattr(GatedEngram, "_host_gather_rows", _forbidden)
        trainer = _make_trainer(config, tmp_path, steps=steps, model=model)
        recorder = _StepRecorder(trainer)
        trainer.train()
        assert not recorder.errors, f"train step returned an exception: {recorder.errors[0]!r}"
        assert recorder.steps == list(range(1, steps + 1))
        losses = np.asarray(recorder.losses)
        assert np.isfinite(losses).all(), losses
        assert losses[-1] < losses[0], losses
    finally:
        _release_host_tables(model)


def test_vl_trains_through_trainer_step():
    """VL head, one fixed image+text batch, the Trainer's compiled ``training_step``: loss falls, vision params move.

    Driven directly instead of ``ed.Trainer(...).train()``: the Trainer data path builds batches row by row
    (``_purify_batch`` stacks one array per example), but DeepSeek-V4.1 ``pixel_values`` are per *image*, not per row
    (``modeling_deepseek_v41.py:1491-1495`` requires the IMAGE slot count to equal the image rows, and the test batch
    has an image in row 0 only), and a flat ``image_grid`` tuple would turn into a tracer under jit
    (``_prepare_patches``). The gridded layout ``[N, h, w, 3, p, p]`` avoids the static grid. The step function is the
    exact one ``Trainer.configure_functions`` compiles (``trainer/trainer.py:268``), so loss/grad/optimizer code is shared.
    """
    config = _finalize(DeepseekV41Config(**ref.vl_config_kwargs()))
    assert config.vision_enabled
    inputs = ref.vl_inputs(config)
    hashes = EngramHasher.from_config(config, np.arange(VOCAB, dtype=np.int64)).hash(inputs.ids, inputs.token_types >= 0)
    n_h, n_w = inputs.grid
    patch = inputs.patches.shape[-1]
    ids = jnp.asarray(inputs.ids, dtype=jnp.int32)
    token_types = jnp.asarray(inputs.token_types, dtype=jnp.int32)
    batch = {
        "input_ids": ids,
        "attention_mask": jnp.ones_like(ids),
        "labels": jnp.where(token_types >= 0, -100, ids),  # supervise text positions only
        "token_types": token_types,
        "engram_hash_ids": jnp.asarray(hashes, dtype=jnp.int32),
        "pixel_values": jnp.asarray(inputs.patches).reshape(1, n_h, n_w, 3, patch, patch),
    }

    model = _build_model(DeepseekV41ForConditionalGeneration, config)
    lr_fn = optax.constant_schedule(5e-3)

    @ed.ejit(static_argnums=(2, 3, 4, 5, 6, 7))  # pyright: ignore[reportUntypedFunctionDecorator]
    def step(st, b, loss_config, learning_rate_fn, partition_spec, grad_accum, ste, loss_type):
        return training_step(st, b, loss_config, learning_rate_fn, partition_spec, grad_accum, ste, loss_type)

    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(5e-3), init_opt_state=True)
        before = _param_arrays(state.graphstate)
        losses = []
        for _ in range(12):
            state, metrics = step(state, batch, None, lr_fn, None, 1, None, "nll")
            losses.append(float(metrics.loss))
        after = _param_arrays(state.graphstate)

    assert int(jax.device_get(state.step)) == 12
    assert np.isfinite(losses).all(), f"non-finite losses: {losses}"
    assert losses[-1] < 0.85 * losses[0], f"VL loss did not decrease: {losses}"

    vision = sorted(n for n in before if any(tag in n for tag in ("vision", "aligner", "image_")))
    assert vision, "no vision-side parameters found"
    assert all(np.isfinite(after[n]).all() for n in after)
    unchanged = [n for n in vision if np.array_equal(before[n], after[n])]
    assert not unchanged, f"vision-side parameters that did not move: {unchanged[:10]}"
    text = [n for n in before if n not in vision]
    assert any(not np.array_equal(before[n], after[n]) for n in text), "no text parameter moved"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
