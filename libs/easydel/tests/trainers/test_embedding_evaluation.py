"""Contrastive evaluation coverage (accelerator-only numerics)."""

from types import SimpleNamespace

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from datasets import Dataset
from easydel.trainers.embedding_trainer._fn import embedding_evaluation_step
from jax import numpy as jnp


class Encoder:
    mesh = None

    def eval(self):
        self.training = False

    def __call__(self, input_ids, attention_mask):
        embeddings = input_ids.astype(jnp.float32)
        # Evaluation must disable the encoder's training-time perturbation.
        if getattr(self, "training", True):
            embeddings = jnp.ones_like(embeddings)
        return SimpleNamespace(embeddings=embeddings)


@pytest.mark.parametrize("loss_type", ["infonce", "mnrl", "triplet"])
def test_embedding_evaluation_scores_query_positive_batch(loss_type):
    state = SimpleNamespace(graphstate={}, merge=lambda tree: Encoder())
    batch = {
        "query_input_ids": jnp.asarray([[1, 0], [0, 1]]),
        "positive_input_ids": jnp.asarray([[1, 0], [0, 1]]),
        "negative_input_ids": jnp.asarray([[0, 1], [1, 0]]),
    }
    for prefix in ("query", "positive", "negative"):
        batch[f"{prefix}_attention_mask"] = jnp.ones((2, 2), dtype=jnp.int32)
    metrics = embedding_evaluation_step(state, batch, loss_type=loss_type, temperature=1.0)
    if loss_type == "triplet":
        np.testing.assert_allclose(metrics.loss, 0.0)
        np.testing.assert_allclose(metrics.other_metrics["fraction_active_triplets"], 0.0)
    else:
        np.testing.assert_allclose(metrics.loss, np.log(2 * np.e + 2) - 1, rtol=1e-6)
    assert state.graphstate == {}


@pytest.mark.parametrize("matryoshka_dims", [None, [1, 2]])
def test_embedding_evaluation_zero_vectors_have_finite_uniform_loss(matryoshka_dims):
    state = SimpleNamespace(graphstate={}, merge=lambda tree: Encoder())
    batch = {}
    for prefix in ("query", "positive"):
        batch[f"{prefix}_input_ids"] = jnp.zeros((4, 2), dtype=jnp.int32)
        batch[f"{prefix}_attention_mask"] = jnp.ones((4, 2), dtype=jnp.int32)
    metrics = embedding_evaluation_step(state, batch, matryoshka_dims=matryoshka_dims)
    # Every candidate score is zero; each query assigns probability 1/4 to its
    # positive, independent of temperature or truncated embedding dimension.
    np.testing.assert_allclose(metrics.loss, np.log(4), rtol=1e-6)
    assert state.graphstate == {}


class MaskedEncoder(Encoder):
    def __call__(self, input_ids, attention_mask):
        # Fixed embedding width independent of the stream's token width.
        token_embeddings = jnp.stack((input_ids == 1, input_ids == 2), axis=-1).astype(jnp.float32)
        embeddings = (token_embeddings * attention_mask[..., None]).sum(axis=1)
        return SimpleNamespace(embeddings=embeddings)


@pytest.mark.parametrize("loss_type", ["infonce", "mnrl", "triplet"])
@pytest.mark.parametrize("include_negatives", [False, True])
def test_embedding_evaluation_allows_independently_padded_streams(loss_type, include_negatives):
    state = SimpleNamespace(graphstate={}, merge=lambda tree: MaskedEncoder())
    batch = {
        "query_input_ids": jnp.asarray([[1, 2], [2, 1]]),
        "query_attention_mask": jnp.asarray([[1, 0], [1, 0]]),
        "positive_input_ids": jnp.asarray([[1, 2, 2], [2, 1, 1]]),
        "positive_attention_mask": jnp.asarray([[1, 0, 0], [1, 0, 0]]),
    }
    if include_negatives:
        batch.update(
            negative_input_ids=jnp.asarray([[2, 1, 1, 1], [1, 2, 2, 2]]),
            negative_attention_mask=jnp.asarray([[1, 0, 0, 0], [1, 0, 0, 0]]),
        )
    metrics = embedding_evaluation_step(state, batch, loss_type=loss_type, temperature=1.0)
    if loss_type == "triplet":
        np.testing.assert_allclose(metrics.loss, 0.0)
        np.testing.assert_allclose(metrics.other_metrics["fraction_active_triplets"], 0.0)
    else:
        expected = np.log(np.e + 1) - 1
        if include_negatives:
            expected += np.log(2)
        np.testing.assert_allclose(metrics.loss, expected, rtol=1e-6)
        np.testing.assert_allclose(metrics.accuracy, 1.0)


class Tokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __len__(self):
        return 32


@pytest.mark.parametrize("eval_overrides", [None, {"attn_mechanism": "vanilla"}])
def test_configured_embedding_evaluator_accepts_eval_config_overrides(tmp_path, eval_overrides):
    config = ed.Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        attn_mechanism="sdpa",
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
    )
    config.attn_dtype = jnp.float32
    # Materializing abstract lazy-init leaves zero-fills parameters; it is not
    # weight initialization. Use real initialized weights for a nontrivial loss.
    model = ed.Qwen2ForEmbedding(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    ).shard_model()
    examples = {
        "query_input_ids": [[2, 3], [4, 5], [6, 7], [8, 9]],
        "query_attention_mask": [[1, 1]] * 4,
        "positive_input_ids": [[2, 3, 0], [4, 5, 0], [6, 7, 0], [8, 9, 0]],
        "positive_attention_mask": [[1, 1, 0]] * 4,
    }
    dataset = Dataset.from_dict(examples)
    arguments = ed.EmbeddingConfig(
        model_name="embedding-eval-overrides",
        save_directory=str(tmp_path),
        max_training_steps=1,
        total_batch_size=4,
        eval_batch_size=4,
        max_length=4,
        temperature=0.7,
        matryoshka_dims=[8, 16],
        eval_config_overrides=eval_overrides,
        use_wandb=False,
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )
    trainer = ed.EmbeddingTrainer(
        arguments=arguments,
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=Tokenizer(),
    )
    training_graphdef = trainer.model_state.graphdef
    eval_state = trainer._eval_state_for_compiled_step(trainer.model_state)
    batch = {key: jnp.asarray(value) for key, value in examples.items()}
    # Run the actual configured compiler and evaluator, including their static
    # state metadata. A training-graphdef sharding tree rejects the override.
    metrics = trainer.sharded_evaluation_step_function(
        eval_state, batch, *trainer._eval_shared_fn_extra_args, *trainer._eval_shared_fn_static_args
    )
    eval_model = eval_state.merge(eval_state.graphstate)
    eval_model.eval()
    assert str(eval_model.config.attn_mechanism) == ("vanilla" if eval_overrides else "sdpa")
    embeddings = []
    for prefix in ("query", "positive"):
        outputs = eval_model(
            input_ids=batch[f"{prefix}_input_ids"],
            attention_mask=batch[f"{prefix}_attention_mask"],
        )
        embeddings.append(np.asarray(outputs.embeddings, dtype=np.float32))
    assert all(np.isfinite(values).all() for values in embeddings)
    assert all(np.linalg.norm(values) > 0 for values in embeddings)
    losses = []
    for dim in arguments.matryoshka_dims:
        query, positive = (values[:, :dim] for values in embeddings)
        # Zero vectors (including truncated prefixes) stay zero under the
        # encoder/trainer's epsilon-protected L2 normalization.
        query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
        positive = positive / np.maximum(np.linalg.norm(positive, axis=-1, keepdims=True), 1e-12)
        scores = query @ positive.T / arguments.temperature
        scores -= scores.max(axis=-1, keepdims=True)
        losses.append(np.mean(np.log(np.exp(scores).sum(axis=-1)) - np.diag(scores)))
    np.testing.assert_allclose(metrics.loss, np.mean(losses), rtol=2e-4, atol=2e-5)
    assert trainer.model_state.graphdef is training_graphdef
