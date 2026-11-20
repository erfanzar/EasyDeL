"""Vocabulary-streamed on-policy distillation divergences on a real tensor-parallel mesh (vocab sharded over tp)."""

import easydel as ed
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.distillation_trainer._fn import chunked_distillation_loss, distillation_loss
from easydel.trainers.on_policy_distillation_trainer import _fn as opd_fn
from easydel.trainers.on_policy_distillation_trainer._fn import on_policy_distillation_step
from easydel.trainers.on_policy_distillation_trainer._streamed_divergences import (
    FullVocabProjection,
    VocabChunkedProjection,
    streamed_distillation_loss,
)
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

if jax.device_count() < 4:
    pytest.skip("needs a mesh of at least 4 devices", allow_module_level=True)

AXES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
TP4 = (1, 1, 1, 1, 4, 1)
B, L, H, V = 2, 9, 16, 1024
HI = jax.lax.Precision.HIGHEST
TEMPERATURE = 2.0

SELECTORS = {
    "forward": {},
    "reverse": {"beta": 1.0},
    "jsd": {"beta": 0.3},
    "cakld": {"cakld_gamma": 0.4},
    "topk": {"loss_top_k": 17},
    "topk_tail": {"loss_top_k": 17, "loss_add_tail": True},
}


def _mesh():
    return Mesh(np.array(jax.devices()[:4]).reshape(TP4), AXES)


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    h_s = rng.normal(size=(B, L, H)).astype(np.float32)
    h_t = rng.normal(size=(B, L, H)).astype(np.float32)
    w_s = (rng.normal(size=(H, V)) * 0.5).astype(np.float32)
    w_t = (rng.normal(size=(H, V)) * 0.5).astype(np.float32)
    mask = rng.integers(0, 2, size=(B, L)).astype(np.int32)
    mask[:, -3:] = 1
    return h_s, h_t, w_s, w_t, mask


def _place(mesh, h_s, h_t, w_s, w_t, mask):
    rep = NamedSharding(mesh, PartitionSpec())
    vocab = NamedSharding(mesh, PartitionSpec(None, "tp"))
    return (
        jax.device_put(h_s, rep),
        jax.device_put(h_t, rep),
        jax.device_put(w_s, vocab),
        jax.device_put(w_t, vocab),
        jax.device_put(mask, rep),
    )


def _streamed_fn(kind, chunk, options):
    def fn(hs, ht, ws, wt, mask):
        if kind == "vocab":
            s_proj = VocabChunkedProjection(ws, chunk_size=chunk, precision=HI)
            t_proj = VocabChunkedProjection(wt, chunk_size=chunk, precision=HI)
            seq_chunk = None
        else:
            s_proj = FullVocabProjection(lambda h: jnp.einsum("...h,hv->...v", h, ws, precision=HI))
            t_proj = FullVocabProjection(lambda h: jnp.einsum("...h,hv->...v", h, wt, precision=HI))
            seq_chunk = chunk
        return streamed_distillation_loss(
            hs,
            ht,
            s_proj,
            t_proj,
            loss_mask=mask,
            temperature=TEMPERATURE,
            alpha=1.0,
            sequence_chunk_size=seq_chunk,
            **options,
        )

    return jax.jit(jax.value_and_grad(fn, argnums=(0, 2), has_aux=True))


def _dense_fn(options):
    def fn(hs, ht, ws, wt, mask):
        return distillation_loss(
            student_logits=jnp.einsum("blh,hv->blv", hs, ws, precision=HI),
            teacher_logits=jnp.einsum("blh,hv->blv", ht, wt, precision=HI),
            loss_mask=mask,
            temperature=TEMPERATURE,
            alpha=1.0,
            **options,
        )

    return jax.jit(jax.value_and_grad(fn, argnums=(0, 2), has_aux=True))


def _numpy_forward_kl_and_grads(h_s, h_t, w_s, w_t, mask):
    """float64 forward KL ``T^2 * masked-mean KL(p_t || q_s)`` and its gradients wrt ``h_s`` and ``w_s``."""
    hs, ht = np.asarray(h_s, np.float64), np.asarray(h_t, np.float64)
    ws, wt = np.asarray(w_s, np.float64), np.asarray(w_t, np.float64)
    m = np.asarray(mask, np.float64)[:, 1:]
    norm = max(m.sum(), 1.0)

    def log_softmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))

    ls = log_softmax(hs[:, :-1] @ ws / TEMPERATURE)
    lt = log_softmax(ht[:, :-1] @ wt / TEMPERATURE)
    kl = (np.exp(lt) * (lt - ls)).sum(-1)
    loss = (kl * m).sum() / norm * TEMPERATURE**2
    # d loss / d student logits = T * m / N * (q - p).
    d_logits = TEMPERATURE * (m / norm)[..., None] * (np.exp(ls) - np.exp(lt))
    grad_w = np.einsum("blh,blv->hv", hs[:, :-1], d_logits)
    grad_h = np.zeros_like(hs)
    grad_h[:, :-1] = d_logits @ ws.T
    return loss, grad_h, grad_w


@pytest.mark.parametrize("kind,chunk", [("vocab", 128), ("vocab", 96), ("sequence", 4)])
@pytest.mark.parametrize("selector", list(SELECTORS))
def test_streamed_matches_unchunked_on_vocab_sharded_mesh(selector, kind, chunk):
    host = _problem()
    mesh = _mesh()
    placed = _place(mesh, *host)
    options = SELECTORS[selector]
    with mesh:
        (loss_c, metrics_c), (gh_c, gw_c) = _streamed_fn(kind, chunk, options)(*placed)
        (loss_d, metrics_d), (gh_d, gw_d) = _dense_fn(options)(*placed)
    # Unchunked on one device: sharding the vocabulary must not change the dense math either.
    (loss_1, _), (gh_1, gw_1) = _dense_fn(options)(*(jnp.asarray(x) for x in host))
    assert placed[2].sharding.spec == PartitionSpec(None, "tp")
    np.testing.assert_allclose(loss_c, loss_d, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(loss_d, loss_1, rtol=2e-5, atol=2e-6)
    for key in ("kl_loss", "distill_xent_loss", "teacher_entropy_loss"):
        np.testing.assert_allclose(metrics_c[key], metrics_d[key], rtol=2e-5, atol=2e-6, err_msg=key)
    np.testing.assert_allclose(gh_c, gh_d, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(gw_c, gw_d, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(gh_d, gh_1, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(gw_d, gw_1, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("kind,chunk", [("vocab", 128), ("vocab", 96), ("sequence", 4)])
def test_streamed_forward_kl_matches_float64_reference_on_mesh(kind, chunk):
    host = _problem(1)
    mesh = _mesh()
    with mesh:
        (loss, _), (grad_h, grad_w) = _streamed_fn(kind, chunk, {})(*_place(mesh, *host))
    ref_loss, ref_h, ref_w = _numpy_forward_kl_and_grads(*host)
    print(
        f"MEASURE opd-f64 {kind}{chunk}: loss {abs(float(loss) - ref_loss):.3e} "
        f"gh {np.max(np.abs(np.asarray(grad_h) - ref_h)):.3e} gw {np.max(np.abs(np.asarray(grad_w) - ref_w)):.3e} "
        f"gw.spec {grad_w.sharding.spec}"
    )
    np.testing.assert_allclose(loss, ref_loss, rtol=2e-5)
    np.testing.assert_allclose(grad_h, ref_h, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(grad_w, ref_w, rtol=1e-4, atol=1e-6)


# --- the full on-policy distillation step with tiny tp=4 Llama student/teacher ---------------------------------

VOCAB, ROWS, SEQ, PROMPT = 256, 4, 12, 6


def _llama(seed, axis_dims=TP4):
    cfg = ed.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        mlp_bias=False,
        sharding_axis_dims=axis_dims,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    with cfg.mesh:
        model = ed.LlamaForCausalLM(
            config=cfg, rngs=spx.Rngs(seed), dtype=jnp.float32, param_dtype=jnp.float32, precision=HI
        )
        return model.shard_model()


def _batch():
    rng = np.random.default_rng(3)
    attention = np.ones((ROWS, SEQ), np.int32)
    attention[0, :2] = 0
    ids = np.where(attention > 0, rng.integers(1, VOCAB, (ROWS, SEQ)), 0).astype(np.int32)
    completion = (np.arange(SEQ)[None, :] >= PROMPT).astype(np.int32) * attention
    completion[1, -2:] = 0
    return {
        "input_ids": jnp.asarray(ids),
        "attention_mask": jnp.asarray(attention),
        "completion_mask": jnp.asarray(completion),
    }


STEP_SELECTORS = {
    "forward": {},
    "jsd": {"beta": 0.3},
    "topk": {"loss_top_k": 17},
    "trust_region": {"distillation_mode": "trust_region", "trust_region_top_k": 16, "sampling_temperature": 1.0},
}


_MODELS: dict = {}


def _models(mesh_name):
    if mesh_name not in _MODELS:
        dims = TP4 if mesh_name == "tp4" else (1, 1, 1, 1, 1, 1)
        _MODELS[mesh_name] = (_llama(0, dims), _llama(1, dims))
    return _MODELS[mesh_name]


def _run_steps(mesh_name, chunk_sizes, options):
    student_model, teacher_model = _models(mesh_name)
    batch = _batch()
    with jax.default_matmul_precision("highest"), student_model.mesh:
        student = EasyDeLState.create(model=student_model, tx=optax.sgd(1.0), init_opt_state=True)
        teacher = EasyDeLState.create(model=teacher_model)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(student.graphstate)]
        results = {}
        for size in chunk_sizes:
            state, metrics = on_policy_distillation_step(
                student,
                batch,
                teacher,
                partition_spec=PartitionSpec(),
                is_training=True,
                temperature=TEMPERATURE,
                alpha=1.0,
                logits_chunk_size=size,
                **options,
            )
            after = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
            results[size] = (float(metrics.loss), [b - a for b, a in zip(before, after, strict=True)])
        s_logits = student_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        t_logits = teacher_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return results, s_logits, t_logits, batch


def _float64_forward_kl(s_logits, t_logits, completion_mask):
    s = np.asarray(s_logits, np.float64)[:, :-1] / TEMPERATURE
    t = np.asarray(t_logits, np.float64)[:, :-1] / TEMPERATURE
    ls = s - s.max(-1, keepdims=True)
    ls -= np.log(np.exp(ls).sum(-1, keepdims=True))
    lt = t - t.max(-1, keepdims=True)
    lt -= np.log(np.exp(lt).sum(-1, keepdims=True))
    m = np.asarray(completion_mask, np.float64)[:, 1:]
    return float(((np.exp(lt) * (lt - ls)).sum(-1) * m).sum() / m.sum() * TEMPERATURE**2)


@pytest.mark.parametrize("chunk", [64, 96])
@pytest.mark.parametrize("selector", list(STEP_SELECTORS))
def test_opd_step_chunked_matches_unchunked_on_tp_mesh(selector, chunk):
    head = _models("tp4")[0].get_lm_head().weight.value
    assert "tp" in str(head.sharding.spec), head.sharding.spec  # the vocabulary really is tp-sharded
    results, *_ = _run_steps("tp4", (None, chunk), STEP_SELECTORS[selector])
    (loss_d, upd_d), (loss_c, upd_c) = results[None], results[chunk]
    upd_err = max(float(np.max(np.abs(c - d))) for c, d in zip(upd_c, upd_d, strict=True))
    upd_max = max(float(np.max(np.abs(d))) for d in upd_d)
    print(
        f"MEASURE opd-step {selector}{chunk}: loss {abs(loss_c - loss_d):.3e} upd {upd_err:.3e} "
        f"(loss {loss_d:.4f} max|upd| {upd_max:.3e})"
    )
    assert upd_max > 0
    # Measured max update diff 2.1e-7 (trust_region); ~4x headroom.
    for c, d in zip(upd_c, upd_d, strict=True):
        np.testing.assert_allclose(c, d, rtol=0, atol=8.5e-7)
    if selector != "forward":  # forward-KL loss value: see the xfailed float64 tests below.
        # Measured max loss diff 9.9e-7 (trust_region); ~4x headroom.
        np.testing.assert_allclose(loss_c, loss_d, rtol=0, atol=4e-6)


# Max |diff| vs the plain dense reference over all selectors and chunk in (None, 64), measured on TPU tp=4:
# updates 5.96e-8 (forward dense, jsd streamed), loss 5.59e-7 (forward dense); ~4x headroom.
UPD_TOL_VS_PLAIN = 2.4e-7
LOSS_TOL_VS_PLAIN = 2.2e-6


@pytest.mark.parametrize("selector", list(STEP_SELECTORS))
def test_opd_step_tp_mesh_matches_plain_unchunked_reference(selector, monkeypatch):
    """Dense (vocab-parallel shard_map) and streamed steps on tp=4 vs the plain unchunked single-path dense step.

    The reference disables the vocab-parallel routing (no ``PartitionManager``), so the dense loss runs on the
    gathered ``[B, T, V]`` logits with no shard_map and no vocabulary or sequence chunking.
    """
    options = STEP_SELECTORS[selector]
    with monkeypatch.context() as patch:
        patch.setattr(opd_fn, "_vocab_parallel_partition_manager", lambda module: None, raising=False)
        loss_r, upd_r = _run_steps("tp4", (None,), options)[0][None]
    results, *_ = _run_steps("tp4", (None, 64), options)
    upd_max = max(float(np.max(np.abs(r))) for r in upd_r)
    assert upd_max > 0
    for chunk, (loss_t, upd_t) in results.items():
        upd_err = max(float(np.max(np.abs(t - r))) for t, r in zip(upd_t, upd_r, strict=True))
        print(
            f"MEASURE opd-step-vs-plain {selector} chunk={chunk}: loss {abs(loss_t - loss_r):.3e} "
            f"upd {upd_err:.3e} (loss {loss_r:.4f} max|upd| {upd_max:.3e})"
        )
    for chunk, (loss_t, upd_t) in results.items():
        for t, r in zip(upd_t, upd_r, strict=True):
            np.testing.assert_allclose(t, r, rtol=0, atol=UPD_TOL_VS_PLAIN, err_msg=f"chunk={chunk}")
        if not (selector == "forward" and chunk is not None):  # chunked forward-KL loss value: see _CANCELLATION.
            np.testing.assert_allclose(loss_t, loss_r, rtol=0, atol=LOSS_TOL_VS_PLAIN, err_msg=f"chunk={chunk}")


@pytest.mark.parametrize("selector", ["jsd", "topk", "trust_region"])
def test_opd_streamed_step_keeps_lm_head_gradient_tp_sharded(selector):
    """Streamed vocab-chunk OPD loss on tp=4: the jitted LM-head weight gradient keeps its ``tp`` axis.

    The gradient is taken of the step's loss directly (eval mode), so no optimizer update re-imposes the parameter
    sharding on the output. Note: measured on TPU, this also holds with the pre-fix lookup (the module-level
    ``partition_manager`` getattr), so it guards against regressions to full replication but does not by itself
    discriminate the prefix routing.
    """
    student_model, teacher_model = _models("tp4")
    options = STEP_SELECTORS[selector]

    def loss(graphstate, batch, teacher, student):
        metrics = on_policy_distillation_step(
            student.replace(graphstate=graphstate),
            batch,
            teacher,
            partition_spec=PartitionSpec(),
            is_training=False,
            temperature=TEMPERATURE,
            alpha=1.0,
            logits_chunk_size=64,
            **options,
        )
        return metrics.loss

    with jax.default_matmul_precision("highest"), student_model.mesh:
        student = EasyDeLState.create(model=student_model)
        teacher = EasyDeLState.create(model=teacher_model)
        flat = jax.tree_util.tree_flatten_with_path(student.graphstate)[0]
        (index,) = [i for i, (p, x) in enumerate(flat) if "lm_head" in jax.tree_util.keystr(p) and x.ndim == 2]
        param = flat[index][1]
        grad_fn = jax.jit(jax.grad(loss))
        args = (student.graphstate, _batch(), teacher, student)
        hlo = grad_fn.lower(*args).compile().as_text()
        grad = jax.tree_util.tree_leaves(grad_fn(*args))[index]
    head_shape = "f32[{},{}]".format(*param.shape)
    head_gathers = sum(1 for line in hlo.splitlines() if "all-gather" in line and head_shape in line)
    print(
        f"MEASURE opd-sharding {selector}: param {param.sharding.spec} "
        f"grad {getattr(grad.sharding, 'spec', grad.sharding)} full-head all-gathers {head_gathers}"
    )
    assert "tp" in str(param.sharding.spec)
    assert float(jnp.max(jnp.abs(grad))) > 0
    assert not grad.sharding.is_fully_replicated, grad.sharding
    assert "tp" in str(getattr(grad.sharding, "spec", "")), grad.sharding


_CANCELLATION = (
    "chunked forward KL (chunked_distillation_loss, distillation_trainer/_fn.py:365) reports kl = xent - entropy, "
    "two ~ln(V) float32 sums, so a small KL loses precision; measured rel. error vs float64 3.1e-4 (tp4) / "
    "3.1e-4-3.8e-4 (single device) at KL 0.025, dense path 3.6e-6-4.5e-6. Updates are unaffected (1.4e-8)."
)


_XFAIL_CANCELLATION = pytest.mark.xfail(strict=True, reason=_CANCELLATION)


@pytest.mark.parametrize(
    "chunk", [None, pytest.param(64, marks=_XFAIL_CANCELLATION), pytest.param(96, marks=_XFAIL_CANCELLATION)]
)
@pytest.mark.parametrize("mesh_name", ["tp4", "single"])
def test_opd_step_forward_kl_matches_float64_reference(mesh_name, chunk):
    results, s_logits, t_logits, batch = _run_steps(mesh_name, (chunk,), {})
    ref = _float64_forward_kl(s_logits, t_logits, batch["completion_mask"])
    loss = results[chunk][0]
    print(f"MEASURE opd-step-f64 {mesh_name} chunk={chunk}: ref {ref:.6f} rel-error {abs(loss - ref) / abs(ref):.3e}")
    # Dense measured rel. error <= 4.5e-6 -> 2e-5 (same as the single-device suite).
    np.testing.assert_allclose(loss, ref, rtol=2e-5)


def _near_teacher_problem():
    """Teacher close to the student: KL is ~1e-3 per token, entropy ~ln(V); the regime where cancellation shows."""
    h_s, _, w_s, _, mask = _problem(4)
    noise = np.random.default_rng(5).normal(size=w_s.shape).astype(np.float32)
    return h_s, h_s.copy(), w_s, w_s + 0.02 * noise, mask


def _chunked_kl_fn():
    def fn(hs, ht, ws, wt, mask):
        return chunked_distillation_loss(
            student_hidden=hs,
            teacher_hidden=ht,
            student_lm_head_fn=lambda h: jnp.einsum("...h,hv->...v", h, ws, precision=HI),
            teacher_lm_head_fn=lambda h: jnp.einsum("...h,hv->...v", h, wt, precision=HI),
            loss_mask=mask,
            temperature=TEMPERATURE,
            alpha=1.0,
            chunk_size=4,
        )

    return jax.jit(jax.value_and_grad(fn, argnums=(0, 2), has_aux=True))


def test_chunked_forward_kl_gradient_matches_float64_reference_on_mesh():
    host = _near_teacher_problem()
    mesh = _mesh()
    with mesh:
        (loss, _), (grad_h, grad_w) = _chunked_kl_fn()(*_place(mesh, *host))
    ref_loss, ref_h, ref_w = _numpy_forward_kl_and_grads(*host)
    print(
        f"MEASURE chunked-kl-f64: loss {float(loss):.6e} ref {ref_loss:.6e} rel {abs(float(loss) - ref_loss) / ref_loss:.3e} "
        f"gh {np.max(np.abs(np.asarray(grad_h) - ref_h)):.3e} gw {np.max(np.abs(np.asarray(grad_w) - ref_w)):.3e}"
    )
    np.testing.assert_allclose(grad_h, ref_h, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(grad_w, ref_w, rtol=1e-4, atol=1e-6)


@pytest.mark.xfail(strict=True, reason=_CANCELLATION)
def test_chunked_forward_kl_value_matches_float64_reference_on_mesh():
    """Minimal repro: chunked_distillation_loss alone with plain einsum heads, teacher near the student."""
    host = _near_teacher_problem()
    mesh = _mesh()
    with mesh:
        (loss, _), _ = _chunked_kl_fn()(*_place(mesh, *host))
    ref_loss, _, _ = _numpy_forward_kl_and_grads(*host)
    print(f"MEASURE chunked-kl-repro: rel {abs(float(loss) - ref_loss) / ref_loss:.3e}")
    np.testing.assert_allclose(loss, ref_loss, rtol=2e-5)
