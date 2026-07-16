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

"""Tiny-but-faithful repro of the ep>1 distillation-training NaN (run "M3").

The production-shaped M3 run (qwen3_5_moe 35B-A3B distillation, prismcore
``lloyd_adam_mirror`` ternary optimizer, mesh ``(1,4,16,4,4,1)``) finished
step 1 with a finite KL and died at step 2 with "NaN Loss"; the identical
recipe on the ep=1 mesh trained 1000+ steps. The fused-MoE ep>1 dispatch is
already TPU-parity-proven in isolation (scripts/tpu_probe_moe_ep_parity.py),
so this probe reproduces the *training step*: a tiny qwen3_5_moe hybrid
(GDN + full attention, 256 tiny experts), student + frozen teacher, the real
``distillation_step`` (chunked KD path), and the real optimizer chain
(clip_by_global_norm(1.0) + lloyd_adam_mirror with production kwargs).

Detection is per-step and per-leaf: ``metrics.grad_norms`` (the trainer
already computes a per-leaf norm tree) is checked for non-finite leaves and
the student params / opt-state are checked after every update — the M3
trainer only gates on the scalar loss, which is one step too late.

Run from the WORKTREE root (ships edited libs + prismcore):

    /home/erfan/easy-venv/bin/eray run -- \
        /home/erfan/easy-venv/bin/python scripts/tpu_repro_ep_nan.py
    /home/erfan/easy-venv/bin/eray logs <jobname> -f
    /home/erfan/easy-venv/bin/eray stop <jobname>

Env knobs (forwarded to the pod):
    REPRO_MESHES      comma list of ep1 / ep4 (default "ep1,ep4")
    REPRO_STEPS       optimizer steps per mesh (default 6)
    REPRO_OPT         lloyd | adamw (default lloyd)
    REPRO_LOSS        kd | ce (default kd; ce drops the teacher/KL entirely)
    REPRO_LAYERS      hybrid | full (default hybrid: GDN,GDN,GDN,full)
    REPRO_FSDPW       1|0 moe_fsdp_shard_expert_weights (default 1)
    REPRO_CHUNK       moe_chunk_size; 0 disables chunking (default 8192)
    REPRO_EXPERTS     expert count (default 256)
    REPRO_B/REPRO_S   global batch / seq len (default 512 / 1024)
    REPRO_LR_MULT     scales the production-shaped WSD schedule (default 1.0)
    REPRO_GDR_PLATFORM  pallas | xla (default pallas; xla for CPU smoke)
    REPRO_ATTN        vanilla | splash (default vanilla — the bucket M3 died in)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("EFORMER_DISABLE_FORKIFY", "1")

_eray_path = os.path.abspath(os.path.join("libs", "eray"))  # noqa
if os.path.isdir(_eray_path) and _eray_path not in sys.path:  # noqa
    sys.path.insert(0, _eray_path)  # noqa

_FORWARDED = (
    "REPRO_MESHES",
    "REPRO_STEPS",
    "REPRO_OPT",
    "REPRO_LOSS",
    "REPRO_LAYERS",
    "REPRO_FSDPW",
    "REPRO_CHUNK",
    "REPRO_EXPERTS",
    "REPRO_B",
    "REPRO_S",
    "REPRO_LR_MULT",
    "REPRO_GDR_PLATFORM",
    "REPRO_ATTN",
)


def run_probe():
    """Body shared by the eray entrypoint and local CPU smoke runs."""
    import functools

    _repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    for _p in (_repo_root, os.path.abspath(".")):
        if os.path.isdir(os.path.join(_p, "prismcore")) and _p not in sys.path:
            sys.path.insert(0, _p)

    import easydel as ed  # noqa: F401
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu
    import numpy as np
    import optax
    import spectrax as spx
    from eformer.optimizers import OptimizerFactory, SchedulerConfig
    from ejkernel.modules.operations import GatedDeltaRuleConfig
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from prismcore import build_lloyd_adam_kwargs, build_projection_activation_kwargs

    from easydel.infra.base_module import _parameter_init_sharding_context
    from easydel.infra.loss_utils import LossConfig, LossMetrics
    from easydel.modules.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    from easydel.modules.qwen3_5_moe.qwen3_5_moe_configuration import Qwen3_5MoeTextConfig
    from easydel.trainers.distillation_trainer._fn import distillation_step
    from easydel.trainers.training_utils import (
        constrain_batch_sharding,
        make_assertions_and_get_sizes,
        minibatch_call,
        update_metrics,
        update_state_respectfully,
    )

    AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
    ndev = jax.device_count()
    on_pod = ndev >= 64

    if on_pod:
        MESH_DIMS = {
            "ep1": (1, 4, 64, 1, 4, 1),  # healthy production control
            "ep4": (1, 1, 64, 4, 4, 1),  # ordered production target
        }
    else:
        MESH_DIMS = {  # CPU fake-8-device smoke (ep>1 cannot run on CPU)
            "ep1": (1, 2, 2, 1, 2, 1),
            "ep4": (1, 1, 2, 2, 2, 1),
        }

    MESHES = [m.strip() for m in os.environ.get("REPRO_MESHES", "ep1,ep4").split(",") if m.strip()]
    STEPS = int(os.environ.get("REPRO_STEPS", "6"))
    OPT = os.environ.get("REPRO_OPT", "lloyd")
    LOSS = os.environ.get("REPRO_LOSS", "kd")
    LAYERS = os.environ.get("REPRO_LAYERS", "hybrid")
    FSDPW = os.environ.get("REPRO_FSDPW", "1") == "1"
    CHUNK = int(os.environ.get("REPRO_CHUNK", "8192")) or None
    EXPERTS = int(os.environ.get("REPRO_EXPERTS", "256"))
    B = int(os.environ.get("REPRO_B", "512" if on_pod else "8"))
    S = int(os.environ.get("REPRO_S", "1024" if on_pod else "256"))
    LR_MULT = float(os.environ.get("REPRO_LR_MULT", "1.0"))
    GDR_PLATFORM = os.environ.get("REPRO_GDR_PLATFORM", "pallas" if on_pod else "xla")
    ATTN = os.environ.get("REPRO_ATTN", "vanilla")
    VOCAB = 4096
    H = 256
    PDTYPE = jnp.bfloat16
    TOTAL_STEPS = 15_400  # production schedule length -> mix ~= 0, lr ~= warmup floor at steps 0..6

    is_main = jax.process_index() == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    log(
        f"[repro] devices={ndev} meshes={MESHES} steps={STEPS} opt={OPT} loss={LOSS} layers={LAYERS} "
        f"fsdpW={FSDPW} chunk={CHUNK} experts={EXPERTS} B={B} S={S} lr_mult={LR_MULT} "
        f"gdr={GDR_PLATFORM} attn={ATTN}"
    )

    layer_types = (
        ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        if LAYERS == "hybrid"
        else ["full_attention"] * 4
    )

    gdr_config = GatedDeltaRuleConfig(
        platform=GDR_PLATFORM,
        backend="tpu" if on_pod else "cpu",
        chunk_size=128,
        use_chunked=True,
        use_input_dtype_phase1_outputs=False,
        use_input_dtype_state=False,
    )

    def build_config(dims):
        config = Qwen3_5MoeTextConfig(
            vocab_size=VOCAB,
            hidden_size=H,
            intermediate_size=2 * H,
            num_hidden_layers=len(layer_types),
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=131_072,
            layer_types=list(layer_types),
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts_per_tok=8,
            num_experts=EXPERTS,
            norm_topk_prob=True,
            router_aux_loss_coef=0.001,
            mtp_num_hidden_layers=0,
            mtp_loss_coef=0.0,
            tie_word_embeddings=False,
            sharding_axis_dims=dims,
            sharding_axis_names=AXIS_NAMES,
            scan_layers=False,
        )
        config.add_basic_configurations(
            fsdp_is_ep_bound=False,
            sp_is_ep_bound=False,
            use_ring_of_experts=False,
            moe_chunk_size=CHUNK,
            moe_fsdp_shard_expert_weights=FSDPW,
            attn_mechanism=ATTN,
            attn_dtype="bf16",
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            use_grouped_gdr_prefill=True,
            freq_max_position_embeddings=131_072,
            mask_max_position_embeddings=131_072,
            operation_configs={"gated_delta_rule": gdr_config},
        )
        return config

    def build_model(dims, seed):
        config = build_config(dims)
        with _parameter_init_sharding_context(config):
            return Qwen3_5MoeForCausalLM(config=config, dtype=PDTYPE, param_dtype=PDTYPE, rngs=spx.Rngs(seed))

    def make_wsd_schedule(total_steps: int) -> optax.Schedule:
        """Production launch.py WSD shape, optionally LR-scaled."""
        lr = 1e-4 * LR_MULT
        lr_end = 6e-5 * LR_MULT
        lr_final = 1e-9 * LR_MULT
        warmup_steps, final_decay = 200, 1000
        body_steps = max(total_steps - warmup_steps - final_decay, 1)
        warmup = optax.linear_schedule(1e-8 * LR_MULT, lr, warmup_steps)
        body = optax.linear_schedule(lr, lr_end, body_steps)
        final = optax.linear_schedule(lr_end, lr_final, final_decay)
        return optax.join_schedules([warmup, body, final], [warmup_steps, warmup_steps + body_steps])

    def build_tx():
        sched_cfg = SchedulerConfig(
            scheduler_type="linear",
            steps=TOTAL_STEPS,
            learning_rate=1e-4 * LR_MULT,
            learning_rate_end=6e-5 * LR_MULT,
            warmup_steps=200,
        )
        if OPT == "adamw":
            return OptimizerFactory.create(
                optimizer_type="adamw",
                scheduler_config=sched_cfg,
                clip_grad=1.0,
                weight_decay=0.0,
                custom_scheduler=make_wsd_schedule,
            )
        projection_activation = build_projection_activation_kwargs(
            num_steps=TOTAL_STEPS,
            projection_mix_schedule_type="sigmoid",
            schedule_policies=[
                {"layers": ["embed"], "start": 200, "end": 14_200},
                {"layers": ["mlp"], "start": 200, "end": 4_200},
                {"layers": ["attn"], "start": 5_200, "end": 9_200},
                {"layers": ["lm_head"], "start": 10_200, "end": 14_200},
            ],
        )
        opt_kwargs = build_lloyd_adam_kwargs(
            weight_decay=0.0,
            max_grad_norm=None,
            epsilon=1e-9,
            beta1=0.9,
            beta2=0.95,
            epsilon_root=0.0,
            embed_lr_multiplier=0.05,
            lm_head_lr_multiplier=0.2,
            group_size=128,
            projection_kind="ternary",
            ternary_threshold_mode="adaptive",
            ternary_threshold_val=2.0,
            ternary_scale_mode="kept",
            projection_axis_mode="input_features",
            param_projection_policies=[
                {"patterns": ["embed_tokens", "wte", "wpe"], "match_type": "contains", "axis": "last"},
                {
                    "patterns": ["lm_head", "output_projection", "final_logits"],
                    "match_type": "contains",
                    "axis": "within_token",
                },
            ],
            projection_activation=projection_activation,
            reinit_z_on_load=False,
            reinit_z_step=0,
        )
        opt_kwargs.pop("weight_decay", None)
        return OptimizerFactory.create(
            optimizer_type="lloyd_adam_mirror",
            scheduler_config=sched_cfg,
            clip_grad=1.0,
            weight_decay=0.0,
            custom_scheduler=make_wsd_schedule,
            **opt_kwargs,
        )

    loss_config = LossConfig(break_on_nan=True)

    def make_batch_fn(mesh):
        sharding2d = NamedSharding(mesh, P(("dp", "fsdp"), None))

        def gen(key):
            ids = jax.random.randint(key, (B, S), 3, VOCAB, dtype=jnp.int32)
            row = jnp.arange(S, dtype=jnp.int32)[None, :]
            has_tail = (jnp.arange(B, dtype=jnp.int32) % 8 == 7)[:, None]
            tail_start = jnp.where(has_tail, S - S // 4, S)
            amask = (row < tail_start).astype(jnp.int32)
            ids = ids * amask
            labels = jnp.where(amask == 1, ids, -100)
            return {"input_ids": ids, "attention_mask": amask, "labels": labels}

        return jax.jit(gen, out_shardings={"input_ids": sharding2d, "attention_mask": sharding2d, "labels": sharding2d})

    def ce_step(student_state, batch, scheduler):
        """Plain masked-CE step through the same minibatch/update plumbing (no teacher)."""
        _bs, minibatch_size, pspec = make_assertions_and_get_sizes(
            batch=batch, gradient_accumulation_steps=1, batch_partition_spec=None
        )
        batch = dict(constrain_batch_sharding(batch, pspec, mesh=student_state.model.mesh))

        def loss_fn(tree, minibatch):
            module = student_state.merge(tree)
            out = module(input_ids=minibatch["input_ids"], attention_mask=minibatch["attention_mask"])
            logits = out.logits[:, :-1].astype(jnp.float32)
            labels = minibatch["labels"][:, 1:]
            mask = (labels != -100).astype(jnp.float32)
            safe = jnp.where(labels == -100, 0, labels)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, safe)
            loss = jnp.sum(ce * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            return loss, LossMetrics(loss=loss, other_metrics={"kl_loss": loss})

        gradients, metrics = minibatch_call(
            state=student_state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        student_state = update_state_respectfully(
            state=student_state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics, learning_rate_fn=scheduler, step=student_state.step, gradients=gradients
            ),
        )
        return student_state, metrics

    def finiteness_report(tree):
        """(all_finite, [paths of non-finite float leaves]) for a pytree; host-side."""
        flat, _ = jtu.tree_flatten_with_path(tree)
        checks = []
        for path, leaf in flat:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                checks.append((jtu.keystr(path), leaf))
        if not checks:
            return True, []
        finite = jax.jit(lambda ls: [jnp.all(jnp.isfinite(x)) for x in ls])([leaf for _, leaf in checks])
        finite = jax.device_get(finite)
        bad = [name for (name, _), ok in zip(checks, finite, strict=True) if not bool(ok)]
        return (len(bad) == 0), bad

    def run_mesh(mesh_key):
        dims = MESH_DIMS[mesh_key]
        log(f"[repro:{mesh_key}] dims={dims} building student/teacher ...")
        student = build_model(dims, seed=0)
        mesh = student.config.mesh
        tx, scheduler = build_tx()
        with mesh:
            student_state = student.to_state().init_tx(tx)
            teacher_state = None
            if LOSS == "kd":
                teacher = build_model(dims, seed=1)
                teacher.eval()
                teacher_state = teacher.to_state()

            gen = make_batch_fn(mesh.jax_mesh if hasattr(mesh, "jax_mesh") else mesh)

            if LOSS == "kd":
                step_fn = functools.partial(
                    distillation_step,
                    loss_config=loss_config,
                    learning_rate_fn=scheduler,
                    partition_spec=None,
                    gradient_accumulation_steps=1,
                    is_training=True,
                    temperature=1.0,
                    alpha=1.0,
                    logits_chunk_size=256,
                    checkpoint_kl_loss=True,
                    teacher_matched_compilation=True,
                )
                jstep = jax.jit(lambda st, b, tst: step_fn(st, b, tst), donate_argnums=(0,))
            else:
                jstep = jax.jit(lambda st, b: ce_step(st, b, scheduler), donate_argnums=(0,))

            for i in range(STEPS):
                batch = gen(jax.random.PRNGKey(1000 + i))
                if LOSS == "kd":
                    student_state, metrics = jstep(student_state, batch, teacher_state)
                else:
                    student_state, metrics = jstep(student_state, batch)

                loss = float(jax.device_get(metrics.loss))
                kl = metrics.other_metrics.get("kl_loss") if metrics.other_metrics else None
                kl = float(jax.device_get(kl)) if kl is not None else float("nan")
                gnorm_tree = metrics.grad_norms
                bad_grads = []
                if gnorm_tree is not None:
                    flat, _ = jtu.tree_flatten_with_path(gnorm_tree)
                    vals = jax.device_get([leaf for _, leaf in flat])
                    bad_grads = [jtu.keystr(p) for (p, _), v in zip(flat, vals, strict=True) if not np.isfinite(v)]
                max_gn = float(jax.device_get(metrics.max_grad_norm)) if metrics.max_grad_norm is not None else -1.0
                params_ok, bad_params = finiteness_report(student_state.graphstate)
                opt_ok, bad_opt = finiteness_report(student_state.opt_state)
                log(
                    f"[repro:{mesh_key}] step {i}: loss={loss:.6e} kl={kl:.6e} max_grad_norm={max_gn:.3e} "
                    f"bad_grad_leaves={len(bad_grads)} params_finite={params_ok} opt_finite={opt_ok}"
                )
                for name in bad_grads[:12]:
                    log(f"[repro:{mesh_key}]   NONFINITE grad: {name}")
                for name in bad_params[:12]:
                    log(f"[repro:{mesh_key}]   NONFINITE param: {name}")
                for name in bad_opt[:12]:
                    log(f"[repro:{mesh_key}]   NONFINITE opt: {name}")
                if bad_grads or not params_ok or not opt_ok:
                    log(f"[repro:{mesh_key}] NONFINITE DETECTED at step {i} -- stopping this mesh")
                    return False
        log(f"[repro:{mesh_key}] all {STEPS} steps finite")
        return True

    results = {}
    for mesh_key in MESHES:
        try:
            results[mesh_key] = run_mesh(mesh_key)
        except Exception as e:  # noqa: BLE001
            import traceback

            log(f"[repro:{mesh_key}] EXC {type(e).__name__}: {e}\n{traceback.format_exc()}")
            results[mesh_key] = None
    log(f"[repro] RESULTS: {results}")


if os.environ.get("REPRO_LOCAL_CPU") != "1":
    import ray  # noqa: E402
    from eray import TpuAcceleratorConfig, execute, print_remote_raise  # noqa: E402
    from ray.runtime_env import RuntimeEnv  # noqa: E402

    RUNTIME_WORKING_DIR = os.path.abspath(".")

    if os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR"):
        ray.init(address="auto")
    else:
        ray.init(address="auto", runtime_env=RuntimeEnv(working_dir=RUNTIME_WORKING_DIR))

    @execute(
        TpuAcceleratorConfig(
            tpu_version="v5p-2048",
            pod_count=1,
            execution_env=RuntimeEnv(
                env_vars={
                    "EFORMER_DISABLE_FORKIFY": "1",
                    "EFORMER_SUBPROCESS_TIMEOUT_S": "1000000",
                    "EASYDEL_TARGETED_TPU_GENERATION": "v5p",
                    "PYTHONPATH": "libs/eray:libs/easydel:libs/ejkernel:libs/eformer:libs/spectrax:.",
                    "JAX_COMPILATION_CACHE_DIR": "/home/erfan/jax-compile-cache",
                    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "10",
                    **{k: os.environ[k] for k in _FORWARDED if k in os.environ},
                },
            ),
        )
    )
    @ray.remote
    def main():
        """Run the tiny distillation training loop on every requested mesh."""
        import os
        import sys

        for _lib in ("eray", "easydel", "ejkernel", "eformer", "spectrax"):
            _p = os.path.abspath(os.path.join("libs", _lib))
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)
        _root = os.path.abspath(".")
        if _root not in sys.path:
            sys.path.insert(0, _root)

        run_probe()
        return "ok"


if __name__ == "__main__":
    if os.environ.get("REPRO_LOCAL_CPU") == "1":
        run_probe()
    else:
        result = main()
        if hasattr(result, "error") and result.error is not None:
            try:
                print_remote_raise(result.error)
            except Exception:  # noqa: BLE001
                print(f"Raw error: {result.error}")
                raise result.error from None
