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

"""End-to-end smoke test for Qwen3-8B with full pipeline parallelism (pp=4).

Goal:
  1. Build Qwen3-8B with random init under a 4-stage pipeline mesh on 4 TPUs.
  2. Inspect parameter shardings -- every PP-tagged leaf must live on exactly
     one physical chip (the chip that owns its stage).
  3. Initialise an Adam optimiser via ``state.init_tx`` and inspect the
     resulting ``opt_state`` shardings -- the per-stage placement of params
     must be mirrored leaf-for-leaf on the optimiser slots (mu, nu, ...).
  4. Run a couple of fwd/bwd/update steps on a synthetic batch.  Assert the
     loss is finite and that the post-step sharding signature is unchanged.

The script does NOT download the 16 GB safetensors -- it only fetches the
~10 KB ``config.json`` from HuggingFace, then random-initialises params at
bf16.  That keeps total HBM at ~16 GB for params + ~16 GB for Adam state,
well within 4 x 96 GB on v5p.
"""

from __future__ import annotations

import collections
import sys
import time

# easydel must be imported BEFORE optax: optax 0.2.7 calls
# ``jax.config.update('jax_pmap_shmap_merge', False)`` at import time, which
# raises on jax 0.10+ unless easydel's compat layer has already trimmed the
# unrecognised config option.
import easydel as ed
import optax  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import spectrax as spx  # noqa: E402

from easydel.infra.base_state import EasyDeLState  # noqa: E402

import os

MODEL_NAME = os.environ.get("EASYDEL_PP_TEST_MODEL", "Qwen/Qwen3-8B")


# --------------------------------------------------------------------------- helpers


def device_signature(named_sharding) -> tuple[int, ...]:
    """Return the sorted set of physical device IDs covered by a NamedSharding."""
    if named_sharding is None:
        return ()
    devices = named_sharding.mesh.devices.flatten()
    return tuple(sorted(int(d.id) for d in devices))


def collect_leaf_shardings(tree) -> list:
    """Return ``[(path, sharding_or_None)]`` for every leaf in *tree*."""
    pairs = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        sh = getattr(leaf, "sharding", None) if hasattr(leaf, "shape") else None
        pairs.append((jax.tree_util.keystr(path), sh))
    return pairs


def summarize_by_stage(label: str, pairs):
    """Bucket leaves by physical-device set and print a per-bucket count."""
    counter: collections.Counter = collections.Counter()
    fully_replicated = 0
    unsharded = 0
    for _path, sh in pairs:
        if sh is None:
            unsharded += 1
            continue
        sig = device_signature(sh)
        if len(sig) == len(jax.devices()):
            fully_replicated += 1
        counter[sig] += 1
    print(f"\n[{label}] {len(pairs)} leaves total")
    print(f"  unsharded (no .sharding): {unsharded}")
    print(f"  fully replicated across all {len(jax.devices())} chips: {fully_replicated}")
    print("  by device-set:")
    for sig, n in sorted(counter.items(), key=lambda x: (len(x[0]), x[0])):
        marker = " (REPLICATED)" if len(sig) == len(jax.devices()) else ""
        print(f"    devices={sig} -> {n} leaves{marker}")
    return counter


def assert_opt_state_matches_params(param_pairs, opt_pairs):
    """Per-leaf check: every opt-state slot lives on the same device set as its
    corresponding parameter (mu/nu/etc.).

    Adam ``opt_state`` is a tuple ``(scale_state, ScaleByAdamState(count, mu, nu))``.
    The mu/nu pytrees mirror the param tree exactly, so we can correlate by
    matching the param suffix path against any opt-state path that ends with
    the same suffix.
    """
    param_by_path = {p: device_signature(s) for p, s in param_pairs if s is not None}
    mismatches = []
    matched = 0
    for opt_path, opt_sh in opt_pairs:
        if opt_sh is None:
            continue
        opt_sig = device_signature(opt_sh)
        # Find the param whose path is a suffix of this opt path.
        candidate_param = None
        for p_path, p_sig in param_by_path.items():
            if opt_path.endswith(p_path):
                candidate_param = (p_path, p_sig)
                break
        if candidate_param is None:
            continue
        matched += 1
        p_path, p_sig = candidate_param
        if p_sig != opt_sig:
            mismatches.append((opt_path, opt_sig, p_path, p_sig))
    print(f"\n[opt-state vs param placement] matched {matched} opt slots to params")
    if mismatches:
        print(f"  MISMATCHES: {len(mismatches)}")
        for opt_path, opt_sig, p_path, p_sig in mismatches[:10]:
            print(f"    OPT  {opt_path}  on devs {opt_sig}")
            print(f"    PARAM{p_path}  on devs {p_sig}")
        return False
    print("  all checked opt-state slots co-located with their params [OK]")
    return True


# --------------------------------------------------------------------------- main


def main() -> int:
    devices = jax.devices()
    n = len(devices)
    print(f"jax devices: {n} x {devices[0].device_kind}")
    if n < 4:
        print("ERROR: need at least 4 chips for pp=4")
        return 2

    # Step 1 -- pull config (no weights), override sharding to pp=4.
    print(f"\nLoading config for {MODEL_NAME} (no weights)...")
    t0 = time.perf_counter()
    config = ed.AutoEasyDeLConfig.from_pretrained(
        MODEL_NAME,
        sharding_axis_dims=(4, 1, 1, 1, 1, 1),
        sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
    )
    print(f"  config loaded in {time.perf_counter() - t0:.2f}s")
    print(f"  num_hidden_layers = {config.num_hidden_layers}")
    print(f"  hidden_size = {config.hidden_size}")
    print(f"  vocab_size = {config.vocab_size}")
    print(f"  mesh shape = {dict(config.mesh.shape)}")
    print(f"  mesh axis_names = {config.mesh.axis_names}")

    # Step 2 -- random-init the model under the pp=4 mesh.  Stage assignment
    # is recorded into variable metadata via the modelling code's
    # ``_assign_layer_stage`` context.
    print("\nRandom-initialising Qwen3-8B at bf16...")
    t0 = time.perf_counter()
    model = ed.AutoEasyDeLModelForCausalLM.from_config(
        config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=spx.Rngs(0),
    )
    print(f"  model built in {time.perf_counter() - t0:.2f}s")

    # Step 3 -- wrap in EasyDeLState (no optimiser yet).
    print("\nWrapping in EasyDeLState...")
    t0 = time.perf_counter()
    state = EasyDeLState.create(model=model, init_opt_state=False)
    print(f"  state ready in {time.perf_counter() - t0:.2f}s")

    # Step 4 -- inspect param shardings.
    print("\n=== parameter sharding ===")
    param_pairs = collect_leaf_shardings(state.graphstate)
    summarize_by_stage("params", param_pairs)

    # Step 5 -- init Adam optimiser (this is where opt-state sharding is wired).
    print("\nInitialising AdamW optimiser via state.init_tx ...")
    t0 = time.perf_counter()
    tx = optax.adamw(learning_rate=1e-5, weight_decay=0.0)
    state = state.init_tx(tx)
    print(f"  optimiser initialised in {time.perf_counter() - t0:.2f}s")

    # Step 6 -- inspect opt-state shardings + assert they mirror params.
    print("\n=== opt-state sharding (after init_tx) ===")
    opt_pairs = collect_leaf_shardings(state.opt_state)
    summarize_by_stage("opt_state", opt_pairs)
    co_located = assert_opt_state_matches_params(param_pairs, opt_pairs)
    if not co_located:
        return 3

    # Step 7 -- record the placement signature so we can assert it doesn't drift.
    pre_step_signature = {
        path: device_signature(sh) for path, sh in opt_pairs if sh is not None
    }

    # Step 8 -- run a few train steps on synthetic data.
    print("\n=== training steps on synthetic data ===")
    seq_len = 64
    batch_size = 4
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(
        rng, (batch_size, seq_len), 0, config.vocab_size, dtype=jnp.int32
    )
    attention_mask = jnp.ones_like(input_ids)
    labels = jnp.roll(input_ids, shift=-1, axis=1)
    labels = labels.at[:, -1].set(-100)

    def step_fn(state_arg, input_ids, attention_mask, labels):
        # All work happens INSIDE spx.jit so MPMD can schedule per-stage forwards
        # and place per-stage gradients on the right chips.  Calling the model
        # eagerly with multi-submesh params raises "incompatible devices".
        def loss_fn(graphstate):
            full_state = graphstate.merge(state_arg.graphother, copy=True)
            m = spx.bind(state_arg.graphdef, full_state)
            out = m(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.astype(jnp.float32)
            valid = labels != -100
            safe_labels = jnp.where(valid, labels, 0)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            per_tok = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
            per_tok = jnp.where(valid, per_tok, 0.0)
            return -per_tok.sum() / jnp.maximum(valid.sum(), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(state_arg.graphstate)
        new_state = state_arg.apply_gradients(grads=grads)
        return new_state, loss

    jax_mesh = config.mesh.jax_mesh
    replicated = jax.sharding.NamedSharding(jax_mesh, jax.sharding.PartitionSpec())
    batch_in_shardings = (replicated, replicated, replicated)
    # Passing the SpxMesh (with mpmd_axis="pp") makes spx.jit route to sxjit/MPMD.
    step_jit = spx.jit(
        step_fn,
        mesh=config.mesh,
        in_shardings=(None, *batch_in_shardings),
        out_shardings=(None, replicated),
    )

    for step_i in range(2):
        print(f"\n  step {step_i}: tracing+compiling fwd/bwd/update ...")
        t0 = time.perf_counter()
        state, loss = step_jit(state, input_ids, attention_mask, labels)
        loss = float(loss)
        print(f"    loss = {loss:.4f}  (took {time.perf_counter() - t0:.2f}s)")
        if not jnp.isfinite(loss):
            print("    ERROR: non-finite loss")
            return 4

    # Step 9 -- re-check opt-state sharding signature is stable across updates.
    print("\n=== opt-state sharding (after 2 train steps) ===")
    opt_pairs_after = collect_leaf_shardings(state.opt_state)
    summarize_by_stage("opt_state (post-step)", opt_pairs_after)
    post_step_signature = {
        path: device_signature(sh) for path, sh in opt_pairs_after if sh is not None
    }
    drifted = [p for p in pre_step_signature
               if pre_step_signature[p] != post_step_signature.get(p, ())]
    if drifted:
        print(f"\nERROR: {len(drifted)} opt-state leaves drifted to a different device set:")
        for p in drifted[:10]:
            print(f"  {p}: pre={pre_step_signature[p]}  post={post_step_signature.get(p, ())}")
        return 5
    print("  opt-state placement preserved across train steps [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
