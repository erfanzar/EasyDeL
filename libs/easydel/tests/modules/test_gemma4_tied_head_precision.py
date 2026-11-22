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

"""Array-free tracing checks for Gemma4's public tied projection contract."""

from contextlib import nullcontext
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor, _project_stage_lm_head
from easydel.layers import Embed
from easydel.layers.norms import promote_dtype
from easydel.modules.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from jax import numpy as jnp

from .test_lm_logit_contract import _head

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("entry_point", ["apply_lm_head", "make_lm_head_fn"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "precision",
    [None, jax.lax.Precision.HIGHEST, (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST)],
    ids=["ambient", "highest", "per-operand"],
)
def test_tied_projection_traces_requested_precision_and_embedding_dtype(entry_point, dtype, precision):
    # Only abstract values enter the projection: make_jaxpr neither compiles nor
    # executes array computations. Check the actual public output program rather
    # than the implementation's choice of helper/embedding method.
    def project(hidden, weight):
        head = _head(Gemma4ForCausalLM, tied=True)
        object.__setattr__(head, "precision", precision)
        embedding = SimpleNamespace(weight=SimpleNamespace(value=weight), dtype=dtype, promote_dtype=promote_dtype)
        object.__setattr__(head, "get_embedding", lambda: embedding)
        if entry_point == "apply_lm_head":
            return head.apply_lm_head(hidden)
        return head.make_lm_head_fn()(hidden)

    with jax.default_matmul_precision("default"):
        program = jax.make_jaxpr(project)(
            jax.ShapeDtypeStruct((2, 3), np.float32), jax.ShapeDtypeStruct((5, 3), np.float32)
        )
    dots = [eqn for eqn in program.jaxpr.eqns if eqn.primitive.name == "dot_general"]
    assert len(dots) == 1
    expected = precision if isinstance(precision, tuple) else (precision or jax.lax.Precision.DEFAULT,) * 2
    assert dots[0].params["precision"] == expected
    assert program.out_avals[0].shape == (2, 5)
    assert program.out_avals[0].dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("soft_cap", [None, 2.0], ids=["no-soft-cap", "soft-cap"])
@pytest.mark.parametrize(
    "precision",
    [None, jax.lax.Precision.HIGHEST, (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST)],
    ids=["ambient", "highest", "per-operand"],
)
def test_split_export_preserves_owning_head_precision_and_dtype(dtype, soft_cap, precision, monkeypatch):
    # Run the real model-to-head refresh/export/bind path with abstract parameters.
    # Only the backbone binding and mesh placement are replaced; no arrays are
    # constructed, compiled or executed by this cpu_ok test.
    model_graph = object()
    original_bind = spx.bind
    bound_model = [None]
    monkeypatch.setattr(
        spx, "bind", lambda graph, state: bound_model[0] if graph is model_graph else original_bind(graph, state)
    )

    def project(hidden, weight):
        embedding = Embed(
            5,
            3,
            dtype=dtype,
            param_dtype=jnp.float32,
            embedding_init=lambda key, shape, dtype: weight,
            rngs=SimpleNamespace(parameters=None),
        )
        head = _head(Gemma4ForCausalLM, tied=True, soft_cap=soft_cap, clip=0.125)
        object.__setattr__(head, "precision", precision)
        object.__setattr__(head, "get_embedding", lambda: embedding)
        bound_model[0] = head
        executor = ModelStepExecutor.__new__(ModelStepExecutor)
        executor.model = SimpleNamespace(mesh=nullcontext())
        executor._uses_mpmd_mesh = lambda: True
        executor._uses_tied_lm_head = lambda model: True
        executor._lm_head_uses_tied_projection = True
        executor._lm_head_projection_is_complete = True
        executor._lm_head_stage_mesh = lambda: None
        executor._stage_local_lm_head_state = lambda state: state
        executor.refresh_lm_head_state(
            graphdef=model_graph,
            graphstate=SimpleNamespace(merge=lambda other, copy: None),
            graphother=None,
        )
        return _project_stage_lm_head(
            executor._lm_head_graphdef,
            executor._lm_head_state,
            hidden,
            uses_tied_projection=(
                executor._lm_head_uses_tied_projection and not executor._lm_head_projection_is_complete
            ),
            postprocess=executor._lm_head_postprocess,
            scope="test/exported_gemma4_head",
        )

    with jax.default_matmul_precision("default"):
        program = jax.make_jaxpr(project)(
            jax.ShapeDtypeStruct((2, 3), np.float32), jax.ShapeDtypeStruct((5, 3), np.float32)
        )
    dots = [eqn for eqn in program.jaxpr.eqns if eqn.primitive.name == "dot_general"]
    assert len(dots) == 1
    expected = precision if isinstance(precision, tuple) else (precision or jax.lax.Precision.DEFAULT,) * 2
    assert dots[0].params["precision"] == expected
    assert program.out_avals[0].shape == (2, 5)
    assert program.out_avals[0].dtype == jnp.dtype(dtype)
    # The complete head owns the soft-cap, so the executor must not duplicate it
    # or accidentally infer the unrelated base clipping configured above.
    assert sum(eqn.primitive.name == "tanh" for eqn in program.jaxpr.eqns) == (soft_cap is not None)
    assert not any(eqn.primitive.name == "clamp" for eqn in program.jaxpr.eqns)
