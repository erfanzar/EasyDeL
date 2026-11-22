"""Shape-only coverage for explicit precision in Gemma4's tied text heads.

Tracing uses abstract arguments exclusively: no model initialization, array
allocation, numerical execution, compilation, or accelerator access is needed.
"""

from types import SimpleNamespace

import jax
import pytest
from easydel.layers.embeddings import Embed
from easydel.modules.gemma4 import Gemma4ForCausalLM
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("head", ["direct", "callable"])
@pytest.mark.parametrize("sequence_length", [1, 3])
@pytest.mark.parametrize("softcap", [None, 2.0])
def test_explicit_gemma4_tied_head_precision_overrides_low_ambient(head, sequence_length, softcap):
    """Full and callable/chunk-sized heads must honor constructor precision.

    Exercise the real embedding constructor and native tied projection, and both
    real model projection methods. Only the unrelated decoder/config container
    is replaced: the projection inputs and embedding table are abstract tracers.
    """

    def project(hidden_states, table):
        embedding = Embed(
            num_embeddings=64,
            features=128,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            embedding_init=lambda *_: table,
            rngs=SimpleNamespace(parameters=None),
        )
        model = SimpleNamespace(
            config=SimpleNamespace(tie_word_embeddings=True, final_logit_softcapping=softcap),
            precision=jax.lax.Precision.HIGHEST,
            get_embedding=lambda: embedding,
        )
        model.get_lm_logit_postprocess = lambda: Gemma4ForCausalLM.get_lm_logit_postprocess(model)
        model._make_tied_lm_head_fn = lambda: Gemma4ForCausalLM._make_tied_lm_head_fn(model)
        if head == "direct":
            return Gemma4ForCausalLM.apply_lm_head(model, hidden_states)
        return Gemma4ForCausalLM.make_lm_head_fn(model)(hidden_states)

    previous_precision = jax.config.jax_default_matmul_precision
    with jax.default_matmul_precision("bfloat16"):
        traced = jax.make_jaxpr(project)(
            jax.ShapeDtypeStruct((4, sequence_length, 128), jnp.float32),
            jax.ShapeDtypeStruct((64, 128), jnp.float32),
        )
    assert jax.config.jax_default_matmul_precision == previous_precision
    assert traced.out_avals[0].shape == (4, sequence_length, 64)
    assert traced.out_avals[0].dtype == jnp.dtype(jnp.float32)
    dots = [eqn for eqn in traced.jaxpr.eqns if eqn.primitive.name == "dot_general"]
    assert len(dots) == 1
    assert dots[0].params["precision"] == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
