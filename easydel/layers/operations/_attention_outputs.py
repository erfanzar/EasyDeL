from __future__ import annotations

from eformer.pytree import auto_pytree
from jax import Array
from jaxtyping import Float

from ..caching import RaggedPagesCacheView, TransformerCacheView
from ._operation_impl import OperationOutput


@auto_pytree
class AttentionOutput(OperationOutput):
    """
    This dataclass encapsulates the results computation
    """

    attention_weights: Float[Array, "batch num_heads seq_len seq_len"] | None = None
    attention_outputs: Float[Array, "batch seq_len num_heads head_dim"] | None = None
    cache_view: TransformerCacheView | RaggedPagesCacheView | None = None
