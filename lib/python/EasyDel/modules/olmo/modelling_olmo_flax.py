import math
from functools import partial
from typing import Optional, Tuple, Union

from einops import einops
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec
import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput, FlaxSequenceClassifierOutput
# EasyDel.modules
from ..flax_modelling_utils import (
    with_sharding_constraint,
    get_gradient_checkpoint_policy,
    repeat_kv_bnsh,
    apply_rotary_pos_emb,
    precompute_freq_cis,
    get_dot_general_by_bits
)
from ..easy_attention import AttentionOutput, EasyAttention

from ..easydel_modelling_utils import EasyDelFlaxPretrainedModel
import chex
from .olmo_configuration import OLMoConfig
