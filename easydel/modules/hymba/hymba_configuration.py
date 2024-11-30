import math

from git import Optional

from easydel.etils.etils import EasyDeLGradientCheckPointers
from easydel.modules.factory import register_config
from easydel.modules.modeling_utils import EasyDeLBaseConfig


@register_config("hymba")
class HymbaConfig(EasyDeLBaseConfig):
	model_type = "hymba"
	keys_to_ignore_at_inference = ["past_key_values"]

	def __init__(
		self,
		vocab_size=65536,
		tie_word_embeddings=False,
		hidden_size=4096,
		intermediate_size=14336,
		num_hidden_layers=32,
		num_attention_heads=32,
		num_key_value_heads=8,
		hidden_act="silu",
		initializer_range=0.02,
		rms_norm_eps=1e-6,
		use_cache=True,
		calc_logits_for_entire_prompt=False,
		output_router_logits=False,
		router_aux_loss_coef=0.001,
		pad_token_id=0,
		bos_token_id=1,
		eos_token_id=2,
		sliding_window=None,
		max_position_embeddings=262144,
		orig_max_position_embeddings=None,
		attention_dropout=0.0,
		num_experts_per_tok=2,
		num_experts=16,
		use_mamba_kernels=True,
		mamba_d_state=16,
		mamba_d_conv=4,
		mamba_expand=2,
		mamba_dt_rank="auto",
		mamba_conv_bias=True,
		mamba_proj_bias=False,
		mamba_inner_layernorms=True,
		kv_reuse_every_i_layer=-1,
		kv_reuse_group=None,
		kv_weight_reuse=False,
		global_attn_idx=None,
		num_mamba=1,
		attn_implementation_new="sdpa",
		rope_type=None,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.tie_word_embeddings = tie_word_embeddings
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.sliding_window = sliding_window
		self.max_position_embeddings = max_position_embeddings
		self.orig_max_position_embeddings = orig_max_position_embeddings
		self.attention_dropout = attention_dropout

		if num_key_value_heads is None:
			num_key_value_heads = num_attention_heads

		self.num_key_value_heads = num_key_value_heads
		self.hidden_act = hidden_act
		self.initializer_range = initializer_range
		self.rms_norm_eps = rms_norm_eps

		self.use_cache = use_cache
		self.calc_logits_for_entire_prompt = calc_logits_for_entire_prompt
		self.output_router_logits = output_router_logits
		self.router_aux_loss_coef = router_aux_loss_coef

		self.num_experts_per_tok = num_experts_per_tok
		self.num_experts = num_experts

		self.use_mamba_kernels = use_mamba_kernels
		self.mamba_d_state = mamba_d_state
		self.mamba_d_conv = mamba_d_conv
		self.mamba_expand = mamba_expand
		self.mamba_dt_rank = (
			math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
		)
		self.mamba_conv_bias = mamba_conv_bias
		self.mamba_proj_bias = mamba_proj_bias
		self.mamba_inner_layernorms = mamba_inner_layernorms

		self.attn_hidden_size = kwargs.pop("attn_hidden_size", -1)
		self.kq_head_dim = kwargs.pop("kq_head_dim", -1)
		self.v_head_dim = kwargs.pop("v_head_dim", -1)
		self.kq_norm = kwargs.pop("kq_norm", None)
		self.rope = kwargs.pop("rope", False)
		self.rope_theta = kwargs.pop("rope_theta", 10000.0)
		self.num_memory_tokens = kwargs.pop("num_memory_tokens", 0)
		self.memory_tokens_interspersed_every = kwargs.pop(
			"memory_tokens_interspersed_every", 0
		)

		self.kv_reuse_every_i_layer = kv_reuse_every_i_layer
		self.kv_reuse_group = kv_reuse_group
		self.kv_weight_reuse = kv_weight_reuse

		self.global_attn_idx = global_attn_idx

		self.num_mamba = num_mamba

		self.attn_implementation_new = attn_implementation_new

		self.rope_type = rope_type

		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			tie_word_embeddings=tie_word_embeddings,
			**kwargs,
		)

	def add_jax_args(
		self,
		tie_word_embeddings: bool = False,
		gradient_checkpointing: str = EasyDeLGradientCheckPointers.NONE,
		bits: Optional[int] = None,
		**kwargs,
	):
		"""The add_jax_args function adds the following arguments to the Transformer class:

		Args:
		    self: Refer to the current object
		    tie_word_embeddings: bool: Tie the word embeddings to the
		        decoder
		    gradient_checkpointing: str: Control the amount of memory
		        used by jax
		    bits: Optional[int]: Determine the number of bits used in
		        the quantization
		"""
		self.tie_word_embeddings = tie_word_embeddings
		self.gradient_checkpointing = gradient_checkpointing
		self.bits = bits

	@staticmethod
	def get_weight_decay_exclusions():
		return tuple()

	@staticmethod
	def rng_keys():
		return "params", "dropout"

	@property
	def granted_freq_max_position_embedding(self) -> int:
		return getattr(
			self,
			"freq_max_position_embeddings",
			self.max_position_embeddings,
		)

	@property
	def granted_mask_max_position_embedding(self) -> int:
		return getattr(
			self,
			"mask_max_position_embeddings",
			self.max_position_embeddings,
		)
