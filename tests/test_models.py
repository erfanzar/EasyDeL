import gc
import os

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import sys
import time
import unittest

import jax

# jax.config.update("jax_platform_name", "cpu")  # CPU Test !
import jax.extend
import jax.random
from fjformer import make_shard_and_gather_fns, match_partition_rules

sys.path.append(
	os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		"..",
	)
)
import copy
from typing import Dict, Optional, Union

import jax
import numpy as np
import torch
import transformers
from fjformer.functions import cross_entropy_loss_and_accuracy
from flax import nnx as nn
from jax import numpy as jnp
from tabulate import tabulate

import easydel as ed
from easydel.etils.etils import (
	AVAILABLE_ATTENTION_MECHANISMS,
	DEFAULT_ATTENTION_MECHANISM,
	EasyDeLGradientCheckPointers,
)

torch.manual_seed(42)


class EasyModelsTest(unittest.TestCase):
	def setUp(self) -> None:
		self.batch_size: int = 2
		self.vocab_size: int = 32000
		self.hidden_size: int = 128
		self.intermediate_size: int = 256
		self.num_hidden_layers: int = 6
		self.num_attention_heads: int = 8
		self.num_key_value_heads: Optional[int] = 4
		self.num_experts_per_tok = 4
		self.num_experts = 8
		self.num_local_experts = self.num_experts
		self.rms_norm_eps: float = 1e-6
		self.layer_norm_eps = self.rms_norm_eps
		self.initializer_range: float = 0.02
		self.use_cache: bool = True
		self.bos_token_id: int = 0
		self.eos_token_id: int = 1
		self.resid_pdrop: float = 0.0
		self.embd_pdrop: float = 0.0
		self.attention_dropout: float = 0.0
		self.rope_theta: float = 10000.0
		self.attention_bias: bool = False
		self.tie_word_embeddings: bool = False
		self.gradient_checkpointing = EasyDeLGradientCheckPointers.NONE
		self.fcm_min_ratio: float = -1
		self.fcm_max_ratio: float = -1
		self.rope_scaling: Optional[Dict[str, Union[str, float]]] = None
		self.use_scan_mlp: bool = False
		self.bits: Optional[int] = None
		self.hidden_act: str = "silu"
		self.pretraining_tp: int = 1
		self.scan_layers: bool = False
		self.shard_attention_computation: bool = True
		self.rotary_dim = 32
		self.dtype: jax.numpy.dtype = jnp.float32
		self.precision = jax.lax.Precision("highest")
		self.attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = DEFAULT_ATTENTION_MECHANISM
		self.blocksize_k: int = 64
		self.blocksize_q: int = 128
		self.sequence_length = 64
		self.scan_mlp_chunk_size = self.sequence_length // 2
		self.head_dim = self.hidden_size // self.num_attention_heads
		self.use_parallel_residual = True
		self.qk_layernorm = True
		self.max_position_embeddings: int = self.sequence_length
		self.use_sharding_constraint = False
		self.header_config = None
		self.pad_token_id = None
		self.rope_scaling = None
		self.attn_dtype = (
			jnp.float16 if jax.extend.backend.get_backend().platform == "gpu" else jnp.float32
		)
		self.platform = "triton"

	def create_test_for_models(self, module_name: str, hf_module_class, task):
		(
			module_config,
			module_class,
			transform_function,
		) = ed.get_modules_by_type(module_name, task)
		if self.header_config is None:
			config = module_config(
				num_experts_per_tok=self.num_experts_per_tok,
				num_experts=self.num_experts,
				num_local_experts=self.num_local_experts,
				vocab_size=self.vocab_size,
				hidden_size=self.hidden_size,
				num_attention_heads=self.num_attention_heads,
				num_hidden_layers=self.num_hidden_layers,
				num_layers=self.num_hidden_layers,
				gradient_checkpointing=self.gradient_checkpointing,
				max_position_embeddings=self.max_position_embeddings,
				num_key_value_heads=self.num_key_value_heads,
				scan_mlp_chunk_size=self.scan_mlp_chunk_size,
				intermediate_size=self.intermediate_size,
				rotary_dim=self.rotary_dim,
				rms_norm_eps=self.rms_norm_eps,
				layer_norm_eps=self.layer_norm_eps,
				head_dim=self.head_dim,
				new_decoder_architecture=True,
				num_kv_heads=self.num_key_value_heads,
				multi_query=True,
				num_ln_in_parallel_attn=1,
				parallel_attn=True,
				use_parallel_residual=self.use_parallel_residual,
				qk_layernorm=self.qk_layernorm,
				rope_scaling=self.rope_scaling,
				platform=self.platform,
			)
		else:
			config = self.header_config

		config.axis_dims = (1, 1, 1, -1)
		config.pad_token_id = 0
		hf_model = hf_module_class(config=copy.deepcopy(config))
		hf_model.eval()
		hf_model = hf_model.float()
		params = transform_function(
			state_dict=hf_model.state_dict(),
			device=jax.devices("cpu")[0],
			remove_state_dict=True,
		)

		config.add_jax_args()
		config.add_basic_configurations(
			shard_attention_computation=self.shard_attention_computation,
			use_sharding_constraint=self.use_sharding_constraint,
			scan_mlp_chunk_size=self.scan_mlp_chunk_size,
		)
		mesh = config.mesh

		with mesh:
			config.add_basic_configurations(
				attn_mechanism=self.attn_mechanism,
				blocksize_k=self.blocksize_k,
				blocksize_q=self.blocksize_q,
				attn_dtype=self.attn_dtype,
			)

			torch_input_ids, jax_input_ids = self.make_input_id(
				self.vocab_size,
				(self.batch_size, self.sequence_length + 1),
			)
			torch_time = time.time()
			hf_output = hf_model(
				input_ids=torch_input_ids[:, :-1],
				labels=torch_input_ids[:, 1:],
				past_key_values=None,
			)
			torch_time = time.time() - torch_time
			ed_model = module_class(
				config=config,
				dtype=self.dtype,
				param_dtype=self.dtype,
				precision=self.precision,
				rngs=nn.Rngs(0),
			)
			ed_model = ed.traversals.merge_model_and_tree(ed_model, params)
			ed_model.eval()
			ed_model = ed_model.shard_model()

			@jax.jit
			def jited(ids):
				return ed_model(
					input_ids=ids[:, :-1],
					return_dict=True,
				)

			ed_output = jited(jax_input_ids)
			easy_time = time.time()
			ed_output = jited(jax_input_ids)
			easy_time = time.time() - easy_time
			loss, _ = cross_entropy_loss_and_accuracy(
				ed_output.logits,
				jax_input_ids[:, 1:],
			)

			del params
			del hf_model
			gc.collect()
			return self.compare_torch_to_jax(
				module_name,
				hf_output,
				ed_output,
				loss,
				easy_time=easy_time,
				torch_time=torch_time,
			)

	def create_moe_test_for_models(self, module_name: str, hf_module_class):
		module_config, module_class, transform_function = ed.get_modules_by_type(
			module_name
		)
		if self.header_config is None:
			config = module_config(
				num_experts_per_tok=self.num_experts_per_tok,
				num_experts=self.num_experts,
				num_local_experts=self.num_local_experts,
				vocab_size=self.vocab_size,
				hidden_size=self.hidden_size,
				num_attention_heads=self.num_attention_heads,
				num_hidden_layers=self.num_hidden_layers,
				gradient_checkpointing=self.gradient_checkpointing,
				max_position_embeddings=self.max_position_embeddings,
				num_key_value_heads=self.num_key_value_heads,
				scan_mlp_chunk_size=self.scan_mlp_chunk_size,
				intermediate_size=self.intermediate_size,
				rotary_dim=self.rotary_dim,
				rms_norm_eps=self.rms_norm_eps,
				layer_norm_eps=self.layer_norm_eps,
				head_dim=self.head_dim,
				# residual_in_fp32=True
			)
		else:
			config = self.header_config

		hf_model = hf_module_class(config=copy.deepcopy(config))
		hf_model.eval()
		params = {
			"params": transform_function(
				state_dict=hf_model.state_dict(),
				device=jax.devices("cpu")[0],
			)
		}
		config.add_jax_args()
		config.add_basic_configurations(
			shard_attention_computation=self.shard_attention_computation,
			scan_mlp_chunk_size=self.scan_mlp_chunk_size,
			use_sharding_constraint=self.use_sharding_constraint,
		)
		mesh = config.mesh

		with mesh:
			partition_specs = match_partition_rules(config.get_partition_rules(True), params)
			shard, _ = make_shard_and_gather_fns(partition_specs, mesh)

			params = jax.tree_util.tree_map(lambda p, f: f(p), params, shard)
			config.add_basic_configurations(
				attn_mechanism=self.attn_mechanism,
				blocksize_k=self.blocksize_k,
				blocksize_q=self.blocksize_q,
			)

			ed_model = module_class(
				config=config,
				dtype=self.dtype,
				param_dtype=self.dtype,
				precision=self.precision,
			)

			torch_input_ids, jax_input_ids = self.make_input_id(
				self.vocab_size,
				(self.batch_size, self.sequence_length + 1),
			)
			hf_output = hf_model(
				input_ids=torch_input_ids[:, :-1],
				labels=torch_input_ids[:, 1:],
				output_router_logits=True,
			)
			ed_output = ed_model(
				input_ids=jax_input_ids[:, :-1],
				params=params,
				return_dict=True,
				output_router_logits=True,
			)
			loss, _ = cross_entropy_loss_and_accuracy(
				ed_output.logits,
				jax_input_ids[:, 1:],
			)
			loss += ed_output.aux_loss
			del params
			del hf_model
			gc.collect()

			return self.compare_torch_to_jax(
				"Moe-" + module_name,
				hf_output,
				ed_output,
				loss,
			)

	def test_llama(self):
		self.header_config = None
		self.rope_scaling = {
			"factor": 8.0,
			"low_freq_factor": 1.0,
			"high_freq_factor": 4.0,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3",
		}
		res, err = self.create_test_for_models(
			"llama",
			transformers.LlamaForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"Llama model Failed [ERROR {err}]")
		self.rope_scaling = None

	def test_mpt(self):
		self.header_config = ed.MptConfig(
			d_model=self.hidden_size,
			n_heads=self.num_attention_heads,
			n_layers=4,
			attn_config=ed.MptAttentionConfig(),
			axis_dims=(1, 1, 1, -1),
		)
		res, err = self.create_test_for_models(
			"mpt",
			transformers.MptForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.header_config = None
		self.assertTrue(res, f"MPT model Failed [ERROR {err}]")

	def test_falcon(self):
		# hf_model, conf = self.get_hf_model_from_hub("tiiuae/falcon-11B")
		self.header_config = None
		res, err = self.create_test_for_models(
			"falcon",
			transformers.FalconForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.header_config = None
		self.assertTrue(res, f"Falcon model Failed [ERROR {err}]")

	def test_mistral(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"mistral",
			transformers.MistralForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"Mistral model Failed [ERROR {err}]")

	def test_exaone(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
		res, err = self.create_test_for_models(
			"exaone",
			hf_model,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"EXAONE model Failed [ERROR {err}]")

	def test_internlm2(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("internlm/internlm2_5-7b-chat")
		res, err = self.create_test_for_models(
			"internlm2",
			hf_model,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"INTERNLM2 model Failed [ERROR {err}]")

	def test_mixtral(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"mixtral",
			transformers.MixtralForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"Mixtral model Failed [ERROR {err}]")

	def test_gpt2(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"gpt2",
			transformers.GPT2LMHeadModel,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"GPT2 model Failed [ERROR {err}]")

	def test_gptj(self):
		self.header_config = ed.GPTJConfig(
			vocab_size=self.vocab_size,
			n_positions=self.max_position_embeddings,
			n_embd=self.hidden_size,
			n_layer=self.num_hidden_layers,
			n_head=self.num_attention_heads,
			rotary_dim=self.hidden_size // self.num_attention_heads,
		)
		res, err = self.create_test_for_models(
			"gptj",
			transformers.GPTJForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.header_config = None
		self.assertTrue(res, f"GPT-J model Failed [ERROR {err}]")

	def test_gpt_noex(self):
		self.header_config = ed.GPTNeoXConfig(
			vocab_size=self.vocab_size,
			max_position_embeddings=self.max_position_embeddings,
			hidden_size=self.hidden_size,
			num_hidden_layers=self.num_hidden_layers,
			num_attention_heads=self.num_attention_heads,
			rotary_pct=1,
			rope_scaling=None,
		)
		res, err = self.create_test_for_models(
			"gpt_neox",
			transformers.GPTNeoXForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.header_config = None
		self.assertTrue(res, f"GPT-NoeX model Failed [ERROR {err}]")

	def test_qwen2(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"qwen2",
			transformers.Qwen2ForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"Qwen 2 model Failed [ERROR {err}]")

	def test_olmo(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"olmo",
			transformers.OlmoForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"OLMo model Failed [ERROR {err}]")

	def test_olmo2(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"olmo2",
			transformers.Olmo2ForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"OLMo2 model Failed [ERROR {err}]")

	def test_phi(self):
		self.header_config = ed.PhiConfig(
			vocab_size=51200,
			hidden_size=256,
			intermediate_size=512,
			num_hidden_layers=8,
			num_attention_heads=8,
			num_key_value_heads=None,
			resid_pdrop=0.0,
			embd_pdrop=0.0,
			attention_dropout=0.0,
			hidden_act="gelu_new",
			max_position_embeddings=2048,
			initializer_range=0.02,
			layer_norm_eps=1e-5,
			use_cache=True,
			tie_word_embeddings=False,
			rope_theta=10000.0,
			rope_scaling=None,
			partial_rotary_factor=0.5,
			qk_layernorm=False,
			bos_token_id=1,
			eos_token_id=2,
		)
		res, err = self.create_test_for_models(
			"phi",
			transformers.PhiForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.header_config = None
		self.assertTrue(res, f"PHI 2 model Failed [ERROR {err}]")

	def test_gemma(self):
		self.header_config = None
		org = self.tie_word_embeddings
		self.tie_word_embeddings = True
		res, err = self.create_test_for_models(
			"gemma",
			transformers.GemmaForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.tie_word_embeddings = org
		self.assertTrue(res, f"Gemma model Failed [ERROR {err}]")

	def test_dbrx(self):
		self.header_config = ed.DbrxConfig(
			d_model=self.hidden_size,
			n_heads=self.num_attention_heads,
			n_layers=self.num_hidden_layers,
			ffn_config=ed.DbrxFFNConfig(
				ffn_hidden_size=self.intermediate_size,
				moe_top_k=self.num_experts_per_tok,
				moe_num_experts=self.num_local_experts,
			),
			attn_config=ed.DbrxAttentionConfig(),
		)

		res, err = self.create_test_for_models(
			"dbrx",
			transformers.DbrxForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"DBRX model Failed [ERROR {err}]")

	def test_stablelm(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"stablelm",
			transformers.StableLmForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)

		self.assertTrue(res, f"StableLM model Failed [ERROR {err}]")

	def test_phi3(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("microsoft/Phi-3-mini-128k-instruct")
		res, err = self.create_test_for_models(
			"phi3",
			hf_model,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"PHI3 model Failed [ERROR {err}]")

	def test_phimoe(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("microsoft/Phi-3.5-MoE-instruct")
		self.rope_scaling = {
			"long_factor": [
				1.0199999809265137,
				1.0299999713897705,
				1.0399999618530273,
				1.0499999523162842,
				1.0499999523162842,
				1.0499999523162842,
				1.059999942779541,
				1.059999942779541,
			],
			"long_mscale": 1.243163121016122,
			"original_max_position_embeddings": 4096,
			"short_factor": [
				1.0,
				1.0399999618530273,
				1.0399999618530273,
				1.0399999618530273,
				1.0499999523162842,
				1.0499999523162842,
				1.0499999523162842,
				1.0499999523162842,
			],
			"short_mscale": 1.243163121016122,
			"type": "longrope",
		}
		res, err = self.create_test_for_models(
			"phimoe",
			hf_model,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"PHIMOE model Failed [ERROR {err}]")

	def test_deepseek_v2(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("deepseek-ai/DeepSeek-V2")
		res, err = self.create_test_for_models(
			"deepseek_v2", hf_model, ed.TaskType.CAUSAL_LM
		)

		self.assertTrue(res, f"DeepSeekv2 model Failed [ERROR {err}]")

	def test_openelm(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("apple/OpenELM-270M-Instruct")
		conf_f = ed.OpenELMConfig()
		for k, v in conf.__dict__.items():
			setattr(conf_f, k, v)
		self.header_config = conf_f
		res, err = self.create_test_for_models(
			"openelm",
			hf_model,
			ed.TaskType.CAUSAL_LM,
		)
		self.header_config = None
		self.assertTrue(res, f"OpenELM model Failed [ERROR {err}]")

	def test_arctic(self):
		self.header_config = None
		hf_model, conf = self.get_hf_model_from_hub("Snowflake/snowflake-arctic-instruct")
		res, err = self.create_test_for_models(
			"arctic",
			hf_model,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"ARCTIC model Failed [ERROR {err}]")

	def test_rwkv(self):
		self.header_config = None
		res, err = self.create_test_for_models("rwkv", transformers.RwkvForCausalLM)
		self.assertTrue(res, f"RWKV model Failed [ERROR {err}]")

	def test_gemma2(self):
		self.header_config = ed.Gemma2Config(
			32000, 128, 256, 4, 8, 4, 128 // 8, use_scan_mlp=False
		)
		res, err = self.create_test_for_models(
			"gemma2",
			transformers.Gemma2ForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"Gemma2 model Failed [ERROR {err}]")

	def test_mamba(self):
		self.header_config = None
		res, err = self.create_test_for_models("mamba", transformers.MambaForCausalLM)
		self.assertTrue(res, f"MAMBA model Failed [ERROR {err}]")

	def test_mamba2(self):
		self.header_config = ed.Mamba2Config(
			hidden_size=256,
			num_hidden_layers=16,
			num_heads=8,
		)
		res, err = self.create_test_for_models("mamba2", transformers.Mamba2ForCausalLM)
		self.assertTrue(res, f"Mamba 2 model Failed [ERROR {err}]")
		self.header_config = None

	def test_cohere(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"cohere", transformers.CohereForCausalLM, ed.TaskType.CAUSAL_LM
		)

		self.assertTrue(res, f"CoHERE model Failed [ERROR {err}]")

	def test_qwen2_moe(self):
		self.header_config = None
		res, err = self.create_test_for_models(
			"qwen2_moe",
			transformers.Qwen2MoeForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res, f"Qwen2Moe model Failed [ERROR {err}]")

	def test_moe_mixtral(self):
		self.header_config = None
		res = self.create_moe_test_for_models(
			"mixtral",
			transformers.MixtralForCausalLM,
			ed.TaskType.CAUSAL_LM,
		)
		self.assertTrue(res)

	def test_moe_qwen2_moe(self):
		self.header_config = None
		res = self.create_moe_test_for_models("qwen2_moe", transformers.Qwen2MoeForCausalLM)
		self.assertTrue(res)

	def test_moe_dbrx_moe(self):
		self.header_config = ed.DbrxConfig(
			d_model=self.hidden_size,
			n_heads=self.num_attention_heads,
			n_layers=self.num_hidden_layers,
			ffn_config=ed.DbrxFFNConfig(
				ffn_hidden_size=self.intermediate_size,
				moe_top_k=self.num_experts_per_tok,
				moe_num_experts=self.num_local_experts,
			),
			attn_config=ed.DbrxAttentionConfig(),
		)
		res = self.create_moe_test_for_models("dbrx", transformers.DbrxForCausalLM)
		self.header_config = None
		self.assertTrue(res)

	@staticmethod
	def compare_torch_to_jax(
		name,
		hf_out,
		ed_out,
		ed_loss,
		atol: float = 0.125,
		rtol: float = 0,
		easy_time: float = None,
		torch_time: float = None,
	):
		to, jo = hf_out.logits.cpu().detach().numpy(), ed_out.logits
		err = jnp.mean(to) - jnp.mean(jo)
		hf_loss = hf_out.loss.cpu().detach().numpy()
		all_close = jnp.allclose(to, jo, atol=atol, rtol=rtol)
		all_close_loss = jnp.allclose(hf_loss, ed_loss, atol=0.125, rtol=0)

		def color(text, color_code):
			return f"\x1b[{color_code}m{text}\x1b[0m"

		correct_percentage = jnp.mean(
			jnp.where(
				jnp.isclose(to, jo, atol=0.125, rtol=0),
				1,
				0,
			).reshape(-1)
		)
		err = jnp.abs(to - jo).max()

		table = tabulate(
			[
				["Last 5 elements", str(to[0, -1, -5:]), str(jo[0, -1, -5:])],
				["Loss", str(hf_loss), str(ed_loss)],
				["Took", str(torch_time), str(easy_time)],
			],
			headers=["Metric", "HuggingFace", "EasyDeL"],
			tablefmt="grid",
		)
		lose_close_string = color(str(all_close_loss), "32" if all_close_loss else "31")
		max_error_string = color(f"{err:.6f}", "32" if err < 1e-2 else "31")
		co_p = color(
			f"{correct_percentage:.2%}", "32" if correct_percentage > 0.99 else "31"
		)
		print()
		print(f"{color(name, '36;1')}")
		print(table)
		print()
		print(f"{color('Additional Information:', '33;1')}")
		print(f"Correct %: {co_p}")
		print(f"Max Error: {max_error_string}")
		print(f"Losses Close: {lose_close_string}")

		return all_close or all_close_loss, err

	@staticmethod
	def make_input_id(vocab_size: int, input_shape: tuple[int, int]):
		np_input_ids = np.random.randint(0, vocab_size, input_shape)
		return (
			torch.from_numpy(np_input_ids).to(torch.long),
			jnp.asarray(np_input_ids, dtype="i4"),
		)

	def get_hf_model_from_hub(self, repo_id):
		conf = transformers.AutoConfig.from_pretrained(
			repo_id,
			trust_remote_code=True,
		)
		for k, v in self.__dict__.items():
			if isinstance(v, (bool, str, float, type(None), int)):
				setattr(conf, k, v)
		model = type(
			transformers.AutoModelForCausalLM.from_config(
				conf,
				trust_remote_code=True,
			)
		)

		return model, conf


if __name__ == "__main__":
	# unittest.main()
	test = EasyModelsTest()
	test.setUp()
	test.test_arctic()  # Passed
	test.test_cohere()  # Passed
	test.test_dbrx()  # Passed
	test.test_deepseek_v2()  # Passed
	test.test_exaone()  # Passed
	test.test_falcon()  # Passed
	test.test_gemma()  # Passed
	test.test_gemma2()  # Passed
	test.test_gptj()  # Passed
	# test.test_gpt_noex()  # Failed
	test.test_gpt2()  # Passed
	# test.test_grok1() # Not Tested Yet!
	test.test_internlm2()  # Passed
	test.test_llama()  # Passed
	# test.test_mamba()
	# test.test_mamba2()
	test.test_mistral()  # Passed
	test.test_mixtral()  # Passed
	test.test_mpt()  # Passed
	test.test_olmo()  # Passed
	test.test_olmo2()  # Passed
	test.test_openelm()  # Passed
	test.test_phi()  # Passed
	test.test_phi3()  # Passed
	# test.test_phimoe()  # Failed v0.0.80 - N  Runtime
	test.test_qwen2()  # Passed
	test.test_qwen2_moe()  # Passed
	test.test_stablelm()  # Passed
	# -----------------------------------------------
