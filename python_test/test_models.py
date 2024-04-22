import gc
from unittest import TestCase
import unittest

import flax.traverse_util
from fjformer import make_shard_and_gather_fns, match_partition_rules

try:
    import lib.python.EasyDel as ed
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    import lib.python.EasyDel as ed

from jax import numpy as jnp
import torch
import numpy as np
from fjformer.func import cross_entropy_loss_and_accuracy
import copy
import jax
import transformers
from typing import Optional, Dict, Union, Literal

torch.manual_seed(42)


class EasyModelsTest(TestCase):

    def setUp(self) -> None:
        self.batch_size: int = 1
        self.vocab_size: int = 32000
        self.hidden_size: int = 256
        self.intermediate_size: int = 512
        self.num_hidden_layers: int = 32
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
        self.rope_theta: float = 10000.
        self.attention_bias: bool = False
        self.tie_word_embeddings: bool = False
        self.gradient_checkpointing: str = "nothing_saveable"
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
        self.precision = jax.lax.Precision("fastest")
        self.attn_mechanism: Literal[
            "vanilla",
            "flash",
            "splash",
            "ring",
            "cudnn",
            "local_ring",
            "sharded_vanilla",
            "wise_ring"
        ] = "sharded_vanilla"
        self.block_k: int = 32
        self.block_q: int = 32
        self.sequence_length = 64
        self.scan_mlp_chunk_size = self.sequence_length // 2
        self.head_dim = 256
        self.max_position_embeddings: int = self.sequence_length
        self.use_sharding_constraint = False
        self.header_config = None

    def create_test_for_models(
            self,
            module_name: str,
            hf_module_class
    ):
        module_config, module_class, transform_function = ed.get_modules_by_type(module_name)
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
                axis_dims=(1, -1, 1, 1),
                head_dim=self.head_dim
                # residual_in_fp32=True
            )
        else:
            config = self.header_config

        hf_model = hf_module_class(
            config=copy.deepcopy(config)
        )
        hf_model.eval()
        params = {
            "params":
                transform_function(
                    state_dict=hf_model.state_dict(),
                    device=jax.devices("cpu")[0],
                    remove_state_dict=True
                )
        }
        config.add_jax_args()
        config.add_basic_configurations(
            shard_attention_computation=self.shard_attention_computation,
            use_sharding_constraint=self.use_sharding_constraint,
            scan_mlp_chunk_size=self.scan_mlp_chunk_size
        )
        mesh = config.jax_mesh()

        with mesh:
            partition_specs = match_partition_rules(config.get_partition_rules(True), params)
            shard, _ = make_shard_and_gather_fns(partition_specs, jnp.float32)

            params = jax.tree_map(lambda p, f: f(p), params, shard)
            config.add_basic_configurations(
                attn_mechanism=self.attn_mechanism,
                block_k=self.block_k,
                block_q=self.block_q
            )
            prm = flax.traverse_util.flatten_dict(params, sep=".")

            torch_input_ids, jax_input_ids = self.make_input_id(
                self.vocab_size,
                (self.batch_size, self.sequence_length + 1)
            )
            hf_output = hf_model(
                input_ids=torch_input_ids[:, :-1],
                labels=torch_input_ids[:, 1:],
            )
            ed_model = module_class(
                config=config,
                dtype=self.dtype,
                param_dtype=self.dtype,
                precision=self.precision,
                _do_init=False,
                input_shape=(self.batch_size, self.sequence_length)
            )

            ed_output = ed_model(
                input_ids=jax_input_ids[:, :-1],
                params=params,
                return_dict=True,
                add_params_field=False,
                train=False
            )
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
                loss
            )

    def create_moe_test_for_models(
            self,
            module_name: str,
            hf_module_class
    ):
        module_config, module_class, transform_function = ed.get_modules_by_type(module_name)
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
                head_dim=self.head_dim
                # residual_in_fp32=True
            )
        else:
            config = self.header_config

        hf_model = hf_module_class(
            config=copy.deepcopy(config)
        )
        hf_model.eval()
        params = {
            "params":
                transform_function(
                    state_dict=hf_model.state_dict(),
                    device=jax.devices("cpu")[0],
                )
        }
        config.add_jax_args()
        config.add_basic_configurations(
            shard_attention_computation=self.shard_attention_computation,
            scan_mlp_chunk_size=self.scan_mlp_chunk_size,
            use_sharding_constraint=self.use_sharding_constraint
        )
        mesh = config.jax_mesh()

        with mesh:
            partition_specs = match_partition_rules(config.get_partition_rules(True), params)
            shard, _ = make_shard_and_gather_fns(partition_specs, jnp.float32)

            params = jax.tree_map(lambda p, f: f(p), params, shard)
            config.add_basic_configurations(
                attn_mechanism=self.attn_mechanism,
                block_k=self.block_k,
                block_q=self.block_q
            )
            prm = flax.traverse_util.flatten_dict(params, sep=".")
            ed_model = module_class(
                config=config,
                dtype=self.dtype,
                param_dtype=self.dtype,
                precision=self.precision,
                _do_init=False,
                input_shape=(self.batch_size, self.sequence_length)
            )

            torch_input_ids, jax_input_ids = self.make_input_id(
                self.vocab_size,
                (self.batch_size, self.sequence_length + 1)
            )
            hf_output = hf_model(
                input_ids=torch_input_ids[:, :-1],
                labels=torch_input_ids[:, 1:],
                output_router_logits=True
            )

            ed_output = ed_model(
                input_ids=jax_input_ids[:, :-1],
                params=params,
                return_dict=True,
                add_params_field=False,
                train=False,
                output_router_logits=True
            )
            loss, _ = cross_entropy_loss_and_accuracy(
                ed_output.logits,
                jax_input_ids[:, 1:],
            )
            loss += ed_output.aux_loss
            del params
            del hf_model
            gc.collect()
            print(f"\nHF MoE LOSS : {hf_output.loss.detach().cpu().numpy()}")
            print(f"ED MoE LOSS : {loss}")
            return jnp.allclose(hf_output.loss.detach().cpu().numpy(), loss)

    def test_llama(self):
        res, err = self.create_test_for_models("llama", transformers.LlamaForCausalLM)
        self.assertTrue(
            res,
            f"Llama model Failed [ERROR {err}]"
        )

    def test_falcon(self):
        res, err = self.create_test_for_models("falcon", transformers.FalconForCausalLM)
        self.assertTrue(
            res,
            f"Falcon model Failed [ERROR {err}]"
        )

    def test_mistral(self):
        res, err = self.create_test_for_models("mistral", transformers.MistralForCausalLM)
        self.assertTrue(
            res,
            f"Mistral model Failed [ERROR {err}]"
        )

    def test_mixtral(self):
        res, err = self.create_test_for_models("mixtral", transformers.MixtralForCausalLM)
        self.assertTrue(
            res,
            f"Mixtral model Failed [ERROR {err}]"
        )

    def test_gpt2(self):
        res, err = self.create_test_for_models("gpt2", transformers.GPT2LMHeadModel)
        self.assertTrue(
            res,
            f"GPT2 model Failed [ERROR {err}]"
        )

    def test_gptj(self):
        res, err = self.create_test_for_models("gptj", transformers.GPTJForCausalLM)
        self.assertTrue(
            res,
            f"GPT-J model Failed [ERROR {err}]"
        )

    def test_qwen2(self):
        res, err = self.create_test_for_models("qwen2", transformers.Qwen2ForCausalLM)
        self.assertTrue(
            res,
            f"Qwen 2 model Failed [ERROR {err}]"
        )

    def test_phi(self):
        res, err = self.create_test_for_models("phi", transformers.PhiForCausalLM)
        self.assertTrue(
            res,
            f"PHI 2 model Failed [ERROR {err}]"
        )

    def test_gemma(self):
        org = self.tie_word_embeddings
        self.tie_word_embeddings = True
        res, err = self.create_test_for_models("gemma", transformers.GemmaForCausalLM)
        self.tie_word_embeddings = org
        self.assertTrue(
            res,
            f"Gemma model Failed [ERROR {err}]"
        )

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
            attn_config=ed.DbrxAttentionConfig()
        )

        res, err = self.create_test_for_models("dbrx", transformers.DbrxForCausalLM)
        self.assertTrue(
            res,
            f"DBRX model Failed [ERROR {err}]"
        )

    def test_stablelm(self):
        res, err = self.create_test_for_models("stablelm", transformers.StableLmForCausalLM)

        self.assertTrue(
            res,
            f"StableLM model Failed [ERROR {err}]"
        )

    def test_rwkv(self):
        res, err = self.create_test_for_models("rwkv", transformers.RwkvForCausalLM)
        self.assertTrue(
            res,
            f"RWKV model Failed [ERROR {err}]"
        )

    def test_mamba(self):
        res, err = self.create_test_for_models("mamba", transformers.MambaForCausalLM)

        self.assertTrue(
            res,
            f"MAMBA model Failed [ERROR {err}]"
        )

    def test_cohere(self):
        res, err = self.create_test_for_models("cohere", transformers.CohereForCausalLM)

        self.assertTrue(
            res,
            f"MAMBA model Failed [ERROR {err}]"
        )

    def test_qwen2_moe(self):
        res, err = self.create_test_for_models("qwen2_moe", transformers.Qwen2MoeForCausalLM)
        self.assertTrue(
            res,
            f"Qwen2Moe model Failed [ERROR {err}]"
        )

    def test_moe_mixtral(self):
        res = self.create_moe_test_for_models("mixtral", transformers.MixtralForCausalLM)
        self.assertTrue(res)

    def test_moe_qwen2_moe(self):
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
            attn_config=ed.DbrxAttentionConfig()
        )
        res = self.create_moe_test_for_models("dbrx", transformers.DbrxForCausalLM)
        self.assertTrue(res)

    @staticmethod
    def compare_torch_to_jax(name, hf_out, ed_out, ed_loss, atol: float = 1e-035, rtol: float = 1e-08):
        to, jo = hf_out.logits.cpu().detach().numpy(), ed_out.logits
        err = jnp.mean(to - jo)
        hf_loss = hf_out.loss.cpu().detach().numpy()
        all_close = jnp.allclose(to, jo, atol=atol, rtol=rtol)
        all_close_loss = jnp.allclose(hf_loss, ed_loss, atol=1e-02, rtol=rtol)

        if not all_close:
            print(f"\n{name} LAST F HF : ", to[0, -1, -5:])
            print(f"{name} LAST F ED : ", jo[0, -1, -5:])
            print(f"{name} CORRECT % : ", jnp.mean(
                jnp.where(
                    jnp.isclose(to, jo, atol=atol, rtol=rtol), 1, 0).reshape(-1)
            )
                  )
            print(f"{name} LOSS F HF : ", hf_loss)
            print(f"{name} LOSS F ED : ", ed_loss)
            print(f"IS LOSS CLOSE    : ", all_close_loss)
        return all_close or all_close_loss, err

    @staticmethod
    def make_input_id(
            vocab_size: int,
            input_shape: tuple[int, int]
    ):
        np_input_ids = np.random.randint(0, vocab_size, input_shape)
        return (
            torch.from_numpy(np_input_ids).to(torch.long),
            jnp.asarray(np_input_ids, dtype="i4")
        )


if __name__ == "__main__":
    unittest.main()
