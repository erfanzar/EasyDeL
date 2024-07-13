import gc
import unittest
from unittest import TestCase

import flax.traverse_util
from fjformer import make_shard_and_gather_fns, match_partition_rules

try:
    import src.python.easydel as ed
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    import src.python.easydel as ed

import copy
from typing import Dict, Literal, Optional, Union

import jax
import numpy as np
import torch
import transformers
from jax import numpy as jnp

torch.manual_seed(42)


class EasyModelsGenerationTest(TestCase):

    def setUp(self) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.sequence_length: int = 128
        self.batch_size: int = 1
        self.vocab_size: int = self.tokenizer.vocab_size
        self.hidden_size: int = 256
        self.intermediate_size: int = 512
        self.num_hidden_layers: int = 4
        self.num_attention_heads: int = 8
        self.num_key_value_heads: Optional[int] = 4
        self.max_position_embeddings: int = 512
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
        self.gradient_checkpointing: str = "nothing_saveable"  #
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
            "legacy_sharded_vanilla",
            "wise_ring",
            "blockwise",
            "pallas_flash",
        ] = "sharded_vanilla"
        self.block_k: int = 64
        self.block_q: int = 64
        self.scan_mlp_chunk_size = self.sequence_length // 2
        self.quantize_kv_cache = True

    def create_generation_test_for_models(
        self,
        module_name: str,
    ):
        module_config, module_class, transform_function = ed.get_models_by_type(
            module_name
        )
        config = module_config(
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
            # residual_in_fp32=True
        )

        input_shape = (self.batch_size, self.sequence_length)

        config.add_jax_args()
        config.add_basic_configurations(
            shard_attention_computation=self.shard_attention_computation,
            scan_mlp_chunk_size=self.scan_mlp_chunk_size,
        )

        model = module_class(
            config=config,
            dtype=self.dtype,
            param_dtype=self.dtype,
            precision=self.precision,
            _do_init=True,
        )
        mesh = config.mesh
        server_config = ed.JAXServerConfig(
            batch_size=1,
            max_sequence_length=self.max_position_embeddings,
            dtype=self.dtype,
            use_prefix_tokenizer=True,
            max_compile_tokens=self.max_position_embeddings // 4,
            max_new_tokens=self.max_position_embeddings,
        )
        server = ed.JAXServer.from_parameters(
            config_model=config,
            model=model,
            params=model.params,
            server_config=server_config,
            tokenizer=self.tokenizer,
        )
        response = None
        for response, tokens_used in server.sample(""):
            ...
        return response

    def test_llama(self):
        response = self.create_generation_test_for_models("llama")
        print(f"LLama EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_falcon(self):
        response = self.create_generation_test_for_models("falcon")
        print(f"Falcon EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_mistral(self):
        response = self.create_generation_test_for_models("mistral")
        print(f"Mistral EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_mixtral(self):
        response = self.create_generation_test_for_models("mixtral")
        print(f"Mixtral EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(True)

    def test_gpt2(self):
        response = self.create_generation_test_for_models("gpt2")
        print(f"GTP2 EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_gptj(self):
        response = self.create_generation_test_for_models("gptj")
        print(f"GPT-J EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_qwen2(self):
        response = self.create_generation_test_for_models("qwen2")
        print(f"QWEN2 EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_phi(self):
        response = self.create_generation_test_for_models("phi")
        print(f"PHI EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_gemma(self):
        response = self.create_generation_test_for_models("gemma")
        print(f"GEMMA EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    def test_stablelm(self):
        response = self.create_generation_test_for_models("stablelm")
        print(f"StableLM EasyDeL Generated Sequence:\n{response}")
        self.assertTrue(
            True,
        )

    # def test_rwkv(self):
    #     response = self.create_generation_test_for_models("rwkv")
    #     print(f"RWKV EasyDeL Generated Sequence:\n{response}")
    #     self.assertTrue(
    #         True,
    #     )
    #
    # def test_mamba(self):
    #     response = self.create_generation_test_for_models("mamba")
    #     print(f"MAMBA EasyDeL Generated Sequence:\n{response}")
    #     self.assertTrue(
    #         True,
    #     )

    @staticmethod
    def make_input_id(vocab_size: int, input_shape: tuple[int, int]):
        np_input_ids = np.random.randint(0, vocab_size, input_shape)
        return (
            torch.from_numpy(np_input_ids).reshape(1, -1).to(torch.long),
            jnp.asarray(np_input_ids, dtype="i4").reshape(1, -1),
        )


if __name__ == "__main__":
    unittest.main()
