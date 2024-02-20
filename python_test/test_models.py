import gc
from unittest import TestCase
import unittest

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

import copy
import jax
import transformers
from typing import Optional, Dict, Union, Literal

torch.manual_seed(42)


class EasyModelsTest(TestCase):

    def setUp(self) -> None:
        self.sequence_length: int = 128
        self.batch_size: int = 1
        self.vocab_size: int = 32000
        self.hidden_size: int = 256
        self.intermediate_size: int = 512
        self.num_hidden_layers: int = 2
        self.num_attention_heads: int = 4
        self.number_rep_kv: int = 1
        self.num_key_value_heads: Optional[int] = 2
        self.max_position_embeddings: int = 2048
        self.rms_norm_eps: float = 1e-6
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
        self.use_pjit_attention_force: bool = False
        self.rope_scaling: Dict[str, Union[str, float]] = None
        self.use_sacn_mlp: bool = False
        self.scan_mlp_chunk_size: int = 1024
        self.bits: Optional[int] = None
        self.hidden_act: str = "silu"
        self.pretraining_tp: int = 1
        self.scan_layers: bool = False
        self.use_shard_map: bool = False
        self.dtype: jax.numpy.dtype = jnp.float32
        self.precision = jax.lax.Precision("fastest")
        self.attn_mechanism: Literal["normal", "flash", "splash", "ring"] = "normal"
        self.block_k: int = 64
        self.block_q: int = 64

    def create_test_for_models(
            self,
            module_name: str,
            hf_module_class
    ):
        module_config, module_class, transform_function = ed.get_modules_by_type(module_name)
        config = module_config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            gradient_checkpointing=self.gradient_checkpointing,
            max_position_embeddings=self.max_position_embeddings,
            num_key_value_heads=self.num_key_value_heads
        )

        input_shape = (self.batch_size, self.sequence_length)

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
            use_shard_map=self.use_shard_map
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
            ed_model = module_class(
                config=config,
                dtype=self.dtype,
                param_dtype=self.dtype,
                precision=self.precision,
                _do_init=False,
                input_shape=input_shape
            )

            torch_input_ids, jax_input_ids = self.make_input_id(self.vocab_size, input_shape)
            hf_output = hf_model(
                input_ids=torch_input_ids,
                attention_mask=torch.ones(*input_shape)
            )
            ed_output = ed_model(
                input_ids=jax_input_ids,
                params=params,
                return_dict=True,
                add_params_field=False,
                train=False

            )
            del params
            del hf_model
            gc.collect()
            return self.compare_torch_to_jax(hf_output, ed_output)

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

    @staticmethod
    def compare_torch_to_jax(to, jo, atol: float = 1e-4):
        to, jo = to.logits.cpu().detach().numpy(), jo.logits
        err = jnp.mean(to - jo)
        return jnp.allclose(to, jo, atol=atol), err

    @staticmethod
    def make_input_id(
            vocab_size: int,
            input_shape: tuple[int, int]
    ):
        np_input_ids = np.random.randint(0, vocab_size, input_shape)
        return (
            torch.from_numpy(np_input_ids).reshape(1, -1).to(torch.long),
            jnp.asarray(np_input_ids, dtype="i4").reshape(1, -1)
        )


if __name__ == "__main__":
    unittest.main()
