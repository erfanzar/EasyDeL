import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import copy
import gc
import unittest

import jax
import jax.extend
import jax.random
import numpy as np
import torch
import transformers
from flax import nnx as nn
from jax import numpy as jnp
from tabulate import tabulate

import easydel as ed
from easydel.infra.etils import AVAILABLE_ATTENTION_MECHANISMS, EasyDeLGradientCheckPointers

torch.manual_seed(42)
STRICT_CHECK = False


class EasyModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size: int = 2
        self.vocab_size: int = 32000
        self.hidden_size: int = 64
        self.intermediate_size: int = 128
        self.num_hidden_layers: int = 8
        self.num_attention_heads: int = 8
        self.num_key_value_heads: int | None = 4
        self.num_experts_per_tok = 2
        self.num_experts = 4
        self.num_local_experts = self.num_experts
        self.rms_norm_eps: float = 1e-6
        self.layer_norm_eps = self.rms_norm_eps
        self.initializer_range: float = 0.02
        self.use_cache: bool = True
        self.use_pallas_group_matmul: bool = False
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
        self.rope_scaling: dict[str, str | float] | None = None
        self.use_scan_mlp: bool = False
        self.bits: int | None = None
        self.hidden_act: str = "silu"
        self.scan_layers: bool = False
        self.shard_attention_computation: bool = True
        self.rotary_dim = 32
        self.dtype: jax.numpy.dtype = jnp.float32
        self.precision = jax.lax.Precision("highest")

        self.attn_mechanism: AVAILABLE_ATTENTION_MECHANISMS = "vanilla"

        self.blocksize_k: int = 128
        self.blocksize_q: int = 128
        self.sequence_length = 128

        self.sliding_window = 64
        self.use_sliding_window = True

        self.scan_mlp_chunk_size = self.sequence_length // 2
        self.sharding_axis_dims = (1, 1, 1, -1, 1)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.use_parallel_residual = True
        self.qk_layernorm = True
        self.max_position_embeddings: int = self.sequence_length
        self.use_sharding_constraint = False
        self.header_config = None
        self.pad_token_id = None
        self.rope_scaling = None
        self.attn_dtype = jnp.float32
        self.attn_softmax_dtype = jnp.float32
        self.platform = None

    @torch.no_grad()
    def create_test_for_models(
        self,
        module_name: str,
        hf_module_class,
        task,
        extra_exec: dict | None = None,
        generation_test: bool = False,
    ):
        module_config, module_class = ed.get_modules_by_type(module_name, task)
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
                max_context_length=self.max_position_embeddings,
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
                use_scan_mlp=self.use_scan_mlp,
                scan_mlp=self.use_scan_mlp,
                use_pallas_group_matmul=self.use_pallas_group_matmul,
                # sliding_window=self.sliding_window,
                # use_sliding_window=self.use_sliding_window,
            )
        else:
            config = self.header_config
        kwargs_torch = {}
        kwargs_easydel = {}
        if extra_exec is not None:
            for k, v in extra_exec.items():
                kwargs_easydel[k] = jnp.ones(v["shape"], dtype=getattr(jnp, v["dtype"]))
                kwargs_torch[k] = torch.ones(v["shape"], dtype=getattr(torch, v["dtype"]))
        config.sharding_axis_dims = self.sharding_axis_dims
        config.pad_token_id = 0

        hf_model = hf_module_class(config=copy.deepcopy(config))
        hf_model.eval()
        hf_model = hf_model.float()

        config.attach_custom_arguments()
        config.add_basic_configurations(
            shard_attention_computation=self.shard_attention_computation,
            use_sharding_constraint=self.use_sharding_constraint,
            scan_mlp_chunk_size=self.scan_mlp_chunk_size,
        )
        mesh = config.mesh
        config.add_basic_configurations(
            attn_mechanism=self.attn_mechanism,
            blocksize_k=self.blocksize_k,
            blocksize_q=self.blocksize_q,
            attn_dtype=self.attn_dtype,
        )
        with mesh:
            torch_input_ids, jax_input_ids = self.make_input_id(self.vocab_size, (self.batch_size, self.sequence_length))
            with ed.utils.capture_time() as torch_time:
                try:
                    hf_output = hf_model(
                        input_ids=torch_input_ids,
                        attention_mask=torch.ones_like(torch_input_ids),
                        labels=torch_input_ids,
                        output_router_logits=True,
                        past_key_values=None,
                        use_cache=generation_test,
                        **kwargs_torch,
                    )
                except Exception:
                    hf_output = hf_model(
                        input_ids=torch_input_ids,
                        attention_mask=torch.ones_like(torch_input_ids),
                        labels=torch_input_ids,
                        past_key_values=None,
                        use_cache=generation_test,
                        **kwargs_torch,
                    )
            torch_time = torch_time()

            ed_model = module_class.lazy_init(
                config=config,
                dtype=self.dtype,
                param_dtype=self.dtype,
                precision=self.precision,
                rngs=nn.Rngs(0),
            )
            ed_model = ed.traversals.merge_model_and_tree(ed_model, tree=ed_model.transform_fn(hf_model.state_dict()))
            ed_model.eval()
            ed_model = ed_model.shard_model()

            try:

                @ed.ejit(static_argnums=(1,))
                def jited(ids, gd, gs, go):
                    return nn.merge(gd, gs, go).compute_loss(
                        input_ids=ids,
                        attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                        output_router_logits=True,
                        **kwargs_easydel,
                    )

                ed_output = jited(jax_input_ids, *ed_model.split_module())
            except Exception:

                @ed.ejit(static_argnums=(1,))
                def jited(ids, gd, gs, go):
                    return nn.merge(gd, gs, go).compute_loss(
                        input_ids=ids,
                        attention_mask=jnp.ones_like(ids, dtype=jnp.bool),
                        **kwargs_easydel,
                    )

                ed_output = jited(jax_input_ids, *ed_model.split_module())

            with ed.utils.capture_time() as easy_time:
                ed_output, metrics = jited(jax_input_ids, *ed_model.split_module())
            easy_time = easy_time()

            del hf_model
            gc.collect()
            return self.compare_torch_to_jax(
                module_name,
                hf_out=hf_output,
                ed_out=ed_output,
                easy_time=easy_time,
                torch_time=torch_time,
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
        res, err = self.create_test_for_models("llama", transformers.LlamaForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Llama model Failed [ERROR {err}]")
        self.rope_scaling = None

    def test_llama4(self):
        self.header_config = ed.Llama4TextConfig(
            hidden_size=128,
            intermediate_size=512,
            intermediate_size_mlp=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            use_qk_norm=False,
        )
        res, err = self.create_test_for_models("llama4_text", transformers.Llama4ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Llama4 model Failed [ERROR {err}]")
        self.rope_scaling = None

    def test_llama4_cond(self):
        self.header_config = ed.Llama4Config(
            boi_token_index=200080,
            eoi_token_index=200081,
            image_token_index=200092,
        )

        text_config = ed.Llama4TextConfig(
            hidden_size=512,
            intermediate_size=2048,
            intermediate_size_mlp=2048,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            use_qk_norm=False,
            vocab_size=202048,
            bos_token_id=200000,
            eos_token_id=[200001, 200007, 200008],
            pad_token_id=200018,
            hidden_act="silu",
            max_position_embeddings=4096,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            use_cache=True,
            attention_bias=False,
            attention_dropout=0.0,
            rope_theta=500000.0,
            rope_scaling=None,
            num_experts_per_tok=1,
            num_local_experts=1,
            output_router_logits=False,
            router_aux_loss_coef=0.0,
            router_jitter_noise=0.0,
        )

        vision_config = ed.Llama4VisionConfig(
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=2048,
            vision_output_dim=512,
            projector_input_dim=512,
            projector_output_dim=512,
            image_size=336,
            patch_size=14,
            num_channels=3,
            hidden_act="gelu",
            initializer_range=0.02,
            norm_eps=1e-05,
            attention_dropout=0.0,
            rope_theta=10000,
            pixel_shuffle_ratio=0.5,
            projector_dropout=0.0,
            multi_modal_projector_bias=False,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        self.header_config.text_config = text_config
        self.header_config.vision_config = vision_config

        res, err = self.create_test_for_models(
            "llama4",
            transformers.Llama4ForConditionalGeneration,
            ed.TaskType.IMAGE_TEXT_TO_TEXT,
            {"pixel_values": {"dtype": "float32", "shape": (17, 3, 336, 336)}},
        )
        self.assertTrue(res, f"Llama4 model Failed [ERROR {err}]")
        self.rope_scaling = None

    def test_mpt(self):
        self.header_config = ed.MptConfig(
            d_model=self.hidden_size,
            n_heads=self.num_attention_heads,
            n_layers=4,
            attn_config=ed.MptAttentionConfig(),
            sharding_axis_dims=(1, 1, 1, 1, -1),
        )
        res, err = self.create_test_for_models("mpt", transformers.MptForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"MPT model Failed [ERROR {err}]")

    def test_falcon(self):
        # hf_model, conf = self.get_hf_model_from_hub("tiiuae/falcon-11B")
        self.header_config = None
        res, err = self.create_test_for_models("falcon", transformers.FalconForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"Falcon model Failed [ERROR {err}]")

    def test_mistral(self):
        self.header_config = None
        res, err = self.create_test_for_models("mistral", transformers.MistralForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Mistral model Failed [ERROR {err}]")

    def test_exaone(self):
        self.header_config = None
        hf_model, conf = self.get_hf_model_from_hub("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
        res, err = self.create_test_for_models("exaone", hf_model, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"EXAONE model Failed [ERROR {err}]")

    def test_internlm2(self):
        self.header_config = None
        hf_model, conf = self.get_hf_model_from_hub("internlm/internlm2_5-7b-chat")
        res, err = self.create_test_for_models("internlm2", hf_model, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"INTERNLM2 model Failed [ERROR {err}]")

    def test_gemma3(self):
        repo_id = "google/gemma-3-4b-it"
        model_task = ed.TaskType.IMAGE_TEXT_TO_TEXT
        conf = ed.AutoEasyDeLConfig.from_pretrained(repo_id, trust_remote_code=True, model_task=model_task)

        conf.text_config.hidden_size = self.hidden_size

        conf.text_config.num_attention_heads = self.num_attention_heads
        conf.text_config.num_key_value_heads = self.num_key_value_heads

        conf.text_config.num_hidden_layers = self.num_hidden_layers
        conf.text_config.sliding_window_pattern = self.num_hidden_layers // 4
        conf.text_config.freq_max_position_embedding = self.max_position_embeddings
        conf.text_config.mask_max_position_embedding = self.max_position_embeddings

        conf.text_config.vocab_size = self.vocab_size
        conf.text_config.attn_mechanism = "vanilla"

        hf_model = transformers.Gemma3ForConditionalGeneration

        self.header_config = conf
        res, err = self.create_test_for_models("gemma3", hf_model, model_task)
        self.header_config = None
        self.assertTrue(res, f"Gemma3 model Failed [ERROR {err}]")

    def test_gemma3_text(self):
        repo_id = "google/gemma-3-1b-it"
        conf = ed.AutoEasyDeLConfig.from_pretrained(repo_id, trust_remote_code=True)
        conf.hidden_size = self.hidden_size
        conf.num_attention_heads = self.num_attention_heads
        conf.num_key_value_heads = self.num_key_value_heads
        conf.num_hidden_layers = self.num_hidden_layers
        conf.freq_max_position_embedding = self.max_position_embeddings
        conf.mask_max_position_embedding = self.max_position_embeddings
        conf.sliding_window_pattern = 2
        conf.sliding_window = 256
        conf.vocab_size = self.vocab_size
        conf.attn_mechanism = "vanilla"
        hf_model = transformers.Gemma3ForCausalLM

        self.header_config = conf
        res, err = self.create_test_for_models("gemma3_text", hf_model, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"Gemma3Text model Failed [ERROR {err}]")

    def test_mixtral(self):
        self.header_config = None
        res, err = self.create_test_for_models("mixtral", transformers.MixtralForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Mixtral model Failed [ERROR {err}]")

    def test_gpt2(self):
        self.header_config = None
        res, err = self.create_test_for_models("gpt2", transformers.GPT2LMHeadModel, ed.TaskType.CAUSAL_LM)
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
        res, err = self.create_test_for_models("gptj", transformers.GPTJForCausalLM, ed.TaskType.CAUSAL_LM)
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
        res, err = self.create_test_for_models("gpt_neox", transformers.GPTNeoXForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"GPT-NoeX model Failed [ERROR {err}]")

    def test_gpt_oss(self):
        self.header_config = ed.GptOssConfig(
            num_hidden_layers=8,
            num_local_experts=8,
            vocab_size=201088,
            hidden_size=128,
            intermediate_size=256,
            head_dim=64,
            num_attention_heads=8,
            num_key_value_heads=4,
            sliding_window=128,
            rope_theta=150000.0,
            tie_word_embeddings=False,
            hidden_act="silu",
            initializer_range=0.02,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            rope_scaling=None,
            attention_dropout=0.0,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
        )
        res, err = self.create_test_for_models("gpt_oss", transformers.GptOssForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"GPT-OSS model Failed [ERROR {err}]")

    def test_glm(self):
        self.header_config = ed.GlmConfig(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            partial_rotary_factor=0.5,
            head_dim=128,
            hidden_act="silu",
            attention_dropout=0.0,
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=0.00000015625,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            pad_token_id=151329,
            eos_token_id=None,
            bos_token_id=None,
            attention_bias=True,
        )
        res, err = self.create_test_for_models("glm", transformers.GlmForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"GLM model Failed [ERROR {err}]")

    def test_glm4(self):
        self.header_config = ed.Glm4Config(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            partial_rotary_factor=0.5,
            head_dim=128,
            hidden_act="silu",
            attention_dropout=0.0,
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=0.00000015625,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            pad_token_id=151329,
            eos_token_id=None,
            bos_token_id=None,
            attention_bias=True,
        )
        res, err = self.create_test_for_models("glm4", transformers.Glm4ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"GLM4 model Failed [ERROR {err}]")

    def test_glm4_moe(self):
        self.header_config = ed.Glm4MoeConfig(
            vocab_size=151552,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            partial_rotary_factor=0.5,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            moe_intermediate_size=1408,
            num_experts_per_tok=4,
            n_shared_experts=4,
            n_routed_experts=4,
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            norm_topk_prob=True,
            use_qk_norm=False,
        )
        res, err = self.create_test_for_models("glm4_moe", transformers.Glm4MoeForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"GLM4-MoE model Failed [ERROR {err}]")

    def test_qwen2(self):
        self.header_config = None
        self.rope_scaling = None
        res, err = self.create_test_for_models("qwen2", transformers.Qwen2ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Qwen 2 model Failed [ERROR {err}]")

    def test_qwen3(self):
        self.header_config = ed.Qwen3Config(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
        )
        self.rope_scaling = None
        res, err = self.create_test_for_models("qwen3", transformers.Qwen3ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Qwen 3 model Failed [ERROR {err}]")

    def test_olmo(self):
        self.header_config = None
        res, err = self.create_test_for_models("olmo", transformers.OlmoForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"OLMo model Failed [ERROR {err}]")

    def test_olmo2(self):
        self.header_config = None
        res, err = self.create_test_for_models("olmo2", transformers.Olmo2ForCausalLM, ed.TaskType.CAUSAL_LM)
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
            max_position_embeddings=self.max_position_embeddings,
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
        res, err = self.create_test_for_models("phi", transformers.PhiForCausalLM, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"PHI 2 model Failed [ERROR {err}]")

    def test_gemma(self):
        self.header_config = None
        org = self.tie_word_embeddings
        self.tie_word_embeddings = True
        res, err = self.create_test_for_models("gemma", transformers.GemmaForCausalLM, ed.TaskType.CAUSAL_LM)
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
            max_seq_len=self.max_position_embeddings,
        )

        res, err = self.create_test_for_models("dbrx", transformers.DbrxForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"DBRX model Failed [ERROR {err}]")

    def test_stablelm(self):
        self.header_config = None
        res, err = self.create_test_for_models("stablelm", transformers.StableLmForCausalLM, ed.TaskType.CAUSAL_LM)

        self.assertTrue(res, f"StableLM model Failed [ERROR {err}]")

    def test_phi3(self):
        self.header_config = None
        self.rope_scaling = {
            "long_factor": [
                1.0199999809265137,
                1.0299999713897705,
                1.0399999618530273,
                1.0499999523162842,
            ],
            "long_mscale": 1.8,
            "original_max_position_embeddings": 4,
            "short_factor": [
                1.0,
                1.0399999618530273,
                1.0399999618530273,
                1.0399999618530273,
            ],
            "short_mscale": 1.1,
            "type": "longrope",
        }
        res, err = self.create_test_for_models("phi3", transformers.Phi3ForCausalLM, ed.TaskType.CAUSAL_LM)
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
        res, err = self.create_test_for_models("phimoe", hf_model, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"PHIMOE model Failed [ERROR {err}]")

    def test_deepseek_v2(self):
        hf_model, conf = self.get_hf_model_from_hub("deepseek-ai/DeepSeek-V2")
        self.header_config = ed.DeepseekV2Config(
            vocab_size=conf.vocab_size,
            hidden_size=conf.hidden_size,
            intermediate_size=conf.intermediate_size,
            moe_intermediate_size=conf.moe_intermediate_size,
            num_hidden_layers=conf.num_hidden_layers,
            num_attention_heads=conf.num_attention_heads,
            num_key_value_heads=conf.num_key_value_heads,
            n_shared_experts=conf.n_shared_experts,
            n_routed_experts=conf.n_routed_experts,
            ep_size=conf.ep_size,
            routed_scaling_factor=conf.routed_scaling_factor,
            kv_lora_rank=conf.kv_lora_rank,
            q_lora_rank=conf.q_lora_rank,
            qk_rope_head_dim=conf.qk_rope_head_dim,
            v_head_dim=conf.v_head_dim,
            qk_nope_head_dim=conf.qk_nope_head_dim,
            topk_method=conf.topk_method,
            n_group=conf.n_group,
            topk_group=conf.topk_group,
            num_experts_per_tok=conf.num_experts_per_tok,
            moe_layer_freq=conf.moe_layer_freq,
            first_k_dense_replace=conf.first_k_dense_replace,
            norm_topk_prob=conf.norm_topk_prob,
            scoring_func=conf.scoring_func,
            aux_loss_alpha=conf.aux_loss_alpha,
            seq_aux=conf.seq_aux,
            hidden_act=conf.hidden_act,
            max_position_embeddings=conf.max_position_embeddings,
            initializer_range=conf.initializer_range,
            rms_norm_eps=conf.rms_norm_eps,
            use_cache=conf.use_cache,
            pad_token_id=conf.pad_token_id,
            bos_token_id=conf.bos_token_id,
            eos_token_id=conf.eos_token_id,
            pretraining_tp=conf.pretraining_tp,
            tie_word_embeddings=conf.tie_word_embeddings,
            rope_theta=conf.rope_theta,
            attention_bias=conf.attention_bias,
            attention_dropout=conf.attention_dropout,
            gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
            rope_scaling=conf.rope_scaling,
        )
        res, err = self.create_test_for_models("deepseek_v2", hf_model, ed.TaskType.CAUSAL_LM)

        self.assertTrue(res, f"DeepSeekv2 model Failed [ERROR {err}]")

    def test_deepseek_v3(self):
        self.header_config = None
        hf_model, conf = self.get_hf_model_from_hub("deepseek-ai/DeepSeek-V3")
        self.header_config = ed.DeepseekV3Config(
            **{
                "aux_loss_alpha": 0.001,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "ep_size": 1,
                "first_k_dense_replace": 3,
                "hidden_act": "silu",
                "hidden_size": 128,
                "initializer_range": 0.02,
                "intermediate_size": 256,
                "kv_lora_rank": 512,
                "max_position_embeddings": 1024,
                "moe_intermediate_size": 128,
                "moe_layer_freq": 1,
                "n_group": 8,
                "n_routed_experts": 32,
                "n_shared_experts": 1,
                "norm_topk_prob": True,
                "num_attention_heads": 128,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 4,
                "num_key_value_heads": 128,
                "num_nextn_predict_layers": 1,
                "pretraining_tp": 1,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 40,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn",
                },
                "rope_theta": 10000,
                "routed_scaling_factor": 2.5,
                "scoring_func": "sigmoid",
                "seq_aux": True,
                "tie_word_embeddings": False,
                "topk_group": 4,
                "topk_method": "noaux_tc",
                "transformers_version": "4.33.1",
                "use_cache": True,
                "v_head_dim": 128,
                "vocab_size": 129280,
            }
        )
        res, err = self.create_test_for_models("deepseek_v3", hf_model, ed.TaskType.CAUSAL_LM)

        self.assertTrue(res, f"DeepSeekv3 model Failed [ERROR {err}]")

    def test_openelm(self):
        self.header_config = None
        hf_model, conf = self.get_hf_model_from_hub("apple/OpenELM-270M-Instruct")
        conf_f = ed.OpenELMConfig()
        for k, v in conf.__dict__.items():
            setattr(conf_f, k, v)

        conf_f.max_context_length = (self.max_position_embeddings,)
        self.header_config = conf_f
        res, err = self.create_test_for_models("openelm", hf_model, ed.TaskType.CAUSAL_LM)
        self.header_config = None
        self.assertTrue(res, f"OpenELM model Failed [ERROR {err}]")

    def test_arctic(self):
        self.header_config = None
        hf_model, conf = self.get_hf_model_from_hub("Snowflake/snowflake-arctic-instruct")
        res, err = self.create_test_for_models("arctic", hf_model, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"ARCTIC model Failed [ERROR {err}]")

    def test_rwkv(self):
        self.header_config = None
        res, err = self.create_test_for_models("rwkv", transformers.RwkvForCausalLM)
        self.assertTrue(res, f"RWKV model Failed [ERROR {err}]")

    def test_gemma2(self):
        self.header_config = ed.Gemma2Config(32000, 128, 256, 4, 8, 4, 128 // 8, use_scan_mlp=False)
        res, err = self.create_test_for_models("gemma2", transformers.Gemma2ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Gemma2 model Failed [ERROR {err}]")

    def test_mamba(self):
        self.header_config = None
        res, err = self.create_test_for_models("mamba", transformers.MambaForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"MAMBA model Failed [ERROR {err}]")

    def test_mamba2(self):
        self.header_config = ed.Mamba2Config(
            hidden_size=256,
            num_hidden_layers=16,
            num_heads=8,
        )
        res, err = self.create_test_for_models("mamba2", transformers.Mamba2ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Mamba 2 model Failed [ERROR {err}]")
        self.header_config = None

    def test_cohere(self):
        self.header_config = None
        res, err = self.create_test_for_models("cohere", transformers.CohereForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Cohere model Failed [ERROR {err}]")

    def test_cohere2(self):
        self.header_config = None
        res, err = self.create_test_for_models("cohere2", transformers.Cohere2ForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Cohere2 model Failed [ERROR {err}]")

    def test_qwen2_moe(self):
        self.header_config = None
        self.rope_scaling = None
        res, err = self.create_test_for_models("qwen2_moe", transformers.Qwen2MoeForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Qwen2Moe model Failed [ERROR {err}]")

    def test_qwen3_moe(self):
        self.header_config = None
        self.rope_scaling = None
        res, err = self.create_test_for_models("qwen3_moe", transformers.Qwen3MoeForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"Qwen3Moe model Failed [ERROR {err}]")

    def test_qwen2_vl(self):
        self.header_config = ed.AutoEasyDeLConfig.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            model_task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
        )
        self.header_config.hidden_size = 128 * 4
        self.header_config.initializer_range = 0.02
        self.header_config.intermediate_size = 256
        self.header_config.max_position_embeddings = 1024
        self.header_config.max_window_layers = 8
        self.header_config.num_attention_heads = 4
        self.header_config.num_hidden_layers = 8
        self.header_config.num_key_value_heads = 2
        task_ = ed.TaskType.IMAGE_TEXT_TO_TEXT
        res, err = self.create_test_for_models("qwen2_vl", transformers.Qwen2VLForConditionalGeneration, task_)
        self.rope_scaling = None
        self.assertTrue(res, f"Qwen2VL model Failed [ERROR {err}]")

    def test_roberta(self):
        self.header_config = ed.RobertaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        res, err = self.create_test_for_models("roberta", transformers.RobertaForCausalLM, ed.TaskType.CAUSAL_LM)
        self.assertTrue(res, f"ROBERTA model Failed [ERROR {err}]")

    @staticmethod
    def compare_torch_to_jax(
        name,
        hf_out,
        ed_out,
        atol: float = 0.125,
        rtol: float = 0,
        easy_time: float | None = None,
        torch_time: float | None = None,
    ):
        jux = getattr(ed_out, "aux_loss", 0)
        tux = getattr(hf_out, "aux_loss", 0)
        to, jo = hf_out.logits.cpu().detach().numpy(), ed_out.logits
        err = jnp.mean(to) - jnp.mean(jo)
        ed_loss = (ed_out.loss - jux) if name not in ["gpt_oss"] else ed_out.loss
        hf_loss = hf_out.loss.cpu().detach().numpy()
        if STRICT_CHECK:
            np.testing.assert_allclose(to, jo, atol=0.125, rtol=0)
        all_close = jnp.allclose(to, jo, atol=atol, rtol=rtol)
        all_close_loss = jnp.allclose(hf_loss, ed_loss, atol=0.125, rtol=0)

        def color(text, color_code):
            return f"\x1b[{color_code}m{text}\x1b[0m"

        correct_percentage = jnp.mean(jnp.where(jnp.isclose(to, jo, atol=0.125, rtol=0), 1, 0))
        err = jnp.abs(to - jo).max()

        table = tabulate(
            [
                ["Last 5 elements", str(to[0, -1, -5:]), str(jo[0, -1, -5:])],
                ["Loss", str(hf_loss), str(ed_loss)],
                ["Took", str(torch_time), str(easy_time)],
                ["AUX", str(tux), str(jux)],
            ],
            headers=["Metric", "HuggingFace", "EasyDeL"],
            tablefmt="grid",
        )
        lose_close_string = color(str(all_close_loss), "32" if all_close_loss else "31")
        max_error_string = color(f"{err:.6f}", "32" if err < 1e-2 else "31")
        co_p = color(f"{correct_percentage:.2%}", "32" if correct_percentage > 0.99 else "31")
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
        np_input_ids = np.random.randint(0, vocab_size, input_shape)  # noqa
        return torch.from_numpy(np_input_ids).to(torch.long), jnp.asarray(np_input_ids, dtype="i4")

    def get_hf_model_from_hub(self, repo_id, factory=transformers.AutoModelForCausalLM):
        conf = transformers.AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        for k, v in self.__dict__.items():
            if isinstance(v, bool | str | float | type(None) | int):
                setattr(conf, k, v)
        model = type(factory.from_config(conf, trust_remote_code=True))

        return model, conf


if __name__ == "__main__":
    # unittest.main()
    print(jax.devices())
    test = EasyModelsTest()
    test.setUp()

    test.test_arctic()  # Passed - Passes MoE CONOV
    test.test_cohere()  # Passed
    test.test_cohere2()  # Passed
    test.test_dbrx()  # Passed
    test.test_deepseek_v2()  # Passed - Passes MoE CONOV
    test.test_deepseek_v3()  # Passed - Passes MoE CONOV
    test.test_exaone()  # Passed
    test.test_falcon()  # Passed
    test.test_gemma()  # Passed
    test.test_gemma2()  # Passed
    test.test_gemma3_text()  # Passed
    # test.test_gemma3()  # Passed
    test.test_glm()  # Passed
    test.test_glm4()  # Passed
    test.test_glm4_moe()  # Passed - Passes MoE CONOV
    test.test_gptj()  # Passed
    test.test_gpt_noex()  # Passed
    test.test_gpt_oss()  # Passed
    test.test_gpt2()  # Passed
    test.test_internlm2()  # Passed
    test.test_llama()  # Passed
    test.test_llama4()  # Passed
    # test.test_llama4_cond()  # Passed
    test.test_mamba()  # Passed
    test.test_mamba2()  # Passed - ReCheck
    test.test_mistral()  # Passed
    test.test_mixtral()  # Passed - Passes MoE CONOV
    test.test_mpt()  # Passed
    test.test_olmo()  # Passed
    test.test_olmo2()  # Passed
    test.test_phi()  # Passed
    test.test_phi3()  # Passed
    # test.test_phimoe()  # Failed v0.0.80 - N  Runtime
    test.test_qwen2()  # Passed
    test.test_qwen2_moe()  # Passed - Passes MoE CONOV
    # test.test_qwen2_vl()  # Passed
    test.test_qwen3()  # Passed
    test.test_qwen3_moe()  # Passed - Passes MoE CONOV

    # You may not belive this but these model have issues on HF side ;\
    # test.test_openelm()  # Passed
    # test.test_stablelm()  # Passed
    # -----------------------------------------------
