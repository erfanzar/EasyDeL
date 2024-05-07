import math

from ..easydel_modelling_utils import EasyDeLPretrainedConfig
from typing import Optional


class MambaConfig(EasyDeLPretrainedConfig):
    model_type: str = "mamba"

    def __init__(
            self,
            vocab_size=50280,
            hidden_size=768,
            state_size=16,
            num_hidden_layers=32,
            layer_norm_epsilon=1e-5,
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
            expand=2,
            conv_kernel=4,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
            initializer_range=0.1,
            residual_in_fp32=True,
            time_step_rank="auto",
            time_step_scale=1.0,
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_init_scheme="random",
            time_step_floor=1e-4,
            rescale_prenorm_residual=False,
            use_cache=True,
            gradient_checkpointing: str = "nothing_saveable",
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.intermediate_size = int(expand * self.hidden_size)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(**kwargs)

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        return super().get_partition_rules(fully_sharded_data_parallel=fully_sharded_data_parallel)

    def add_jax_args(
            self,
            gradient_checkpointing: str = "nothing_saveable"
    ):
        self.gradient_checkpointing = gradient_checkpointing
