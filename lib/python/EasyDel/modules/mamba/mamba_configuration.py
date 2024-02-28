from ..easydel_modelling_utils import EasyDelPretrainedConfig
from typing import Optional


class MambaConfig(EasyDelPretrainedConfig):
    def __int__(
            self,
            hidden_size: int = 2560,
            num_hidden_layers: int = 64,
            vocab_size: int = 50277,
            ssm_cfg: Optional[dict] = None,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,
            **kwargs
    ):
        if ssm_cfg is None:
            ssm_cfg = {}

        self.ssm_cfg = ssm_cfg
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        super().__int__(**kwargs)

    def get_partition_rules(self, fully_sharded_data_parallel: bool = True):
        return super().get_partition_rules(fully_sharded_data_parallel=fully_sharded_data_parallel)
