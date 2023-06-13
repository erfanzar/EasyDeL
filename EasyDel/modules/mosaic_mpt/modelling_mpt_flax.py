from flax import linen as nn
from flax.serialization import to_bytes, from_bytes, to_state_dict, from_state_dict
from jax import grad, jit
from transformers import FlaxPreTrainedModel, PretrainedConfig


class MPTConfig(PretrainedConfig):
    model_type = 'mpt'

    def __init__(self, d_model: int = 2048, n_heads: int = 16, n_layers: int = 24, expansion_ratio: int = 4,
                 max_seq_len: int = 2048, vocab_size: int = 50368, resid_pdrop: float = 0.0, emb_pdrop: float = 0.0,
                 learned_pos_emb: bool = True, init_device: str = 'cpu',
                 logit_scale: Optional[Union[float, str]] = None, no_bias: bool = False, verbose: int = 0,
                 embedding_fraction: float = 1.0, use_cache: bool = False,
                 init_config: Dict = init_config_defaults, **kwargs):

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction

        self.use_cache = use_cache
        self.init_config = init_config
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        super().__init__(**kwargs)
        self._validate_config()

    @staticmethod
    def _set_config_defaults(config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    @staticmethod
    def get_partition_rules():
        ...


class MptMLP(nn.Module):
    config: MPTConfig

    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class MptAttention(nn.Module):
    config: MPTConfig

    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class MptBlock(nn.Module):
    config: MPTConfig

    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class MptCollection(nn.Module):
    config: MPTConfig

    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class MptModule(nn.Module):
    config: MPTConfig

    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class MptPretrainedModel(FlaxPreTrainedModel):
    module: nn.Module = None
    config_class: MPTConfig = MPTConfig

    def __init__(self, config, _do_init: bool = False):
        super().__init__(_do_init=_do_init, config=config, module=self.module)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> Dict:
        ...

    def __call__(self, input_ids, attention_mask=None, params=None):
        ...


class MptModel(MptPretrainedModel):
    module = MptModule


class MptForCausalLMModule(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...


class MptForCausalLM(MptPretrainedModel):
    module = MptForCausalLMModule
