from typing import Optional, Tuple

import flax

from ._base import ELLMMixin
from datasets import Dataset
from ...trainer import CausalLMTrainer, TrainArguments
from jax import lax, numpy as jnp


class ELLM(ELLMMixin):

    def train(
            self,
            train_dataset: Dataset,
            arguments: TrainArguments,
            eval_dataset: Optional[Dataset] = None,
            dtype: jnp.dtype = jnp.bfloat16,
            param_dtype: jnp.dtype = jnp.bfloat16,
            input_shape: Tuple[int, int] = (1, 1),
            **kwargs
    ):
        arguments.model_class = type(self.model)
        arguments.init_input_shape = input_shape
        arguments.configs_to_init_model_class = {
            'config': self.model_config,
            'dtype': dtype,
            'param_dtype': param_dtype,
            'input_shape': input_shape
        }

        for k, v in kwargs.items():
            setattr(arguments, k, v)
        trainer = CausalLMTrainer(
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            arguments=arguments,
            ckpt_path=None
        )
        return trainer.train(flax.core.FrozenDict({'params': self.params}))

    def serve(self):
        raise NotImplementedError("Serve ELLM is not Implemented yet.")
