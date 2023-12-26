import json
import logging
import os

import chex
import flax.core
import jax
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError
from transformers import FlaxPreTrainedModel
from flax import linen as nn
from flax.serialization import from_bytes
import msgpack
from typing import Sequence, Optional, Type, Tuple

from ...modules.auto_models import AutoEasyDelModelForCausalLM

LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


class FlaxPreTrainedModelWrapper(nn.Module):
    pretrained_model: Type[FlaxPreTrainedModel]
    supported_modules = ("v_head",)
    supported_rm_modules = ("score",)
    supported_pretrained_model_architectures = FlaxPreTrainedModel

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            device=jax.devices('cpu')[0],
            dtype: jax.numpy.dtype = jax.numpy.float32,
            param_dtype: jax.numpy.dtype = jax.numpy.float32,
            precision: jax.lax.Precision = jax.lax.Precision("fastest"),
            sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
            sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            q_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            k_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            v_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            b_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), None, None, None),
            a_ps: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            use_shard_map: bool = False,
            input_shape: Tuple[int, int] = (1, 1),
            backend: Optional[str] = None,
            **kwargs
    ):

        model, params = AutoEasyDelModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            sharding_axis_dims=sharding_axis_dims,
            sharding_axis_names=sharding_axis_names,
            q_ps=q_ps,
            k_ps=k_ps,
            v_ps=v_ps,
            b_ps=b_ps,
            a_ps=a_ps,
            use_shard_map=use_shard_map,
            input_shape=input_shape,
            backend=backend,
            **kwargs
        )
        model = cls(pretrained_model=model)
        is_resuming_training = "v_head" in params.keys()
        if not is_resuming_training:
            params = model.post_init(
                params=params,
                input_shape=input_shape,
                head_name="v_head"
            )
        return model, params

    @classmethod
    def _get_checkpoint_from_hub(
            cls,
            pretrained_model,
            pretrained_model_name_or_path,
            index_filename,
            token=None,
            model_name="pytorch_model.bin",
            model_index_name="pytorch_model.bin.index.json",
    ):
        """
        The _get_checkpoint_from_hub function is used to download a pretrained model from the Hugging Face Hub.
        It will first attempt to download the entire model, and if that fails it will try downloading just the v_head weights.
        If neither of those attempts succeed, it will return None for all outputs.

        :param cls: Specify the class of the model
        :param pretrained_model: Load the pretrained model
        :param pretrained_model_name_or_path: Load the pretrained model from a checkpoint
        :param index_filename: Load the index file for sharded models
        :param token: Authenticate with the hugging face model hub
        :param model_name: Specify the name of the model file to be downloaded
        :param model_index_name: Specify the name of the index file
        :param : Load the pretrained model
        :return: A tuple of four elements:
        
        """
        files_to_download = None
        filename = None
        is_resuming_training = True
        is_sharded = False

        try:
            filename = hf_hub_download(
                pretrained_model_name_or_path,
                model_name,
                token=token,
            )
        # sharded
        except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError):
            index_file_name = ''
            if os.path.exists(index_filename):
                index_file_name = index_filename
            else:
                try:
                    index_file_name = hf_hub_download(
                        pretrained_model_name_or_path,
                        model_index_name,
                        token=token,
                    )
                except (EntryNotFoundError, LocalEntryNotFoundError, HFValidationError):
                    # not continue training, do not have v_head weight
                    is_resuming_training = False
                    logging.warning(
                        f"A {type(pretrained_model)} model is loaded from '{pretrained_model_name_or_path}', "
                        f"and no v_head weight is found. This IS expected if you are not resuming PPO training."
                    )
            # load json
            if is_resuming_training:
                with open(index_file_name, "r") as f:
                    index = json.load(f)
                files_to_download = set()
                for k, v in index["weight_map"].items():
                    if any([module in k for module in cls.supported_modules]):
                        files_to_download.add(v)
                is_sharded = True

        return filename, files_to_download, is_sharded, is_resuming_training

    @classmethod
    def _split_kwargs(cls, kwargs):
        """
        The _split_kwargs function is used to split the kwargs into three categories:
            1. supported_kwargs - These are the arguments that are supported by this class and will be passed on to the
            parent class.
            2. unsupported_kwargs - These are arguments that aren't supported by this class, but may be useful for other
             classes in a chain of inheritance (e.g., if you're using multiple mixins).
            3. peft_kwargs - These are arguments specific to PEFT and will not be passed on to any other classes.

        :param cls: Refer to the class itself
        :param kwargs: Pass keyword arguments to the function
        :return: A tuple of three dictionaries
        
        """
        supported_kwargs = {}
        unsupported_kwargs = {}
        peft_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.supported_args:
                supported_kwargs[key] = value
            else:
                unsupported_kwargs[key] = value

        return supported_kwargs, unsupported_kwargs, peft_kwargs

    def push_to_hub(self, *args, **kwargs):
        raise NotImplementedError

    def post_init(
            self,
            params: dict | flax.core.FrozenDict,
            input_shape: Tuple[int, int],
            head_name: str = "v_head"
    ):
        r"""
        Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError

    def compute_reward_score(
            self,
            pretrained_params: dict | flax.core.FrozenDict,
            input_ids: chex.Array = None,
            attention_mask: chex.Array = None,
            ppo_adapter_name="default",
            **kwargs
    ):

        """
        The compute_reward_score function is used to compute the reward score for a given input.
        The function takes in an input_ids tensor and returns a tensor of scores. The shape of the returned
        tensor will be (batch_size, sequence_length). The higher the score, the more likely that token should be kept.

        :param self: Represent the instance of the class
        :param pretrained_params: dict | flax.core.FrozenDict: parameters to be passed to the model
        :param input_ids: Pass the input tokens to the model
        :param attention_mask: Indicate which tokens are padding
        :param ppo_adapter_name: Set the adapter back to its original state
        :param kwargs: Pass a variable number of arguments to a function
        :return: The scores for the given input_ids
        
        """
        if not self.supports_rm_adapter:
            raise ValueError("This model does not support reward modeling adapter.")

        base_model_output = self.pretrained_model(
            params=pretrained_params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        last_hidden_states = base_model_output.hidden_states[-1]
        scores = self.score(last_hidden_states)

        return scores
