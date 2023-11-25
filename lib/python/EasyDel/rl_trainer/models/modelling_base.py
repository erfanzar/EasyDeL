import json
import logging
import os
import jax
# from jax import numpy as jnp, lax
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError
from transformers import FlaxPreTrainedModel
from flax import linen as nn
from flax.serialization import from_bytes
# from flax.traverse_util import flatten_dict, unflatten_dict
import msgpack
from safetensors.torch import load_file

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
    pretrained_model: FlaxPreTrainedModel
    transformers_parent_class = None
    supported_args = None
    supported_modules = ("v_head",)
    supported_rm_modules = ("score",)
    supported_pretrained_model_architectures = FlaxPreTrainedModel

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, from_pt: bool = True, *model_args, **kwargs):

        if kwargs is not None:
            reward_adapter = kwargs.pop("reward_adapter", None)
            is_trainable = kwargs.pop("is_trainable", False)
            trl_model_args, pretrained_kwargs, peft_quantization_kwargs = cls._split_kwargs(kwargs)
            token = pretrained_kwargs.get("token", None)
        else:
            reward_adapter = None
            is_trainable = False
            trl_model_args = {}
            pretrained_kwargs = {}
            token = None

        if reward_adapter is not None and not isinstance(reward_adapter, str):
            raise ValueError(
                "The `reward_adapter` argument should be "
                "a string representing the name of local path or the"
                " Hub id to the Reward Modeling adapter."
            )

        if isinstance(pretrained_model_name_or_path, str):

            remote_adapter_config = None

            local_adapter_present = os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json"))

            pretrained_model = cls.transformers_parent_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, **pretrained_kwargs
            )

        elif isinstance(pretrained_model_name_or_path, cls.supported_pretrained_model_architectures):
            pretrained_model = pretrained_model_name_or_path

        else:
            raise ValueError(
                "pretrained_model_name_or_path should be a string or a PreTrainedModel, "
                f"but is {type(pretrained_model_name_or_path)}"
            )

        model = cls(pretrained_model, **trl_model_args)
        is_resuming_training = True
        if isinstance(pretrained_model_name_or_path, str):
            safe_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

            sharded_index_filename = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
            safe_sharded_index_filename = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
            is_sharded = False
            use_safe = os.path.exists(safe_filename)

            if not (os.path.exists(filename) or os.path.exists(safe_filename)):
                filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                    pretrained_model,
                    pretrained_model_name_or_path,
                    sharded_index_filename,
                    token=token,
                )
                if filename is None and files_to_download is None:
                    safe_filename, files_to_download, is_sharded, is_resuming_training = cls._get_checkpoint_from_hub(
                        pretrained_model,
                        pretrained_model_name_or_path,
                        safe_sharded_index_filename,
                        token=token,
                        model_name="model.safetensors",
                        model_index_name="model.safetensors.index.json",
                    )
                    use_safe = True
                else:
                    use_safe = False
            if from_pt:
                loading_func = load_file
                load_kwargs = {}
            else:
                def loading_func(file_name: str, *args, **kwargs_):
                    tensors = {}
                    with open(file_name, 'rb') as stream:
                        unpacker = msgpack.Unpacker(stream, read_size=83886080, max_buffer_size=0)
                        for key, value in unpacker:
                            key = tuple(key)
                            tensor = from_bytes(None, value)
                            tensors[key] = tensor
                    return tensors

            if is_resuming_training:
                if is_sharded:
                    state_dict = {}

                    for shard_file in files_to_download:
                        filename = hf_hub_download(
                            pretrained_model_name_or_path,
                            shard_file,
                            token=token,
                        )
                        state_dict.update(loading_func(filename, **load_kwargs))
                else:
                    state_dict = loading_func(filename if not use_safe else safe_filename, **load_kwargs)

        else:
            state_dict = pretrained_model_name_or_path.state_dict()

        if from_pt:
            lw = len('.weight')
            with jax.default_device(cls._get_current_device()):
                flax_dict = {}
                for key, tensor in state_dict.items():
                    if match_keywords(key, ['kernel'], ['none']):
                        if len(tensor.shape) == 2:
                            tensor = tensor.transpose(0, 1)
                    if key.endswith('.weight'):
                        key = key[:-lw] + '.kernel'
                    key_tuple = key.split('.')
                    key_names = ()
                    tensor = tensor.detach().cpu().numpy()
                    for k in key_tuple:
                        key_names += k,
                    flax_dict[key_names] = tensor

        model.is_peft_model = False
        model.current_device = cls._get_current_device()

        if is_resuming_training:
            model.post_init(state_dict=state_dict)

        model.supports_rm_adapter = False

        return model

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
    def _get_current_device(cls):
        return jax.devices()[0]

    @classmethod
    def _split_kwargs(cls, kwargs):
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

    def save_pretrained(self, *args, **kwargs):
        state_dict = kwargs.get("state_dict")
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict

        return self.pretrained_model.save_pretrained(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Return the state_dict of the pretrained model.
        """
        raise NotImplementedError

    def post_init(self, *args, **kwargs):
        r"""
        Post initialization method. This method is called after the model is
        instantiated and loaded from a checkpoint. It can be used to perform
        additional operations such as loading the state_dict.
        """
        raise NotImplementedError

    def compute_reward_score(self, input_ids, attention_mask=None, ppo_adapter_name="default", **kwargs):

        if not self.supports_rm_adapter:
            raise ValueError("This model does not support reward modeling adapter.")

        # enable rm adapter
        self.pretrained_model.set_adapter(self.rm_adapter_name)
        self.pretrained_model.eval()

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        last_hidden_states = base_model_output.hidden_states[-1]
        scores = self.score(last_hidden_states)

        self.pretrained_model.set_adapter(ppo_adapter_name)
        self.pretrained_model.train()

        return scores
