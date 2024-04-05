import chex
import jax

from typing import Optional, Dict, Union, Any, List
from jax import numpy as jnp
from flax.struct import dataclass
from contextlib import contextmanager


def pad_to_length(tensor: chex.Array, length: int, pad_value: Union[int, float], axis: int = -1) -> chex.Array:
    if tensor.shape[axis] >= length:
        if tensor.ndim == 2:
            tensor = tensor[:, :length]
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[axis] = length - tensor.shape[axis]
        return jax.numpy.concatenate(
            [
                tensor,
                pad_value * jax.numpy.ones(pad_size, dtype=tensor.dtype),
            ],
            axis=axis,
        )


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    :param pad_token_id: int: The tokenizers pad_token_id.
    :param label_pad_token_id: int: The label used for masking.
    :param is_encoder_decoder: Optional[bool]: Whether you model has an encoder_decoder architecture
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [jnp.array(ex[k], dtype="i4") for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value).astype("i4")
                else:
                    if "prompt" in k:
                        to_pad = [jnp.array(ex[k][::-1], dtype="i4") for ex in features]
                    else:
                        to_pad = [jnp.array(ex[k], dtype="i4") for ex in features]
                    if k.endswith("_input_ids"):
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value).astype("i4")
                    if "prompt" in k:
                        padded_batch[k] = jnp.flip(padded_batch[k], axis=[1])
            elif k.endswith("_logps"):
                padded_batch[k] = jnp.array([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]
        return padded_batch


def pad_sequence(
        sequences,
        batch_first=False,
        padding_value=0,
        max_len: int | None = None
):
    max_len = max(seq.shape[-1] for seq in sequences) if max_len is None else max_len
    padding_value = jnp.array(padding_value).reshape(1)
    if batch_first:
        padded_seqs = [
            jnp.concatenate(
                [
                    seq.reshape(1, -1),
                    jnp.ones((1, max_len - seq.shape[-1])) * padding_value
                ],
                axis=1
            ) if seq.shape[-1] < max_len else seq.reshape(1, -1)
            for seq in sequences
        ]
    else:
        padded_seqs = [
            jnp.concatenate(
                [
                    jnp.ones((1, max_len - seq.shape[-1])) * padding_value,
                    seq.reshape(1, -1)
                ],
                axis=1
            ) if seq.shape[-1] < max_len else seq.reshape(1, -1)
            for seq in sequences
        ]

    return jnp.array(padded_seqs)


@contextmanager
def leave_alone_context_manager():
    # Perform setup actions (none in this case)
    yield
