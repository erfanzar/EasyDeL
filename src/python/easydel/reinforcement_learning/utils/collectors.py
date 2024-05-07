from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from ..core import pad_sequence
from jax import numpy as jnp


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
