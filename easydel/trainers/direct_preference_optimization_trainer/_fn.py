# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import typing as tp

import chex
import flax
import flax.nnx
import jax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.nn import log_sigmoid as logsigmoid
from jax.nn import relu, sigmoid
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully
from ..utils import pad_to_length

# Define allowed loss function variants.
LOSS_FN_VARIENTS = tp.Literal[
    "sigmoid",
    "hinge",
    "ipo",
    "exo_pair",
    "nca_pair",
    "robust",
    "bco_pair",
    "sppo_hard",
    "aot",
    "aot_pair",
    "apo_zero",
    "apo_down",
]


def concatenated_inputs(
    batch: dict[str, list | chex.Array],
    padding_value: int,
) -> dict[str, chex.Array]:
    """
    Concatenates chosen and rejected examples from the batch, and pads the inputs to a uniform length.

    This function is used to merge paired inputs (e.g. chosen vs. rejected examples)
    so that the model can process them in one forward pass. It concatenates the prompt inputs,
    attention masks, and (if present) image-related arrays. The completion inputs (and their attention masks)
    are padded to the length of the longest completion among the chosen and rejected examples.

    Args:
        batch (tp.Dict[str, tp.Union[tp.List, chex.Array]]):
            A dictionary containing the batch of data. Expected keys include:
            - "prompt_input_ids", "prompt_attention_mask"
            - "chosen_input_ids", "rejected_input_ids"
            - "chosen_attention_mask", "rejected_attention_mask"
            Optionally, keys like "pixel_values", "pixel_attention_mask", and "image_sizes" may be present.
        padding_value (int): The padding value to use when padding completion inputs.

    Returns:
        tp.Dict[str, chex.Array]: A dictionary with concatenated arrays under keys such as:
            - "prompt_input_ids", "prompt_attention_mask"
            - "completion_input_ids", "completion_attention_mask"
            and optionally image-related keys.
    """
    output = {}
    # Concatenate the prompt-related arrays (duplicated for chosen and rejected).
    output["prompt_input_ids"] = jnp.concatenate(
        [batch["prompt_input_ids"], batch["prompt_input_ids"]],
        axis=0,
    )
    output["prompt_attention_mask"] = jnp.concatenate(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]],
        axis=0,
    )
    if "pixel_values" in batch:
        output["pixel_values"] = jnp.concatenate(
            [batch["pixel_values"], batch["pixel_values"]],
            axis=0,
        )
    if "pixel_attention_mask" in batch:
        output["pixel_attention_mask"] = jnp.concatenate(
            [batch["pixel_attention_mask"], batch["pixel_attention_mask"]],
            axis=0,
        )
    if "image_sizes" in batch:
        output["image_sizes"] = jnp.concatenate(
            [batch["image_sizes"], batch["image_sizes"]],
            axis=0,
        )

    # Determine maximum length for the completion inputs.
    max_completion_length = max(
        batch["chosen_input_ids"].shape[1],
        batch["rejected_input_ids"].shape[1],
    )
    # Pad chosen and rejected completion input IDs to the same length and concatenate them.
    output["completion_input_ids"] = jnp.concatenate(
        (
            pad_to_length(
                batch["chosen_input_ids"],
                max_completion_length,
                pad_value=padding_value,
            ),
            pad_to_length(
                batch["rejected_input_ids"],
                max_completion_length,
                pad_value=padding_value,
            ),
        ),
    )
    # Similarly pad and concatenate the attention masks.
    output["completion_attention_mask"] = jnp.concatenate(
        (
            pad_to_length(
                batch["chosen_attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
            pad_to_length(
                batch["rejected_attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
        ),
    )

    return output


def get_loss_function(
    loss_type: LOSS_FN_VARIENTS,
    beta: float,
    label_smoothing: float | int,
):
    """
    Returns a loss function based on the specified loss type.

    This function maps a given loss type (e.g., "sigmoid", "hinge", "ipo", etc.)
    to a corresponding loss function implementation that computes the DPO (Direct Preference Optimization) loss.

    Args:
        loss_type (LOSS_FN_VARIENTS): The type of loss function to return.
        beta (float): A scaling factor applied to the loss computation.
        label_smoothing (tp.Union[float, int]): A value for label smoothing used in some loss functions.

    Returns:
        A callable loss function that accepts arguments:
            (chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, label_smoothing, **kwargs)
        and returns the computed loss.
    """

    def _base_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """
        Base computation for DPO loss.

        Computes the log ratios between chosen and rejected log probabilities, and similarly for reference values.

        Args:
            chosen_logps (chex.Array): Log probabilities for chosen examples.
            rejected_logps (chex.Array): Log probabilities for rejected examples.
            ref_chosen_logps (chex.Array): Reference log probabilities for chosen examples.
            ref_rejected_logps (chex.Array): Reference log probabilities for rejected examples.
            beta (float): Scaling factor.
            label_smoothing (float): Label smoothing factor.
            **kwargs: Additional arguments (ignored).

        Returns:
            A tuple of (logits, logratios, ref_logratios) where:
                logits = logratios - ref_logratios.
        """
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = logratios - ref_logratios
        return logits, logratios, ref_logratios

    def _sigmoid_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> chex.Array:
        """
        Computes the DPO loss using a sigmoid-based formulation.

        Args:
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps (chex.Array):
                Log probabilities for chosen/rejected examples and their reference values.
            beta (float): Scaling factor.
            label_smoothing (float): Label smoothing factor.
            **kwargs: Additional arguments (ignored).

        Returns:
            The computed loss as a negative weighted log sigmoid.
        """
        logits, _, _ = _base_dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )
        return -(
            jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
            + jax.nn.log_sigmoid(-beta * logits) * label_smoothing
        )

    def _nca_pair_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> chex.Array:
        """
        Computes the DPO loss using an NCA pair formulation.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on the NCA pair loss formulation.
        """
        chosen_rewards = (chosen_logps - ref_chosen_logps) * beta
        rejected_rewards = (rejected_logps - ref_rejected_logps) * beta
        return -(
            jax.nn.log_sigmoid(chosen_rewards)
            + 0.5 * jax.nn.log_sigmoid(-chosen_rewards)
            + 0.5 * jax.nn.log_sigmoid(-rejected_rewards)
        )

    def _aot_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> chex.Array:
        """
        Computes the DPO loss using the AOT (All Ordered Terms) loss formulation.

        This loss function sorts the log ratios and compares them with the sorted reference log ratios.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on sorted differences.
        """
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logratios_sorted = jnp.sort(logratios, axis=0)
        ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)
        delta = logratios_sorted - ref_logratios_sorted
        return -(
            jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
            + jax.nn.log_sigmoid(-beta * delta) * label_smoothing
        )

    def _discopop_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
        discopop_tau: float = 1.0,
        **kwargs,
    ) -> chex.Array:
        """
        Computes the DPO loss using a Discopo-based modulation.

        Args:
            discopop_tau (float): Temperature parameter for modulation.
            (Other arguments are as described above.)

        Returns:
            The computed loss with a logistic and exponential modulation.
        """
        logits, _, _ = _base_dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )
        logits = logits * beta
        log_ratio_modulation = jax.nn.sigmoid(logits / discopop_tau)
        logistic_component = -jax.nn.log_sigmoid(logits)
        exp_component = jnp.exp(-logits)
        return logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

    def _hinge_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the hinge loss version of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The hinge loss computed as the ReLU of (1 - beta * logits).
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return relu(1 - beta * logits)

    def _ipo_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the IPO loss variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            A squared loss computed from the logits with a bias term.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return (logits - 1 / (2 * beta)) ** 2

    def _kto_pair_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the KTO pair loss variant.

        Args:
            (Same as above.)

        Returns:
            The loss computed using the log-sigmoid function.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return -logsigmoid(beta * logits) * (1 - label_smoothing) - logsigmoid(-beta * logits) * label_smoothing

    def _robust_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes a robust variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The loss computed with an adjustment that involves dividing by (1 - 2 * label_smoothing).
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return (-logsigmoid(beta * logits) * (1 - label_smoothing) + logsigmoid(-beta * logits) * label_smoothing) / (
            1 - 2 * label_smoothing
        )

    def _exo_pair_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the exo-pair variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The computed loss combining sigmoid and log-sigmoid terms with label smoothing.
        """
        import math

        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        label_smoothing = jnp.maximum(label_smoothing, 1e-3)
        return sigmoid(beta * logits) * (logsigmoid(beta * logits) - math.log(1 - label_smoothing)) + sigmoid(
            -beta * logits
        ) * (logsigmoid(-beta * logits) - math.log(label_smoothing))

    def _bco_pair_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the BCO pair variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The loss computed from the log-ratios of chosen and rejected rewards.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios
        delta = jnp.mean(jnp.concatenate([chosen_rewards, rejected_rewards]))
        return -logsigmoid((beta * chosen_logratios) - delta) - logsigmoid(-(beta * rejected_logratios - delta))

    def _sppo_hard_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the SPO PPO hard variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            A squared loss combining the differences for chosen and rejected examples.
        """
        a = chosen_logps - ref_chosen_logps
        b = rejected_logps - ref_rejected_logps
        return (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2

    def _aot_pair_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the AOT pair variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The loss computed from the sorted differences between chosen and rejected log ratios.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_logratios_sorted = jnp.sort(chosen_logratios, axis=0)
        rejected_logratios_sorted = jnp.sort(rejected_logratios, axis=0)
        delta = chosen_logratios_sorted - rejected_logratios_sorted
        return -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing

    def _aot_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the AOT variant of the DPO loss.

        This is similar to _aot_pair_dpo_loss but may be used when the pair version is not required.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on the differences of sorted log ratios.
        """
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logratios_sorted = jnp.sort(logratios, axis=0)
        ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)
        delta = logratios_sorted - ref_logratios_sorted
        return -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing

    def _apo_zero_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the APO zero variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on the sigmoid of the log ratios.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        losses_chosen = 1 - sigmoid(beta * chosen_logratios)
        losses_rejected = sigmoid(beta * rejected_logratios)
        return losses_chosen + losses_rejected

    def _apo_down_dpo_loss(
        chosen_logps: chex.Array,
        rejected_logps: chex.Array,
        ref_chosen_logps: chex.Array,
        ref_rejected_logps: chex.Array,
        beta: float,
        label_smoothing: float,
    ) -> chex.Array:
        """
        Computes the APO down variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on an alternative weighting of the chosen and rejected log ratios.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        losses_chosen = sigmoid(beta * chosen_logratios)
        losses_rejected = 1 - sigmoid(beta * (chosen_logratios - rejected_logratios))
        return losses_chosen + losses_rejected

    # Map loss_type strings to corresponding loss function implementations.
    loss_function = {
        "ipo": _ipo_dpo_loss,
        "kto": _kto_pair_dpo_loss,
        "hinge": _hinge_dpo_loss,
        "sigmoid": _sigmoid_dpo_loss,
        "robust": _robust_dpo_loss,
        "exo_pair": _exo_pair_dpo_loss,
        "bco_pair": _bco_pair_dpo_loss,
        "sppo_hard": _sppo_hard_dpo_loss,
        "nca_pair": _nca_pair_dpo_loss,
        "aot_pair": _aot_pair_dpo_loss,
        "aot": _aot_dpo_loss,
        "apo_zero": _apo_zero_dpo_loss,
        "apo_down": _apo_down_dpo_loss,
        "discopop": _discopop_dpo_loss,
    }.get(loss_type, None)
    assert loss_function is not None, f"given loss_type({loss_function}) is not valid"
    return loss_function


def concatenated_forward(
    model: EasyDeLBaseModule,
    batch: dict[str, list | chex.Array],
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: str = "keep_end",
    aux_loss_enabled: bool = False,
    loss_type: str = "sigmoid",
) -> dict[str, chex.Array]:
    """
    Runs the model on concatenated chosen/rejected inputs for efficiency.

    This function first concatenates inputs (using the `concatenated_inputs` function) and then runs
    a forward pass through the model. It handles both encoder-decoder and decoder-only architectures,
    applies truncation if required, and computes per-token log probabilities.

    Args:
        model (EasyDeLBaseModule): The model to run.
        batch (tp.Dict[str, tp.Union[tp.List, chex.Array]]): The input batch of data.
        is_encoder_decoder (bool): Flag indicating whether the model is an encoder-decoder.
        label_pad_token_id (int): Token id used to mark padded tokens in the labels.
        padding_value (int): Padding value for inputs.
        max_length (int | None, optional): Maximum sequence length for truncation. Defaults to None.
        truncation_mode (str, optional): Truncation strategy ("keep_end" or "keep_start"). Defaults to "keep_end".
        aux_loss_enabled (bool, optional): If True, enables auxiliary loss computation. Defaults to False.
        loss_type (str, optional): The type of loss function to be used. Defaults to "sigmoid".

    Returns:
        tp.Dict[str, chex.Array]: A dictionary containing:
            - "chosen_logps": Log probabilities for chosen examples.
            - "rejected_logps": Log probabilities for rejected examples.
            - "mean_chosen_logits": Mean logits over tokens for chosen examples.
            - "mean_rejected_logits": Mean logits over tokens for rejected examples.
            Optionally, if `aux_loss_enabled` is True and the model output contains "aux_loss",
            it is included in the output dictionary.
    """
    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = concatenated_inputs(batch=batch, padding_value=padding_value)

    model_kwargs = {}
    if aux_loss_enabled:
        model_kwargs["output_router_logits"] = True

    # Include image-related inputs if available.
    if "pixel_values" in concatenated_batch:
        model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
    if "pixel_attention_mask" in concatenated_batch:
        model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
    if "image_sizes" in concatenated_batch:
        model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    if is_encoder_decoder:
        # For encoder-decoder models, use completion inputs as labels.
        labels = completion_input_ids
        labels = jnp.where(
            completion_attention_mask == 0,
            label_pad_token_id,
            completion_input_ids,
        )
        outputs = model(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            labels=labels,
            **model_kwargs,
        )
        logits = outputs.logits
        loss_mask = completion_attention_mask.astype(bool)
    else:
        # For decoder-only models, concatenate prompt and completion.
        input_ids = jnp.concatenate(
            [prompt_input_ids, completion_input_ids],
            axis=1,
        )
        attention_mask = jnp.concatenate(
            [prompt_attention_mask, completion_attention_mask],
            axis=1,
        )
        loss_mask = jnp.concatenate(
            [
                jnp.zeros_like(prompt_attention_mask),
                completion_attention_mask,
            ],
            axis=1,
        )
        if max_length is not None:
            if truncation_mode == "keep_end":
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
                loss_mask = loss_mask[:, -max_length:]
            elif truncation_mode == "keep_start":
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length]
                loss_mask = loss_mask[:, :max_length]
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                )
        model_kwargs["input_ids"] = input_ids
        model_kwargs["attention_mask"] = attention_mask
        outputs = model(**model_kwargs)
        logits = outputs.logits
        labels = jnp.roll(input_ids, shift=-1, axis=1)
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype("bool")

    # Adjust logits shape if necessary.
    if logits.shape[:2] != labels.shape[:2]:
        seq_len = labels.shape[1]
        logits = logits[:, -seq_len:]

    labels = jnp.where(loss_mask, labels, 0)
    lsmax = jax.nn.log_softmax(logits, axis=-1)
    batch_size, seq_len = labels.shape
    per_token_logps = jnp.roll(
        jnp.where(
            loss_mask,
            lsmax[jnp.arange(batch_size)[:, None], jnp.arange(seq_len)[None, :], labels],
            0,
        ),
        shift=1,
        axis=1,
    )
    all_logps = per_token_logps.sum(-1)

    # Special handling for "ipo" loss type.
    if loss_type == "ipo":
        all_logps = all_logps / loss_mask.sum(-1)
    output = {}
    output["chosen_logps"] = all_logps[:num_examples]
    output["rejected_logps"] = all_logps[num_examples:]

    mean_chosen_logits = jnp.sum(
        jnp.where(
            loss_mask[:num_examples, :, None],
            logits[:num_examples],
            0,
        )
    ) / jnp.sum(loss_mask[:num_examples])
    mean_rejected_logits = jnp.sum(
        jnp.where(
            loss_mask[num_examples:, :, None],
            logits[num_examples:],
            0,
        )
    ) / jnp.sum(loss_mask[num_examples:])
    output["mean_chosen_logits"] = mean_chosen_logits
    output["mean_rejected_logits"] = mean_rejected_logits

    if aux_loss_enabled and hasattr(outputs, "aux_loss"):
        output["aux_loss"] = outputs.aux_loss
    return output


def training_step(
    state: EasyDeLState,
    batch: dict,
    reference_state: EasyDeLState,
    learning_rate_fn: tp.Callable,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: LOSS_FN_VARIENTS = "sigmoid",
    reference_free: bool = False,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[EasyDeLState, LossMetrics]:
    """
    Performs a single training step.

    This function computes gradients via minibatch processing over the input batch,
    calculates the loss using a specified loss function, updates the model state,
    and returns the updated state along with loss metrics.

    Args:
        state (EasyDeLState): The current model state.
        batch (dict): Input batch data.
        reference_state (EasyDeLState): A reference model state used for computing reference log probabilities.
        learning_rate_fn (tp.Callable): Function to compute the learning rate.
        concatenated_forward (tp.Callable): Function to perform a forward pass on concatenated inputs.
        beta (float, optional): Scaling factor for loss computation. Defaults to 0.1.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.
        loss_type (LOSS_FN_VARIENTS, optional): Type of loss function to use. Defaults to "sigmoid".
        ref_precalculated (bool, optional): If True, uses precalculated reference log probabilities from the batch.
            Defaults to True.
        loss_config (tp.Optional[LossConfig], optional): Additional configuration for loss. Defaults to None.
        partition_spec (tp.Optional[PartitionSpec], optional): Partitioning specification for sharding the batch.
            Defaults to None.
        gradient_accumulation_steps (int, optional): Number of steps for gradient accumulation. Defaults to 1.

    Returns:
        tp.Tuple[EasyDeLState, LossMetrics]: A tuple containing the updated model state and the loss metrics.
    """
    batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )

    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)
    _loss_func = get_loss_function(
        loss_type=loss_type,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    def calculate_loss(tree: flax.nnx.GraphState, call_batch):
        """
        Inner function to compute loss and metrics for a given minibatch.

        Args:
            tree (flax.nnx.GraphState): The current model graph state.
            call_batch (dict): A minibatch of data.

        Returns:
            A tuple (loss, metrics) where loss is a scalar and metrics is a LossMetrics instance.
        """
        model_output = concatenated_forward(state.merge(tree=tree), call_batch)

        if "ref_chosen_logps" in call_batch and "ref_rejected_logps" in call_batch:
            ref_chosen_logps = jax.lax.stop_gradient(call_batch["ref_chosen_logps"])
            ref_rejected_logps = jax.lax.stop_gradient(call_batch["ref_rejected_logps"])
        else:
            rfm = reference_state.model
            rfm.eval()
            out = jax.lax.stop_gradient(concatenated_forward(rfm, call_batch))
            ref_chosen_logps = out["chosen_logps"]
            ref_rejected_logps = out["rejected_logps"]

        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        losses = _loss_func(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )

        chosen_rewards = beta * jax.lax.stop_gradient(chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * jax.lax.stop_gradient(rejected_logps - ref_rejected_logps)
        if hasattr(model_output, "aux_loss"):
            losses += model_output["aux_loss"]

        metrics = LossMetrics(
            loss=losses.mean(),
            rejected_rewards=rejected_rewards,
            chosen_rewards=chosen_rewards,
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
    )

    metrics = update_metrics(
        metrics=metrics,
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=gradients,
    )
    state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=loss_config,
        metrics=metrics,
    )
    return (state, metrics)


def evaluation_step(
    state: EasyDeLState,
    batch: dict,
    reference_state: EasyDeLState,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: LOSS_FN_VARIENTS = "sigmoid",
    reference_free: bool = False,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """
    Performs a single evaluation step.

    This function computes loss metrics for the input batch using the provided model state.
    It can optionally use a reference state to compute reference log probabilities.

    Args:
        state (EasyDeLState): The current model state.
        batch (dict): Input batch data.
        concatenated_forward (tp.Callable): Function to perform a forward pass on concatenated inputs.
        reference_state (EasyDeLState, optional): A reference model state. Defaults to None.
        beta (float, optional): Scaling factor for loss computation. Defaults to 0.1.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.
        loss_type (LOSS_FN_VARIENTS, optional): Type of loss function to use. Defaults to "sigmoid".
        reference_free (bool, optional): If True, ignores reference log probabilities. Defaults to False.
        partition_spec (tp.Optional[PartitionSpec], optional): Partitioning specification for sharding the batch.
            Defaults to None.

    Returns:
        LossMetrics: The computed loss metrics.
    """
    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )

    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)
    _loss_func = get_loss_function(
        loss_type=loss_type,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    def calculate_loss(tree: flax.nnx.GraphState):
        """
        Inner function to compute loss metrics for evaluation.

        Args:
            tree (flax.nnx.GraphState): The current model graph state.

        Returns:
            LossMetrics: The computed loss metrics.
        """
        (
            mean_chosen_logits,
            mean_rejected_logits,
            _,
            _,
        ) = concatenated_forward(state.merge(tree), batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            if reference_state is None:
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,
                    _,
                ) = concatenated_forward(state.model, batch)
            else:
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,
                    _,
                ) = concatenated_forward(reference_state.model, batch)

        pi_log_ratios = mean_chosen_logits - mean_rejected_logits

        if reference_free:
            ref_log_ratios = 0
        else:
            ref_log_ratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_log_ratios - ref_log_ratios
        losses = _loss_func(
            logits,
            mean_chosen_logits,
            ref_chosen_logps,
            mean_rejected_logits,
            ref_rejected_logps,
        )
        chosen_rewards = beta * (mean_chosen_logits - ref_chosen_logps)
        rejected_rewards = beta * (mean_rejected_logits - ref_rejected_logps)
        metrics = LossMetrics(
            loss=losses.mean(),
            rejected_rewards=rejected_rewards,
            chosen_rewards=chosen_rewards,
        )
        return metrics

    metrics = calculate_loss(state.graphstate)
    return metrics
