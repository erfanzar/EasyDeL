from fjformer.func.loss_func import cross_entropy_loss_and_accuracy

import jax
from jax.sharding import PartitionSpec
from jax import numpy as jnp
from fjformer import (
    with_sharding_constraint
)
import chex
from ...etils.easystate import EasyDeLState
from flax.struct import dataclass


@dataclass
class VisionCausalLanguageModelStepOutput:
    loss: chex.Array
    text_loss: chex.Array
    text_accuracy: chex.Array
    vision_loss: chex.Array
    vision_accuracy: chex.Array


def create_vision_casual_language_model_train_step(partition_spec=PartitionSpec(("dp", "fsdp"), "sp")):
    """
    The create_vision_casual_language_model_train_step function is a training step function that takes in the current
     state of the model,and a batch of data. It then calculates the loss and accuracy for this batch, and returns
    an updated state with new parameters based on these gradients.

    :param partition_spec: Specify which devices the model will be split across
    :return: A casual_language_model_train_step function that takes in the current state of the model,

    """

    def vision_casual_language_model_train_step(state, batch) -> [
        EasyDeLState,
        chex.Array,
        VisionCausalLanguageModelStepOutput
    ]:
        """
        The vision_casual_language_model_train_step function is a training step function that takes in the current state
        of the model and a batch of data. It then calculates the loss and accuracy for this batch,
        and returns an updated state with new parameters based on these gradients.

        :param state: Store the model parameters
        :param batch: Pass the data to the model
        :return: A tuple of (state, loss, VisionCausalLanguageModelStepOutput)

        """
        batch = with_sharding_constraint(batch, partition_spec)

        def calculate_loss(params):
            labels = batch.get("labels", None)
            if labels is None:
                labels = batch["input_ids"][..., 1:]
            else:
                labels = labels[..., 1:]
            label_vision_mask = batch.pop("label_vision_mask")

            model_outputs = state.apply_fn(params=params, **batch, return_dict=True)
            logits = model_outputs.logits
            aux_loss = getattr(model_outputs, "aux_loss", None)

            vision_loss, vision_accuracy = cross_entropy_loss_and_accuracy(
                logits[:, :-1, :],
                jnp.where(label_vision_mask, labels, 0),
                batch["attention_mask"].astype(jnp.float32)[:, 1:] * label_vision_mask
            )
            text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
                logits[:, :-1, :],
                jnp.where(label_vision_mask, 0, labels),
                batch["attention_mask"].astype(jnp.float32)[:, 1:] * (1.0 - label_vision_mask)
            )

            loss = 0.5 * (vision_loss + text_loss + (aux_loss if aux_loss is not None else 0.))

            return loss, VisionCausalLanguageModelStepOutput(
                loss=loss,
                text_accuracy=text_accuracy,
                vision_accuracy=vision_accuracy,
                text_loss=text_loss,
                vision_loss=vision_loss
            )

        grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
        (loss__, metrics), grad = grad_fn(state.params)
        state = state.apply_gradients(grads=grad)
        return state, loss__, metrics

    return vision_casual_language_model_train_step


def create_vision_casual_language_model_evaluation_step(partition_spec=PartitionSpec(("dp", "fsdp"), "sp")):
    """
    The create_vision_casual_language_model_evaluation_step function is used to create a function that calculates the
     loss and accuracy of a model. It takes in a set of parameters, which are then passed into the state.apply_fn function
    to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated from these
    logits.

    :param partition_spec: Specify the partitioning of the model parameters
    :return: A function that can be used to calculate the loss and accuracy of a model

    """

    def vision_casual_language_model_evaluation_step(state, batch) -> [
        EasyDeLState,
        chex.Array,
        VisionCausalLanguageModelStepOutput
    ]:
        """
        The vision_casual_language_model_train_step function is a training step function that takes in the current state
        of the model and a batch of data. It then calculates the loss and accuracy for this batch,
        and returns an updated state with new parameters based on these gradients.

        :param state: Store the model parameters
        :param batch: Pass the data to the model
        :return: A tuple of (state, loss, VisionCausalLanguageModelStepOutput)

        """
        batch = with_sharding_constraint(batch, partition_spec)

        def calculate_loss(params):
            labels = batch.get("labels", None)
            if labels is None:
                labels = batch["input_ids"][..., 1:]
            else:
                labels = labels[..., 1:]
            label_vision_mask = batch.pop("label_vision_mask")
            model_outputs = state.apply_fn(params=params, **batch, return_dict=True)
            logits = model_outputs.logits
            aux_loss = getattr(model_outputs, "aux_loss", None)

            vision_loss, vision_accuracy = cross_entropy_loss_and_accuracy(
                logits[:, :-1, :],
                jnp.where(label_vision_mask, labels, 0),
                batch["attention_mask"].astype(jnp.float32)[:, 1:] * label_vision_mask
            )
            text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
                logits[:, :-1, :],
                jnp.where(label_vision_mask, 0, labels),
                batch["attention_mask"].astype(jnp.float32)[:, 1:] * (1.0 - label_vision_mask)
            )

            loss = 0.5 * (vision_loss + text_loss + (aux_loss if aux_loss is not None else 0.))

            return loss, VisionCausalLanguageModelStepOutput(
                loss=loss,
                text_accuracy=text_accuracy,
                vision_accuracy=vision_accuracy,
                text_loss=text_loss,
                vision_loss=vision_loss
            )

        loss__, metrics = calculate_loss(state.params)
        return loss__, metrics

    return vision_casual_language_model_evaluation_step
