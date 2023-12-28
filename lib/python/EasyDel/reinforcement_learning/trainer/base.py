from fjformer import with_sharding_constraint
from fjformer.func.loss_func import cross_entropy_loss_and_accuracy
from jax import numpy as jnp
import jax
from jax.sharding import PartitionSpec


def create_reinforcement_learning_casual_language_model_train_step(partition_spec=PartitionSpec(("dp", "fsdp"), "sp")):
    """
    The create_reinforcement_learning_casual_language_model_train_step function is a training step function
    that takes in the current state of the model, and a batch of data. It then calculates the loss and accuracy for
    this batch, and returns an updated state with new parameters based on these gradients.

    :param partition_spec: Specify which devices the model will be split across
    :return: A reinforcement_learning_casual_language_model_train_step function that takes in the current state
    of the model,

    """

    def reinforcement_learning_casual_language_model_train_step(state, batch):
        """
        The casual_language_model_train_step function is a training step function that takes in the current state
        of the model, and a batch of data. It then calculates the loss and accuracy for this batch,
        and returns an updated state with new parameters based on these gradients.

        :param state: Store the model parameters
        :param batch: Pass the data to the model
        :return: A tuple of (state, loss, accuracy)

        """
        batch = with_sharding_constraint(batch, partition_spec)

        def calculate_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(params=params, **batch,
                                    return_dict=True).logits

            loss, accuracy = cross_entropy_loss_and_accuracy(
                logits[:, :-1, :], labels, batch["attention_mask"].astype(jnp.float32)[:, 1:]
            )
            return loss, accuracy

        grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
        (loss__, accuracy__), grad = grad_fn(state.params)
        state = state.apply_gradients(grads=grad)
        return state, loss__, accuracy__

    return reinforcement_learning_casual_language_model_train_step
