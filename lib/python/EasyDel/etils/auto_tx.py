import fjformer.optimizers
import jax
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from .etils import (
    EasyDelSchedulers,
    EasyDelOptimizers,
    AVAILABLE_SCHEDULERS,
    AVAILABLE_OPTIMIZERS
)


def get_optimizer_and_scheduler(
        optimizer: AVAILABLE_OPTIMIZERS,
        scheduler: AVAILABLE_SCHEDULERS,
        steps: int,
        learning_rate: float = 1e-5,
        learning_rate_end: float = 1e-5,
        gradient_accumulation_steps: int = 1,
        extra_optimizer_kwargs: dict | None = None,
        weight_decay: float = 0.02,
        warmup_steps: int = 0
):
    """
    The get_optimizer_and_scheduler function is a helper function that returns an optimizer and scheduler
        based on the parameters passed to it.

    :param optimizer: AVAILABLE_OPTIMIZERS: Choose the optimizer
    :param scheduler: AVAILABLE_SCHEDULERS: Determine the learning rate scheduler
    :param steps: int: Specify the number of steps in the training process
    :param learning_rate: float: Set the learning rate for the optimizer
    :param learning_rate_end: float: Set the final learning rate
    :param gradient_accumulation_steps: int: Accumulate the gradients before updating the weights
    :param extra_optimizer_kwargs: dict | None: Pass extra arguments to the optimizer
    :param weight_decay: float: Set the weight decay for adamw optimizer
    :param warmup_steps: int: Specify the number of steps to warm up the learning rate
    :return: A tuple of two objects: (Optimizer and scheduler)
    """
    if extra_optimizer_kwargs is None:
        extra_optimizer_kwargs = {}
    if optimizer == EasyDelOptimizers.ADAFACTOR:
        if scheduler == EasyDelSchedulers.LINEAR:
            tx, sc = fjformer.optimizers.get_adafactor_with_linear_scheduler(
                learning_rate_start=learning_rate,
                learning_rate_end=learning_rate_end,
                gradient_accumulation_steps=gradient_accumulation_steps,
                steps=steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.COSINE:
            tx, sc = fjformer.optimizers.get_adafactor_with_cosine_scheduler(
                learning_rate=learning_rate,
                steps=steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.NONE:
            tx, sc = fjformer.optimizers.get_adafactor_with_linear_scheduler(
                learning_rate_start=learning_rate,
                learning_rate_end=learning_rate,
                steps=steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.WARM_UP_COSINE:
            tx, sc = fjformer.optimizers.get_adafactor_with_warm_up_cosine_scheduler(
                learning_rate=learning_rate,
                steps=steps,
                weight_decay=weight_decay,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.WARM_UP_LINEAR:
            tx, sc = fjformer.optimizers.get_adafactor_with_warmup_linear_scheduler(
                learning_rate_start=learning_rate,
                steps=steps,
                learning_rate_end=learning_rate_end,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                **extra_optimizer_kwargs

            )

        else:
            raise ValueError(
                "seems like you have choose wrong type or unavailable scheduler"
            )
    elif optimizer == EasyDelOptimizers.LION:
        if scheduler == EasyDelSchedulers.LINEAR:
            tx, sc = fjformer.optimizers.get_lion_with_linear_scheduler(
                learning_rate_start=learning_rate,
                learning_rate_end=learning_rate_end,
                steps=steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.COSINE:
            tx, sc = fjformer.optimizers.get_lion_with_cosine_scheduler(
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                steps=steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.NONE:
            tx, sc = fjformer.optimizers.get_lion_with_linear_scheduler(
                learning_rate_start=learning_rate,
                learning_rate_end=learning_rate,
                steps=steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.WARM_UP_COSINE:
            tx, sc = fjformer.optimizers.get_lion_with_warm_up_cosine_scheduler(
                learning_rate=learning_rate,
                steps=steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )

        elif scheduler == EasyDelSchedulers.WARM_UP_LINEAR:
            tx, sc = fjformer.optimizers.get_lion_with_with_warmup_linear_scheduler(
                learning_rate_start=learning_rate,
                steps=steps,
                learning_rate_end=learning_rate_end,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                **extra_optimizer_kwargs
            )
        else:
            raise ValueError(
                "seems like you have choose wrong type or unavailable scheduler")
    elif optimizer == EasyDelOptimizers.ADAMW:
        if scheduler == EasyDelSchedulers.LINEAR:
            tx, sc = fjformer.optimizers.get_adamw_with_linear_scheduler(
                learning_rate_start=learning_rate,
                learning_rate_end=learning_rate_end,
                steps=steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.COSINE:
            tx, sc = fjformer.optimizers.get_adamw_with_cosine_scheduler(
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                steps=steps,
                weight_decay=weight_decay,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.NONE:
            tx, sc = fjformer.optimizers.get_adamw_with_linear_scheduler(
                learning_rate_start=learning_rate,
                learning_rate_end=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                steps=steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.WARM_UP_COSINE:
            tx, sc = fjformer.optimizers.get_adamw_with_warm_up_cosine_scheduler(
                learning_rate=learning_rate,
                steps=steps,
                weight_decay=weight_decay,
                gradient_accumulation_steps=gradient_accumulation_steps,
                **extra_optimizer_kwargs
            )
        elif scheduler == EasyDelSchedulers.WARM_UP_LINEAR:
            tx, sc = fjformer.optimizers.get_adamw_with_warmup_linear_scheduler(
                learning_rate_start=learning_rate,
                steps=steps,
                weight_decay=weight_decay,
                learning_rate_end=learning_rate_end,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                **extra_optimizer_kwargs
            )
        else:
            raise ValueError(
                "seems like you have choose wrong type or unavailable scheduler"
            )
    else:
        raise ValueError(
            f"seems like you have choose wrong type or unavailable optimizer {optimizer} and scheduler {scheduler}"
        )
    return tx, sc
