import copy
from glob import glob
import os
import time
import typing

from flax.core import FrozenDict
import termcolor
import wandb

import jax
import flax
from tqdm import tqdm
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec
from jax import numpy as jnp
from fjformer import (
    match_partition_rules,
    make_shard_and_gather_fns
)
from typing import Any, Optional, Tuple, Callable, Mapping
from ...etils.easystate import EasyDeLState
from ...trainer.base_trainer import BaseTrainer, TrainerConfigureFunctionFuncOutput
from ...etils.errors import EasyDeLTimerError
from ...utils.utils import prefix_print
from .modeling_output import CausalLMTrainerOutput
from .fwd_bwd_functions import (
    create_casual_language_model_train_step,
    create_casual_language_model_evaluation_step
)


class CausalLanguageModelTrainer(BaseTrainer):

    def create_collate_function(
            self,
            max_sequence_length: int,
            truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> Callable:
        def collate_fn(batch):
            results = {}
            for key in batch[0].keys():
                if truncation_mode == "keep_end":
                    corrected_sequence = [
                        jnp.array(f[key])[..., -max_sequence_length:] for f in batch
                    ]
                else:
                    corrected_sequence = [
                        jnp.array(f[key])[..., :max_sequence_length] for f in batch
                    ]
                results[key] = jnp.stack(corrected_sequence).reshape(
                    -1,
                    corrected_sequence[0].shape[-1]
                )
            return results

        return collate_fn

    def configure_functions(self) -> TrainerConfigureFunctionFuncOutput:
        """
        The configure_functions function is responsible for configuring the functions that will be used in training.
        It does this by first defining a function called function_configurations, which initializes the model parameters and returns
        them as a EasyDeLState object. The EasyDeLState object contains all the information needed to train or evaluate
        on a batch of data, including:
        :param self: Access the class attributes
        :return: A TrainerConfigureFunctionFuncOutput object

        """

        def initialize_state_function():
            initialized_parameters = self.model.init_weights(
                jax.random.PRNGKey(0),
                self.arguments.init_input_shape
            )

            if self.arguments.dtype == jnp.bfloat16:
                initialized_parameters = self.model.to_bf16(initialized_parameters)
            elif self.arguments.dtype == jnp.float16:
                initialized_parameters = self.model.to_fp16(initialized_parameters)

            tx = self.tx
            parameters = flax.core.freeze({"params": initialized_parameters})
            tx_init = copy.deepcopy(self.arguments.optimizer_kwargs)

            if self.rapture is not None:
                lora_parameters = self.lora_parameters
                if self.arguments.dtype == jnp.bfloat16:
                    lora_parameters = self.model.to_bf16(lora_parameters)
                elif self.arguments.dtype == jnp.float16:
                    lora_parameters = self.model.to_fp16(lora_parameters)

                return EasyDeLState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=lora_parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDeLState.safe_dict(tx_init),
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.lora_model,
                    module_config=self.model.config,
                    module_config_args=None,
                )
            else:
                return EasyDeLState.create(
                    tx=tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model.config),
                    tx_init=tx_init,
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.model,
                    module_config_args=None
                )

        def create_state_from_params_function(parameters):
            if self.rapture is None:
                return EasyDeLState.create(
                    tx=self.tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model.config),
                    tx_init=copy.deepcopy(self.arguments.optimizer_kwargs),
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.model,
                    module_config_args=None
                )
            else:
                return EasyDeLState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDeLState.safe_dict(copy.deepcopy(self.arguments.optimizer_kwargs)),
                    hyperparameters=EasyDeLState.create_hyperparameters(self.model.config.model_type),
                    module=self.lora_model,
                    module_config=self.model.config,
                    module_config_args=None,
                )

        state_shape = jax.eval_shape(initialize_state_function)
        state_partition_spec = match_partition_rules(
            self.config.get_partition_rules(
                fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
            ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
            state_shape
        )
        create_sharded_state_from_params_function = pjit(
            create_state_from_params_function,
            in_shardings=(state_partition_spec.params,),
            out_shardings=state_partition_spec,
            donate_argnums=(0,)
        )
        sharded_train_step_function = pjit(
            create_casual_language_model_train_step(
                partition_spec=self.arguments.step_partition_spec,
                label_smoothing_factor=self.arguments.label_smoothing_factor,
                z_loss=self.arguments.z_loss,
            ),
            in_shardings=(state_partition_spec, PartitionSpec()),
            out_shardings=(state_partition_spec, PartitionSpec(), PartitionSpec()),
            donate_argnums=(0, 0),
        )

        sharded_eval_step_function = pjit(
            create_casual_language_model_evaluation_step(self.arguments.step_partition_spec),
            in_shardings=(state_partition_spec, PartitionSpec()),
            out_shardings=(PartitionSpec(), PartitionSpec(), PartitionSpec()),
            donate_argnums=(0, 0),
        )

        mesh = self.arguments.get_mesh()
        self.arguments.ckpt_path_exists()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        self.state_partition_spec = state_partition_spec
        self.state_shape = state_shape

        return TrainerConfigureFunctionFuncOutput(
            create_sharded_state_from_params_function=create_sharded_state_from_params_function,
            sharded_train_step_function=sharded_train_step_function,
            sharded_eval_step_function=sharded_eval_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
            initialize_state_function=initialize_state_function
        )

    def initialize_state(
            self,
            model_parameters: Optional[flax.core.FrozenDict] = None,
            state: Optional[EasyDeLState] = None,
    ) -> Tuple[EasyDeLState, Mapping[str, Callable], Mapping[str, Callable]]:
        if model_parameters is None and state is None and self.rapture is None and self.checkpoint_path is None:
            raise RuntimeError(
                "You are passing `model_parameters=None`, `state=None`, and `checkpoint_path=None` and also you are not"
                " using LoRA, if you are "
                "Using LoRA make sure to pass parameters and Rapture Config correctly otherwise pass the "
                "model_parameters or state."
            )
        if model_parameters is None and state is None:
            model_parameters = self.lora_parameters
        with self.mesh:
            shard_fns, gather_fns = make_shard_and_gather_fns(
                self.state_partition_spec,
                dtype_specs=self.dtype
            )
            if state is not None:
                sharded_state = state
                params = sharded_state.params if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                    lambda f, x: f(x),
                    shard_fns.params,
                    sharded_state.params
                )
                sharded_state.params = params
                if sharded_state.opt_state is None:
                    prefix_print(
                        "Action", "Optimizer State is not Found!, initializing one."
                    )
                    with jax.default_device(self.arguments.offload_device):
                        sharded_state = sharded_state.init_opt_state()
                        opt_state = sharded_state.opt_state if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                            lambda f, x: f(x),
                            shard_fns.opt_state,
                            sharded_state.opt_state
                        )
                        sharded_state = sharded_state.replace(
                            opt_state=opt_state
                        )
            elif self.finetune:

                if model_parameters is None and self.checkpoint_path is not None:
                    prefix_print(
                        "Action", f"Loading Model From {self.checkpoint_path}"
                    )
                    with jax.default_device(self.arguments.offload_device):
                        sharded_state = EasyDeLState.load_state(
                            verbose=self.arguments.verbose,
                            state_shard_fns=shard_fns,
                            init_optimizer_state=True,
                            checkpoint_path=self.checkpoint_path,
                            input_shape=self.arguments.init_input_shape,
                            config_kwargs=self.arguments.loaded_model_config_kwargs
                        )
                        state_shape = jax.eval_shape(lambda: sharded_state)
                        state_partition_spec = match_partition_rules(
                            self.config.get_partition_rules(
                                fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
                            ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
                            state_shape
                        )
                        sharded_train_step_function = pjit(
                            create_casual_language_model_train_step(
                                partition_spec=self.arguments.step_partition_spec,
                                label_smoothing_factor=self.arguments.label_smoothing_factor,
                                z_loss=self.arguments.z_loss,
                            ),
                            in_shardings=(state_partition_spec, PartitionSpec()),
                            out_shardings=(state_partition_spec, PartitionSpec(), PartitionSpec()),
                            donate_argnums=(0, 0),
                        )

                        sharded_eval_step_function = pjit(
                            create_casual_language_model_evaluation_step(self.arguments.step_partition_spec),
                            in_shardings=(state_partition_spec, PartitionSpec()),
                            out_shardings=(PartitionSpec(), PartitionSpec(), PartitionSpec()),
                            donate_argnums=(0, 0),
                        )

                        self.state_partition_spec = state_partition_spec
                        self.state_shape = state_shape
                        self.sharded_train_step_function = sharded_train_step_function
                        self.sharded_eval_step_function = sharded_eval_step_function

                    if self.arguments.remove_ckpt_after_load:
                        os.remove(self.checkpoint_path)
                elif model_parameters is not None and self.checkpoint_path is None:
                    prefix_print(
                        "Action", f"Sharding Passed Parameters"
                    )
                    from flax.core import unfreeze
                    if not isinstance(model_parameters, flax.core.FrozenDict):
                        prefix_print(
                            "Warning",
                            "Model Parameters should be like FrozenDict({'params': params}) make sure to "
                            "pass as type FrozenDict in case of not getting UnExcepted Errors "
                        )

                    model_parameters = model_parameters if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                        lambda f, x: f(x),
                        shard_fns.params,
                        model_parameters,
                    )
                    sharded_state = self.create_sharded_state_from_params_function(model_parameters)
                elif model_parameters is not None and self.checkpoint_path is not None:
                    raise EasyDeLTimerError(
                        "You can't pass `model_parameters` and `checkpoint_path` at same time"
                    )
                else:
                    raise EasyDeLTimerError(
                        "You should pass `model_parameters` or `checkpoint_path` to trainer in order to load model"
                    )
            else:
                sharded_state = self.initialize_state_function()
                params = sharded_state.params if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                    lambda f, x: f(x),
                    shard_fns.params,
                    sharded_state.params
                )
                sharded_state.params = params

            self.sharded_state = sharded_state
            return sharded_state, shard_fns, gather_fns

    def _save_state(
            self,
            state: EasyDeLState,
            gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
            milestone: bool = False
    ) -> str:
        step = int(
            jax.device_get(
                state.step
            )
        ) + self.arguments.step_start_point if self.arguments.step_start_point is not None else int(
            jax.device_get(
                state.step
            )
        )

        checkpoint_dir = os.path.join(self.arguments.save_dir, self.arguments.model_name)
        filename_extension = ".easy"
        if self.arguments.save_total_limit:
            checkpoint_files = glob(os.path.join(checkpoint_dir, f"*{filename_extension}"))
            checkpoint_files.sort(key=os.path.getmtime)
            for old_checkpoint in checkpoint_files[:-self.arguments.save_total_limit]:
                os.remove(old_checkpoint)
                termcolor.cprint(f"Removed old checkpoint: {old_checkpoint}", color="red", force_color=True)

        checkpoint_name = f"{self.arguments.model_name}-S{step}"
        filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
        filename += ".easy"
        termcolor.cprint(f"Saving Model {filename}.", color="cyan", force_color=True)
        state.save_state(
            filename=filename,
            checkpoint_dir=checkpoint_dir,
            gather_fns=gather_fns,
            float_dtype=self.dtype,
            verbose=self.arguments.verbose,
            save_optimizer=self.arguments.save_optimizer_state,
        )
        return filename

    def train(
            self,
            model_parameters: Optional[flax.core.FrozenDict] = None,
            state: Optional[EasyDeLState] = None
    ) -> CausalLMTrainerOutput:
        """
        The train function is the main function of this module.
        It takes a model_parameters argument which can be used to load a pretrained model and finetune it.
        The train function returns an CausalLMTrainerOutput object that contains the last saved file name, predict func,
        train state, mesh and checkpoint streamer.


        :param self: Make the class methods aware of other methods and attributes within the class
        :param model_parameters: flax.core.FrozenDict: Load a pre-trained model
        :param state: Optional[EasyDeLState]: Ready to Use State
        :return: An object of type "CausalLMTrainerOutput"

        """

        def get_layer_names(frozen_dict, prefix=""):
            layer_names = {}
            for key, value in frozen_dict.items():
                if isinstance(value, FrozenDict):
                    layer_names.update(get_layer_names(value, prefix=f"{prefix}_{key}"))
                else:
                    layer_name = f"{prefix}_{key}".lstrip("/")
                    layer_names[layer_name] = value
            return layer_names

        def count_model_parameters(_p):
            termcolor.cprint(
                f"Model Contain {sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0]) / 1e9} "
                f"Billion Parameters",
                color="red", force_color=True
            )

        checkpoint_path = "SAVING_SKIPPED"
        if self.arguments.performance_mode:
            termcolor.cprint(
                "Performance Mode is ON, we will ignore the Memory Tracking, WANDB Logging, and extra information "
                "Process.",
                color="red",
                force_color=True
            )
        start_time = time.time()
        sharded_state, shard_fns, gather_fns = self.initialize_state(
            model_parameters=model_parameters,
            state=state
        )

        count_model_parameters(sharded_state.params)
        with self.mesh:
            pbar = tqdm(total=self.max_training_steps)
            current_step = int(jax.device_get(sharded_state.step))
            loss_sum = None
            accuracy_sum = None
            pbar.update(sharded_state.step.tolist())  # type: ignore
            if self.wandb_runtime is not None:
                model_parameters_number = sum(
                    n.size for n in
                    jax.tree_util.tree_flatten(flax.core.unfreeze(sharded_state.params))[0]
                ) / 1e9
                self.wandb_runtime.log(
                    {
                        "Number of Model Parameters (Billion)": model_parameters_number
                    }
                )
                wandb.summary["Number of Model Parameters (Billion)"] = model_parameters_number
            try:
                train_iter = iter(self.dataloader_train)
                for epoch in range(self.arguments.num_train_epochs):
                    time_s = time.time()
                    for _ in range(self.max_training_steps // self.arguments.num_train_epochs):
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(self.dataloader_train)
                            batch = next(train_iter)
                        current_step += 1
                        if (
                                self.arguments.step_start_point is not None
                                and
                                self.arguments.step_start_point > current_step
                        ):
                            pbar.update(1)
                        elif current_step < self.max_training_steps:

                            time_prev = time_s
                            time_s = time.time()
                            step_time = time_s - time_prev

                            for ssb in self.arguments.ids_to_pop_from_dataset:
                                _ = batch.pop(ssb, None)

                            (
                                sharded_state,
                                loss,
                                metrics,
                            ) = self.sharded_train_step_function(sharded_state, batch)

                            trained_tokens = jnp.multiply(
                                self.arguments.max_sequence_length, jnp.multiply(
                                    current_step,
                                    self.arguments.total_batch_size
                                )
                            )  # It's faster

                            with jax.spmd_mode("allow_all"):
                                calculating_metrics_start = time.time()
                                loss_sum = loss if loss_sum is None else loss_sum + loss
                                accuracy = metrics["accuracy"]
                                accuracy_sum = accuracy if accuracy_sum is None else accuracy_sum + accuracy
                                mean_loss = loss_sum / (current_step - self.arguments.step_start_point)
                                mean_accuracy = accuracy_sum / (current_step - self.arguments.step_start_point)
                                perplexity = jnp.exp(loss)
                                calculating_metrics_end = time.time()
                                train_metrics = {
                                    "train/loss": loss.tolist(),
                                    "train/mean_loss": mean_loss.tolist(),
                                    "train/accuracy": accuracy,
                                    "train/mean_accuracy": mean_accuracy.tolist(),
                                    "train/learning_rate": self.scheduler(current_step).tolist(),
                                    "train/step": current_step,
                                    "train/step_time": step_time,
                                    "train/perplexity": perplexity.tolist(),
                                    "train/trained_tokens": trained_tokens,
                                    "train/regularization_z_loss": metrics["regularization_z_loss"].tolist(),
                                    "train/epoch": epoch,
                                }
                            if self.arguments.log_grad_norms:
                                train_metrics.update(
                                    {
                                        "train/max_grad_norm": metrics["max_grad_norm"].tolist(),
                                        "train/mean_grad_norm": metrics["mean_grad_norm"].tolist(),
                                    }
                                )
                            aux_loss = metrics.get("aux_loss", None)
                            if aux_loss is not None:
                                train_metrics.update(
                                    {
                                        "train/aux_loss": aux_loss.tolist()
                                    }
                                )
                            pbar.update(1)
                            pbar.set_postfix(**{k.replace("train/", ""): v for k, v in train_metrics.items()})
                            if not self.arguments.performance_mode:
                                if self.arguments.log_grad_norms:
                                    train_metrics.update({
                                        f"grad_norm/{layer_name}": grad_norm.tolist()
                                        for layer_name, grad_norm in get_layer_names(metrics["grad_norms"]).items()
                                    })
                                train_metrics.update(
                                    {
                                        "time_cal/calculating_metrics_step_time": (
                                                calculating_metrics_end - calculating_metrics_start
                                        )
                                    }
                                )
                                train_metrics.update(self.arguments.captured_memory)
                            if self.wandb_runtime is not None and not self.arguments.performance_mode:
                                with jax.spmd_mode("allow_all"):
                                    self.wandb_runtime.log(train_metrics)
                            if self.arguments.training_time is not None:
                                if time.time() - start_time > self.arguments.training_time:
                                    raise EasyDeLTimerError("Time Out")
                        else:
                            break
                        if self.arguments.save_steps is not None and current_step % self.arguments.save_steps == 0:
                            if self.rapture is None:
                                filename = self._save_state(
                                    state=sharded_state,
                                    gather_fns=gather_fns,
                                    milestone=True
                                )
                                checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"
                            else:
                                print(
                                    termcolor.colored(
                                        "Info : ", color="red", force_color=True
                                    ),
                                    termcolor.colored(
                                        "You can not use `save_steps` while using LoRA "
                                        "right now. this action will be skipped", color="white", force_color=True
                                    )
                                )
            except KeyboardInterrupt:
                termcolor.cprint(
                    "KeyboardInterrupt At training model Will return Current State of the Model with Parameters.",
                    color="cyan",
                    force_color=True
                )

            except EasyDeLTimerError:
                termcolor.cprint(
                    "Training reached out maximum training Time Killing training Process "
                    "and Will return Current State of the Model with Parameters.",
                    color="cyan",
                    force_color=True
                )
            if self.arguments.merge_lora_rapture_parameters and self.rapture is not None:
                print(
                    termcolor.colored(
                        "Info : ", color="red", force_color=True
                    ),
                    termcolor.colored(
                        "Merging LoRA Parameters.", color="white", force_color=True
                    )
                )
                sharded_state = sharded_state.replace(
                    params=self.rapture.merge_parameters(sharded_state.params)
                )
            output = CausalLMTrainerOutput(
                state=sharded_state,
                mesh=self.mesh,
                shard_fns=shard_fns,
                gather_fns=gather_fns,
                checkpoint_manager=self.checkpoint_manager,
            )
            if self.arguments.save_steps is None or self.arguments.do_last_save:
                shard_fns, gather_fns = make_shard_and_gather_fns(
                    match_partition_rules(
                        self.config.get_partition_rules(
                            fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
                        ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
                        jax.eval_shape(lambda: sharded_state)
                    ),
                    dtype_specs=self.dtype
                )  # You have to re-init the new shard and gather functions in order to be able to skip LoRA weight
                # crashing errors and saving errors
                filename = self._save_state(
                    state=sharded_state,
                    gather_fns=gather_fns
                )
                checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

            if self.arguments.do_eval:
                for _ in self.eval(
                        sharded_state
                ):
                    ...

            output.checkpoint_path = checkpoint_path
            output.last_save_file_name = filename
            self.arguments._stop_capturing_memory = True
            wandb.finish()

            return output

    def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
        """Evaluate the Given Model State and yield the eval metrics"""
        assert self.dataloader_eval is not None, "`dataloader_eval` is required by evaluator function."
        with self.mesh:
            pbar = tqdm(total=self.max_evaluation_steps)
            pbar.set_description("Evaluating")
            current_step = 0
            loss_sum = None
            accuracy_sum = None

            try:
                eval_iter = iter(self.dataloader_eval)
                for _ in range(self.max_evaluation_steps):
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(self.dataloader_eval)
                        batch = next(eval_iter)
                    current_step += 1
                    time_start = time.time()
                    for key in self.arguments.ids_to_pop_from_dataset:
                        _ = batch.pop(key, None)
                    metrics = self.sharded_eval_step_function(
                        model_state,
                        batch
                    )
                    total_time = time.time() - time_start
                    (
                        loss, accuracy, aux_loss
                    ) = metrics

                    loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
                    accuracy_sum = (
                        accuracy.tolist() if (
                                accuracy_sum is None
                        ) else accuracy_sum + accuracy
                    )

                    eval_metrics = {
                        "eval/loss": loss.tolist(),
                        "eval/mean_loss": loss_sum / (current_step - self.arguments.step_start_point),
                        "eval/mean_accuracy_sum": accuracy_sum / (
                                current_step - self.arguments.step_start_point
                        ),
                        "eval/step": current_step,
                        "eval/step_time": total_time,
                        "eval/perplexity": jnp.exp(loss).tolist(),
                    }
                    if aux_loss is not None:
                        eval_metrics.update(
                            {"eval/aux_loss": aux_loss}
                        )
                    log_metrics = copy.deepcopy(eval_metrics)
                    eval_metrics.update(self.arguments.captured_memory)
                    if self.arguments.use_wandb:
                        with jax.spmd_mode("allow_all"):
                            self.wandb_runtime.log(
                                eval_metrics
                            )

                    pbar.update(1)
                    pbar.set_postfix(**{k.replace("eval/", ""): v for k, v in log_metrics.items()})
                    yield log_metrics
            except KeyboardInterrupt:
                termcolor.cprint(
                    "KeyboardInterrupt At Evaluation model Will return Nothing and just pass.",
                    color="cyan",
                    force_color=True
                )
