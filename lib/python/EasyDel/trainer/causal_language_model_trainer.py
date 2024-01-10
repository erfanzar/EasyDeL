import copy
import dataclasses
import os
import time

import IPython.display
import fjformer.func.loss_func
from fjformer.func.loss_func import cross_entropy_loss_and_accuracy
import wandb
from datasets import Dataset
from wandb.apis.public import Run
from wandb.sdk.lib import RunDisabled

from .training_configurations import TrainArguments

import jax
import flax
from transformers import FlaxAutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from ..utils.utils import Timers, prefix_print
from ..smi import initialise_tracking, get_mem, get_capacity_matrix
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import PartitionSpec
from jax import numpy as jnp
from torch.utils.data import DataLoader
from fjformer import match_partition_rules, make_shard_and_gather_fns, CheckpointManager
from ..etils.errors import EasyDelTimerError
import chex
from typing import Any, Optional, Union, Tuple, Callable, Mapping
from ..etils.easystate import EasyDelState


def calculate_accuracy(predictions: chex.Array, targets: chex.Array):
    """
    The calculate_accuracy function takes in two arrays, predictions and targets.
    The function then calculates the accuracy of the model by comparing the predicted classes to
    the target classes. The predicted class is determined by taking argmax along axis - 1 of predictions.
    The correct_predictions variable is an array containing True or False values depending on whether or not
    the prediction was correct for each example in a batch. The total number of examples that were correctly
    predicted are summed up and divided by the total number of examples to get an accuracy value between 0 and 1.

    :param predictions: chex.Array: Pass in the predictions from the model
    :param targets: chex.Array: Calculate the accuracy of the model
    :return: A single value, the accuracy

    """
    predicted_classes = jnp.argmax(predictions, axis=-1)
    correct_predictions = (predicted_classes == targets).sum()
    total_predictions = targets.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy


def create_casual_language_model_train_step(partition_spec=PartitionSpec(("dp", "fsdp"), "sp")):
    """
    The create_casual_language_model_train_step function is a training step function that takes in the current state 
    of the model,and a batch of data. It then calculates the loss and accuracy for this batch, and returns 
    an updated state with new parameters based on these gradients.

    :param partition_spec: Specify which devices the model will be split across
    :return: A casual_language_model_train_step function that takes in the current state of the model,

    """

    def casual_language_model_train_step(state, batch):
        """
        The casual_language_model_train_step function is a training step function that takes in the current state 
        of the model and a batch of data. It then calculates the loss and accuracy for this batch, 
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

    return casual_language_model_train_step


def create_casual_language_model_evaluation_step(partition_spec=PartitionSpec(("dp", "fsdp"), "sp")):
    """
    The create_casual_language_model_evaluation_step function is used to create a function that calculates the loss
     and accuracy of a model. It takes in a set of parameters, which are then passed into the state.apply_fn function
    to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated from these 
    logits.

    :param partition_spec: Specify the partitioning of the model parameters
    :return: A function that can be used to calculate the loss and accuracy of a model

    """

    def casual_language_model_evaluation_step(state, batch_eval):
        """
        The casual_language_model_evaluation_step function is used to calculate the loss and accuracy of a model.
        It takes in a set of parameters, which are then passed into the state.apply_fn function
        to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated from 
        these logits.

        :param state: Store the model parameters and other information about the training process
        :param batch_eval: Pass the batch of data to the function
        :return: The loss and accuracy of the model

        """
        batch_eval = with_sharding_constraint(
            batch_eval, partition_spec
        )

        def calculate_loss(params):
            """
            The calculate_loss function is used to calculate the loss and accuracy of a model.
            It takes in a set of parameters, which are then passed into the state.apply_fn function
            to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated 
            from these logits.

            :param params: Pass the model parameters to the function
            :return: The loss and the accuracy

            """
            labels = batch_eval.pop("labels")
            logits = state.apply_fn(params=params, **batch_eval,
                                    return_dict=True).logits

            loss, accuracy = cross_entropy_loss_and_accuracy(
                logits[:, :-1, :], labels, batch_eval["attention_mask"].astype(jnp.float32)[:, 1:]
            )
            return loss, accuracy

        loss__, accuracy__ = calculate_loss(state.params)
        return loss__, accuracy__

    return casual_language_model_evaluation_step


def predict(state, input_ids):
    """
    The predict function takes in a state and input_ids, and returns the next token.

    :param state: Store the model parameters and the input_ids parameter is used to pass in a batch of token ids
    :param input_ids: Pass the input to the model
    :return: The next input_ids

    """
    input_ids = with_sharding_constraint(input_ids, PartitionSpec(("dp", "fsdp")))
    pred = state.apply_fn(params=state.params, input_ids=input_ids, return_dict=True)
    token = jnp.argmax(jax.nn.softmax(pred.logits)[:, -1, :])
    input_ids = jnp.concatenate([input_ids, token.reshape(1, -1)], axis=-1)
    return input_ids


@dataclasses.dataclass
class TrainerOutput:
    state: EasyDelState
    predict_function: Optional[Callable]
    mesh: Optional[jax.sharding.Mesh]
    checkpoint_manager: Any
    gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]] = None
    shard_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]] = None
    last_save_file_name: Optional[str] = None
    checkpoint_path: Optional[str] = None


class CausalLanguageModelTrainer:
    def __init__(
            self,
            arguments: TrainArguments,
            dataset_train: Dataset,
            dataset_eval: Dataset = None,
            finetune: bool = True,
            checkpoint_path: Union[str, os.PathLike] = None,
            _do_init_fns: bool = True
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up all the variables that are needed for training, including:
        - The timer to keep track of how long each epoch takes.
        - The dataloaders for both training and evaluation (if provided).
        - The model itself, which will be created from a checkpoint if one was provided.  Otherwise,
         it will be created from scratch using the arguments passed in by the user.
         Note that this function also handles creating a mesh if one was not already specified in arguments
         or loaded from a checkpoint file (see below).
          This means that you can pass in either

        :param self: Represent the instance of the class
        :param arguments: TrainArguments: Pass the arguments to the trainer
        :param dataset_train: Dataset: Pass the training dataset to the trainer
        :param dataset_eval: Dataset: Pass the validation dataset
        :param finetune: bool: Load the model from a checkpoint
        :param checkpoint_path: Union[str,os.PathLike] : Load the checkpoint path
        :param _do_init_fns: bool: Initialize the functions
        :return: Nothing, it just initializes the class

        """
        self.timer = None
        self.dataloader_train = None
        self.dataloader_eval = None
        self.model = None
        self.wandb_runtime: Run | RunDisabled | None = None
        self.max_steps_train = None
        self.max_steps_eval = None
        self.config = None
        self.scheduler = None
        self.tx = None
        self.create_sharded_state_from_params_fn = None
        self.sharded_train_step_fn = None
        self.sharded_predict = None
        self.mesh = None
        self.checkpoint_manager: fjformer.CheckpointManager | None = None
        self.init_fn = None
        self.state_shape = None
        self.state_partition_spec = None
        self.sharded_state = None
        self.arguments = arguments
        self.dataset_train = dataset_train
        self.dataset_eval = dataset_eval
        self.finetune = finetune
        self.checkpoint_path = checkpoint_path
        self.dtype = arguments.dtype
        self.param_dtype = arguments.param_dtype
        if finetune:
            if checkpoint_path is None:
                prefix_print(
                    "Warning",
                    "In case of using finetune = True and Passing checkpoint_path = None you should pass parameters"
                    "in train function"
                )
        if _do_init_fns:
            self.init_functions()
        else:
            prefix_print("Warning", "you have set _do_init_fns to False so function will not me initialized you have "
                                    f"to do in manually (simply with  trainer.init_functions() )")

    def __str__(self):
        string = f"{self.__class__.__name__}("
        for key, value in self.__dict__.items():
            string += value.__str__().replace("\n", "\n\t")
        string += ")"
        return string

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def finish():
        """
        The finish function is called when the experiment ends.
        It can be used to save data, upload files, or do any other cleanup tasks.

        :return: A dictionary of the run"s metadata

        """
        wandb.finish()

    def init_functions(self):
        """
        The init_functions function is responsible for initializing the following:
            - wandb_runtime (if you use_wandb is True)
            - timer object (for logging time taken by various functions)
            - dataloader objects for training and evaluation data, along with max steps per epoch.
              The configure_dataloader function accomplishes this task.

        :param self: Represent the instance of the class
        :return: A tuple of functions

        """
        self.wandb_runtime = self.arguments.get_wandb_init() if self.arguments.use_wandb else None
        self.timer = Timers(
            use_wandb=False,
            tensorboard_writer=self.arguments.get_board()
        )
        self.timer(
            "configure dataloaders"
        ).start()
        self.dataloader_train, self.max_steps_train, \
            self.dataloader_eval, self.max_steps_eval = self.configure_dataloader()
        self.timer(
            "configure dataloaders"
        ).stop()

        self.timer.log(["configure dataloaders"])

        self.timer(
            "configure Model ,Optimizer ,Scheduler and Config"
        ).start()
        self.model, self.tx, self.scheduler, self.config = self.configure_model()
        self.timer(
            "configure Model ,Optimizer ,Scheduler and Config"
        ).stop()
        self.timer.log(["configure Model ,Optimizer ,Scheduler and Config"])

        self.timer(
            "configure functions and sharding them"
        ).start()
        funcs = self.configure_functions()
        self.create_sharded_state_from_params_fn = funcs[0]
        self.sharded_train_step_fn = funcs[1]
        self.sharded_predict = funcs[2]
        self.mesh = funcs[3]
        self.checkpoint_manager = funcs[4]
        self.init_fn = funcs[5]
        self.timer(
            "configure functions and sharding them"
        ).stop()
        self.timer.log(["configure functions and sharding them"])

    def configure_dataloader(self):

        """
        The configure_dataloader function is used to configure the dataloader for training and evaluation.

        :param self: Refer to the class instance itself
        :return: A dataloader_train, max_steps_train, dataloader_eval and max steps eval

        """

        def collate_fn(batch):
            rs = {}
            for key in batch[0].keys():
                if self.arguments.is_left_padded:
                    ssp = [jnp.array(f[key])[..., -self.arguments.max_length:] for f in batch]
                else:
                    ssp = [jnp.array(f[key])[..., :self.arguments.max_length] for f in batch]
                rs[key] = jnp.stack(ssp).reshape(-1, ssp[0].shape[-1])
            return rs

        dataloader_train = DataLoader(
            self.dataset_train,
            collate_fn=collate_fn,
            batch_size=self.arguments.total_batch_size,
            drop_last=True,
        )
        max_steps_train = self.arguments.num_train_epochs * len(
            dataloader_train) if self.arguments.max_steps is None else self.arguments.max_steps
        if self.dataset_eval is not None and self.arguments.do_eval:
            dataloader_eval = DataLoader(self.dataset_eval, collate_fn=collate_fn,
                                         batch_size=self.arguments.total_batch_size, drop_last=True)
            max_steps_eval = len(
                dataloader_eval) if self.arguments.max_steps is None else self.arguments.max_steps
        else:
            dataloader_eval, max_steps_eval = None, 0
        return dataloader_train, max_steps_train, dataloader_eval, max_steps_eval

    def configure_model(self):
        """
        The configure_model function is responsible for creating the model, optimizer and scheduler.

        :param self: Represent the instance of the class
        :return: A model, optimizer, scheduler and config

        """
        extra_configs = {} if self.arguments.extra_configs is None else self.arguments.extra_configs
        if self.arguments.model_class is None:
            config = AutoConfig.from_pretrained(
                self.arguments.model_id,
                trust_remote_code=True,
                gradient_checkpointing=self.arguments.gradient_checkpointing,
                use_pjit_attention_force=self.arguments.use_pjit_attention_force,
                **extra_configs
            )

            assert hasattr(config, "get_partition_rules")
            model = FlaxAutoModelForCausalLM.from_config(
                config, trust_remote_code=True,
                dtype=self.arguments.dtype,
                param_dtype=self.arguments.param_dtype,
                _do_init=False
            )

        else:
            if not hasattr(self.arguments.configs_to_init_model_class["config"], "get_partition_rules"):
                assert self.arguments.custom_rule is not None, (
                    "if you are using custom model to init you must"
                    " pass custom_rule for partition rules "
                )

            self.arguments.configs_to_init_model_class[
                "config"
            ].use_pjit_attention_force = self.arguments.use_pjit_attention_force

            self.arguments.configs_to_init_model_class["config"].axis_dims = self.arguments.sharding_array

            model = self.arguments.model_class(
                **self.arguments.configs_to_init_model_class,
                _do_init=False
            )

            config = self.arguments.configs_to_init_model_class["config"]

        tx, scheduler = self.arguments.get_optimizer_and_scheduler(self.max_steps_train)
        return model, tx, scheduler, config

    def configure_functions(self):
        """
        The configure_functions function is responsible for configuring the functions that will be used in training.
        It does this by first defining a function called init_fn, which initializes the model parameters and returns
        them as a EasyDelState object. The EasyDelState object contains all the information needed to train or evaluate
        on a batch of data, including:
        :param self: Access the class attributes
        :return: A tuple of functions

        """

        def init_fn():
            params__ = self.model.init_weights(
                jax.random.PRNGKey(0), self.arguments.init_input_shape
            )
            if self.arguments.dtype == jnp.bfloat16:
                params__ = self.model.to_bf16(params__)
            elif self.arguments.dtype == jnp.float16:
                params__ = self.model.to_fp16(params__)
            return EasyDelState.create(
                tx=self.tx,
                params=flax.core.freeze({"params": params__}),
                apply_fn=self.model.__call__,
                module_config=copy.deepcopy(
                    self.model.config
                ),
                tx_init=copy.deepcopy(
                    self.arguments.optimizer_kwargs
                ),
                hyperparameters=EasyDelState.create_hyperparameters(
                    self.model.config.model_type
                ),
                module=self.model,
                module_config_args=None
            )

        def create_state_from_params(params_):
            return EasyDelState.create(
                tx=self.tx,
                params=params_,
                apply_fn=self.model.__call__,
                module_config=copy.deepcopy(
                    self.model.config
                ),
                tx_init=copy.deepcopy(
                    self.arguments.optimizer_kwargs
                ),
                hyperparameters=EasyDelState.create_hyperparameters(
                    self.model.config.model_type
                ),
                module=self.model,
                module_config_args=None
            )

        # OHA is useless
        # if self.arguments.loss_re_mat == "OHA":
        #     loss_fn = fjformer.func.loss_func.cross_entropy_with_logits
        # elif self.arguments.loss_re_mat != "":
        #     loss_fn = fused_cross_entropy_loss_and_accuracy
        # else:
        #     loss_fn = cross_entropy_loss_and_accuracy

        state_shape = jax.eval_shape(init_fn)
        state_partition_spec = match_partition_rules(
            self.config.get_partition_rules(
                fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
            ) if self.arguments.custom_rule is None else self.arguments.custom_rule,
            state_shape
        )
        create_sharded_state_from_params_fn = pjit(
            create_state_from_params,
            in_shardings=(state_partition_spec.params,),
            out_shardings=state_partition_spec,
            donate_argnums=(0,)
        )
        sharded_train_step_fn = pjit(
            create_casual_language_model_train_step(self.arguments.step_partition_spec),
            in_shardings=(state_partition_spec, PartitionSpec()),
            out_shardings=(state_partition_spec, PartitionSpec(), PartitionSpec()),
            donate_argnums=(0, 0),
        )
        sharded_predict = pjit(
            predict,
            out_shardings=PartitionSpec(),
            in_shardings=(state_partition_spec, PartitionSpec())
        )
        mesh = self.arguments.get_mesh()
        self.arguments.ckpt_path_exists()
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        self.state_partition_spec = state_partition_spec
        self.state_shape = state_shape
        return create_sharded_state_from_params_fn, sharded_train_step_fn, sharded_predict, mesh, checkpoint_manager, init_fn

    def eval(
            self,
            state: EasyDelState
    ):
        if self.dataset_eval is not None:
            pbar_eval = tqdm(total=self.max_steps_eval)
            for _, batch_eval in enumerate(self.dataloader_eval):
                _ = batch_eval.pop("token_type_ids", None)
                batch_eval["labels"] = batch_eval["input_ids"][..., 1:]
                for pop_arg in self.arguments.ids_to_pop_from_dataset:
                    _ = batch_eval.pop(pop_arg, None)
                loss_eval, accuracy_eval = create_casual_language_model_evaluation_step(
                    self.arguments.step_partition_spec
                )(
                    state, batch_eval
                )
                pbar_eval.update(1)
                if self.wandb_runtime is not None:
                    self.wandb_runtime.log(
                        {
                            "loss_eval": loss_eval.tolist(),
                            "accuracy_eval": accuracy_eval.tolist()
                        }
                    )
                pbar_eval.set_postfix(loss_eval=loss_eval.tolist(), accuracy_eval=accuracy_eval.tolist())

    def init_state(
            self,
            model_parameters: Optional[flax.core.FrozenDict] = None,
            state: Optional[EasyDelState] = None,
    ) -> Tuple[EasyDelState, Mapping[str, Callable], Mapping[str, Callable]]:
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
                        sharded_state = EasyDelState.load_state(
                            verbose=self.arguments.verbose,
                            state_shard_fns=shard_fns,
                            init_optimizer_state=True,
                            checkpoint_path=self.checkpoint_path,
                        )
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
                    params = model_parameters if not self.arguments.do_shard_fns else jax.tree_util.tree_map(
                        lambda f, x: f(x),
                        shard_fns.params,
                        model_parameters,
                    )
                    sharded_state = self.create_sharded_state_from_params_fn(params)
                elif model_parameters is not None and self.checkpoint_path is not None:
                    raise EasyDelTimerError(
                        "You can't pass `model_parameters` and `checkpoint_path` at same time"
                    )
                else:
                    raise EasyDelTimerError(
                        "You should pass `model_parameters` or `checkpoint_path` to trainer in order to load model"
                    )
            else:
                sharded_state = self.init_fn()
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
            state: EasyDelState,
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
        checkpoint_name = f"{self.arguments.model_name}-S{step}"
        filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
        filename += ".easy"
        print(f"Saving Model \033[1;30m{filename}\033[1;0m")
        state.save_state(
            filename=filename,
            checkpoint_dir=os.path.join(self.arguments.save_dir, self.arguments.model_name),
            gather_fns=gather_fns,
            float_dtype=self.dtype,
            verbose=self.arguments.verbose,
            save_optimizer=self.arguments.save_optimizer_state,
        )
        return filename

    def train(
            self,
            model_parameters: Optional[flax.core.FrozenDict] = None,
            state: Optional[EasyDelState] = None
    ) -> TrainerOutput:
        """
        The train function is the main function of this module.
        It takes a model_parameters argument which can be used to load a pretrained model and finetune it.
        The train function returns an TrainerOutput object that contains the last saved file name, predict func,
        train state, mesh and checkpoint streamer.


        :param self: Make the class methods aware of other methods and attributes within the class
        :param model_parameters: flax.core.FrozenDict: Load a pre-trained model
        :param state: Optional[EasyDelState]: Ready to Use State
        :return: An object of type "TrainerOutput"

        """

        def count_model_parameters(_p):
            print(
                "\033[1;31mModel Contain ",
                sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0]) / 1e9,
                " Billion Parameters"
            )

        dir_prefix: str = "/dev/shm"
        if self.arguments.track_memory:
            initialise_tracking(dir_prefix=dir_prefix)
        start_time = time.time()
        sharded_state, shard_fns, gather_fns = self.init_state(
            model_parameters=model_parameters,
            state=state
        )
        count_model_parameters(sharded_state.params)
        with self.mesh:
            pbar = tqdm(total=self.max_steps_train)
            current_step = sharded_state.step.tolist()
            losses = []
            accuracies = []
            pbar.update(sharded_state.step.tolist())
            learning_rates = []
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
                for epoch in range(self.arguments.num_train_epochs):
                    for batch in self.dataloader_train:
                        current_step += 1
                        if (
                                self.arguments.step_start_point is not None
                                and
                                self.arguments.step_start_point > current_step
                        ):
                            pbar.update(1)
                        elif current_step < self.max_steps_train:

                            batch["labels"] = batch["input_ids"][..., 1:]

                            for ssb in self.arguments.ids_to_pop_from_dataset:
                                _ = batch.pop(ssb, None)
                            time_s = time.time()
                            sharded_state, loss, accuracy = self.sharded_train_step_fn(
                                sharded_state,
                                batch
                            )
                            ttl_time = time.time() - time_s
                            losses.append(loss)
                            learning_rates.append(self.scheduler(current_step).tolist())
                            accuracies.append(accuracy)
                            if self.arguments.track_memory:
                                mem_res = get_mem(dir_prefix=dir_prefix)
                            else:
                                mem_res = "Tracking Option is OFF"
                            pbar.update(1)

                            if self.wandb_runtime is not None:
                                trained_tokens = (
                                        current_step * self.arguments.total_batch_size *
                                        self.arguments.gradient_accumulation_steps * self.arguments.max_length
                                )

                                information_queries = {}
                                if self.arguments.track_memory:
                                    for key in ["Used", "Usage Percent"]:
                                        for device, info in get_capacity_matrix(dir_prefix=dir_prefix).items():
                                            information_queries[f"{device.replace('_', ' ')} ({key})"] = float(
                                                info[key].replace("%", "").replace("GB", ""))
                                with jax.spmd_mode("allow_all"):
                                    self.wandb_runtime.log(
                                        {
                                            "loss": loss.tolist(),
                                            "mean loss": (sum(losses) / len(losses)).tolist(),
                                            "accuracy": accuracy.tolist(),
                                            "mean accuracy": (sum(accuracies) / len(accuracies)).tolist(),
                                            "learning_rate": self.scheduler(
                                                sharded_state.step.tolist()
                                            ).tolist(),
                                            "step": sharded_state.step.tolist(),
                                            "step time": ttl_time,
                                            "perplexity": jnp.exp(loss).tolist(),
                                            "trained_tokens": trained_tokens,
                                            "accelerators": information_queries,
                                            "epoch": epoch
                                        }
                                    ),
                                    wandb.summary["captured_memory_log"] = mem_res

                            if self.arguments.track_memory:
                                IPython.display.clear_output(True)
                                pbar.display(mem_res)
                            pbar.set_postfix(
                                loss=loss,
                                learning_rate=self.scheduler(sharded_state.step.tolist()).tolist(),
                                step=sharded_state.step.tolist(),
                                perplexity=jnp.exp(loss).tolist(),
                                accuracy=accuracy,
                                epoch=epoch
                            )
                            if self.arguments.training_time is not None:
                                if time.time() - start_time > self.arguments.training_time:
                                    raise EasyDelTimerError("Time Out")
                        else:
                            break
                        if self.arguments.save_steps is not None and current_step % self.arguments.save_steps == 0:
                            filename = self._save_state(
                                state=sharded_state,
                                gather_fns=gather_fns,
                                milestone=True
                            )
                            checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

            except KeyboardInterrupt:
                print(
                    "\033[1;30m KeyboardInterrupt At training model Will return current state of the model\033[1;0m"
                )

            except EasyDelTimerError:
                print(
                    "\033[1;30m Training reached out maximum training Time Killing training Process "
                    "and Will return current state of the model * \033[1;0m"
                )
            output = TrainerOutput(
                predict_function=self.sharded_predict,
                state=sharded_state,
                mesh=self.mesh,
                shard_fns=shard_fns,
                gather_fns=gather_fns,
                checkpoint_manager=self.checkpoint_manager,
            )
            if self.arguments.save_steps is None and self.arguments.do_last_save:
                filename = self._save_state(
                    state=sharded_state,
                    gather_fns=gather_fns
                )
                checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

            if self.arguments.do_eval:
                self.eval(
                    sharded_state
                )

            output.checkpoint_path = checkpoint_path
            output.last_save_file_name = filename
            wandb.finish()

            return output
