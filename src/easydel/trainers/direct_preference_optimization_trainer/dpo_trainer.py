import copy
import os
import time
import typing
import warnings
from abc import ABC
from collections import defaultdict
from functools import partial  # noqa
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Union

import flax.core
import jax
import numpy as np
import termcolor
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from tqdm.autonotebook import tqdm
from transformers import PreTrainedTokenizerBase

from easydel.etils.easystate import EasyDeLState
from easydel.etils.errors import EasyDeLTimerError
from easydel.etils.etils import get_logger
from easydel.trainers.base_trainer import (
    BaseTrainer,
    TrainerConfigureDataloaderOutput,
    TrainerConfigureFunctionOutput,
    TrainerConfigureModelOutput,
)
from easydel.trainers.direct_preference_optimization_trainer.fwd_bwd_functions import (
    create_dpo_concatenated_forward,
    create_dpo_eval_function,
    create_dpo_train_function,
)
from easydel.trainers.direct_preference_optimization_trainer.modelling_output import (
    DPOTrainerOutput,
)
from easydel.trainers.direct_preference_optimization_trainer.utils import (
    DPODataCollatorWithPadding,
    leave_alone_context_manager,
    pad_to_length,
)
from easydel.trainers.training_configurations import TrainArguments

logger = get_logger(__name__)


class DPOTrainer(BaseTrainer, ABC):
    """
        Trainer for Direct Preference Optimization (DPO).

        This trainer handles the training, evaluation, and checkpointing of language models
        using the DPO algorithm. It supports sharding, gradient accumulation, mixed precision
        training, LoRA, and precomputed reference model log probabilities.

        Attributes:
            arguments (TrainArguments): The training arguments.
            model_state (EasyDeLState): The EasyDeLState object for the model being trained.
            ref_model_state (Optional[EasyDeLState]): The EasyDeLState object for the reference model (if used).
            beta (float): The strength of the regularization term in the DPO loss.
            label_smoothing (float): The amount of label smoothing to apply.
            loss_type (Literal["sigmoid", "hinge", "ipo", "kto"]): The type of loss function to use.
            label_pad_token_id (int): The ID of the padding token for labels.
            padding_value (int): The padding value for input sequences.
            train_dataset (Optional[Dataset]): The training dataset.
            eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]]): The evaluation dataset.
            tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer used for preprocessing.
            data_collator (Optional[Callable]): The data collator used for batching.
            max_length (Optional[int]): The maximum sequence length.
            max_prompt_length (Optional[int]): The maximum prompt length.
            max_target_length (Optional[int]): The maximum target length.
            precompute_ref_log_probs (bool): Whether to precompute reference model log probabilities.
            reference_free (bool): Whether to use a reference-free DPO variant.
            is_encoder_decoder (bool): Whether the model is an encoder-decoder architecture.
            dataset_map_arguments (Optional[dict]): Arguments to pass to the dataset `map` function for tokenization.
            low_mem_usage (bool): Whether to prioritize low memory usage during training.
            auto_fix_data (bool): Whether to automatically fix data issues.
            _do_init_fns (bool): Whether to automatically initialize trainer functions.

        Methods:
            initialize_trainer_utils(self): Initializes trainer utilities (logging, timer, dataloaders, model, etc.).
            configure_dataloaders(self) -> TrainerConfigureDataloaderOutput: Configures the dataloaders for training and evaluation.
            configure_model(self) -> TrainerConfigureModelOutput: Configures the model, optimizer, scheduler, and configuration.
            configure_functions(self) -> TrainerConfigureFunctionOutput: Configures and JIT-compiles the training and evaluation step functions.
            _configure_lora(self): Configures LoRA if enabled.
            shard_states(self, state: EasyDeLState, rules: Any) -> EasyDeLState: Shards the provided state according to the given rules.
            create_collect_function(self, max_sequence_length: int, truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end") -> Callable:
                Creates a data collection function for batching.
            _get_train_dataloader(self) -> tensorflow.data.Dataset: Creates the training dataloader.
            _get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset: Creates the evaluation dataloader.
            get_train_dataloader(self) -> tensorflow.data.Dataset: Returns the training dataloader, potentially with precomputed reference log probabilities.
            get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> tensorflow.data.Dataset: Returns the evaluation dataloader, potentially with precomputed reference log probabilities.
            build_tokenized_answer(self, prompt: str, answer: str) -> Dict: Tokenizes a prompt and answer pair, handling special tokens and padding/truncation.
            tokenize_row(self, feature: Dict, state: EasyDeLState = None) -> Dict: Tokenizes a single row of data from the DPO dataset.
            compute_reference_log_probs(self, state: EasyDeLState, padded_batch: Dict) -> tuple[Any, Any]: Computes log probabilities for the chosen and rejected responses using the reference model.
            _save_state(self, state: EasyDeLState, gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]], milestone: bool = False) -> str:
                Saves the model state to a checkpoint file.
            train(self) -> DPOTrainerOutput: Trains the DPO model and returns the training output.
            eval(self, model_state: EasyDeLState) -> Iterator[dict]: Evaluates the DPO model and yields evaluation metrics.

    **Examples:**




        >>> import easydel
        >>> from easydel import (
        ...     TrainArguments,
        ...     EasyDeLOptimizers,
        ...     EasyDeLSchedulers,
        ...     EasyDeLGradientCheckPointers,
        ...     DPOTrainer,
        ...     EasyDeLState,
        ...     easystate_to_huggingface_model
        ... )
        >>> from datasets import load_dataset
        >>> from huggingface_hub import HfApi
        >>> from transformers import AutoTokenizer, LlamaForCausalLM as module_pt
        >>> from jax import numpy as jnp
        >>> import jax
        >>> from jax.sharding import PartitionSpec
        >>> from fjformer import GenerateRNG
        >>> from typing import Optional, Dict
        >>> from datasets import Dataset

        >>> rng_g = GenerateRNG()
        >>> api = HfApi()

        >>> max_length = 512  # Overall maximum length
        >>> max_target_length = 1024  # Maximum Length for target column in Dataset
        >>> max_prompt_length = 1024  # Maximum Length for prompt column in Dataset

        >>> model_name_or_path = "erfanzar/LinguaMatic-Tiny"
        >>> ref_model_name_or_path = "teknium/OpenHermes-2.5-Mistral-7B"
        >>> dtype = jnp.bfloat16

        >>> sharding_axis_dims = (1, -1, 1, 1)
        >>> sharding_axis_names = ("dp", "fsdp", "tp", "sp")


        >>> def extract_anthropic_prompt(prompt_and_response):
        ...     search_term = "\\n\\nAssistant:"
        ...     search_term_idx = prompt_and_response.rfind(search_term)
        ...     assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        ...     return prompt_and_response[: search_term_idx + len(search_term)]


        >>> def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
        ...     dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
        ...     if sanity_check:
        ...         dataset = dataset.select(range(min(len(dataset), 1000)))
        ...     def split_prompt_and_responses(sample) -> Dict[str, str]:
        ...         prompt = extract_anthropic_prompt(sample["chosen"])
        ...         return {
        ...             "prompt": prompt,
        ...             "chosen": sample["chosen"][len(prompt):],
        ...             "rejected": sample["rejected"][len(prompt):],
        ...         }
        ...
        ...     return dataset.map(split_prompt_and_responses)


        >>> arguments = TrainArguments(
        ...     model_name="EasyDeL-DPO",
        ...     num_train_epochs=5,
        ...     learning_rate=1e-4,
        ...     learning_rate_end=3e-5,
        ...     warmup_steps=200,
        ...     optimizer=EasyDeLOptimizers.ADAMW,
        ...     scheduler=EasyDeLSchedulers.LINEAR,
        ...     weight_decay=0.02,
        ...     total_batch_size=128,
        ...     gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ...     sharding_array=sharding_axis_dims,
        ...     fully_sharded_data_parallel=True,
        ...     gradient_accumulation_steps=2,
        ...     dtype=dtype,
        ...     param_dtype=dtype,
        ...     step_start_point=0,
        ...     training_time="7H",
        ...     do_train=True,
        ...     do_eval=True,
        ...     track_memory=False  # Performance boost.
        ...     # You can set other options too or play with them but for now I just stick with these arguments.
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        >>> if tokenizer.pad_token is None:
        ...     tokenizer.pad_token = tokenizer.eos_token

        >>> if tokenizer.pad_token_id is None:
        ...     tokenizer.pad_token_id = tokenizer.eos_token_id

        >>> train_dataset = get_hh("train", sanity_check=True)
        >>> eval_dataset = get_hh("test", sanity_check=True)

        >>> state = EasyDeLState.from_pretrained(
        ...     pretrained_model_name_or_path=model_name_or_path,
        ...     dtype=dtype,
        ...     param_dtype=dtype,
        ...     auto_shard_params=True, # shard parameters
        ...     init_optimizer_state=False, # let trainer configure optimizer
        ...     free_optimizer_state=True,
        ...     sharding_axis_dims=sharding_axis_dims,
        ...     sharding_axis_names=sharding_axis_names,
        ...     partition_axis=easydel.PartitionAxis(
        ...         batch_axis=("dp", "fsdp"),
        ...         query_sequence_axis="sp",
        ...         key_sequence_axis="sp",
        ...         head_axis="tp",
        ...         attention_dim_axis=None
        ...     )
        ... )

        >>> ref_state = EasyDeLState.from_pretrained(
        ...     pretrained_model_name_or_path=ref_model_name_or_path,
        ...     dtype=dtype,
        ...     param_dtype=dtype,
        ...     init_optimizer_state=False,
        ...     free_optimizer_state=True,
        ...     sharding_axis_dims=sharding_axis_dims,
        ...     sharding_axis_names=sharding_axis_names,
        ...     load_in_8bit=True, # Now you can train DPO with ref_state in 8Bit (4Bit, NF4 is coming soon.)
        ...     auto_shard_params=True, # shard parameters
        ...     partition_axis=easydel.PartitionAxis(
        ...         batch_axis=("dp", "fsdp"),
        ...         query_sequence_axis="sp",
        ...         key_sequence_axis="sp",
        ...         head_axis="tp",
        ...         attention_dim_axis=None
        ...     )
        ... )

        >>> dpo_trainer = DPOTrainer(
        ...     model_state=state,
        ...     ref_model_state=ref_state,
        ...     beta=0.1,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     tokenizer=tokenizer,
        ...     arguments=arguments,
        ...     max_length=max_length,
        ...     max_target_length=max_target_length,
        ...     max_prompt_length=max_prompt_length,
        ...     ref_model_init_kwargs=None,  # In case that you pass the ref_model_state a string you have to pass this one too
        ...     model_init_kwargs=None,  # In case that you pass the model_state a string you have to pass this one too
        ...     dataset_map_arguments={
        ...         "num_proc": 8,
        ...         "batched": True,
        ...         "batch_size": 100,
        ...     },
        ...     loss_type="sigmoid",
        ...     data_collator=None,  # Pass None in order to use default data_collector (you can create your own)
        ... )

        >>> output = dpo_trainer.train()

        >>> easydel_jax_model = output.state  # Here's you EasyDeL Model
    """

    def __init__(
        self,
        arguments: TrainArguments,
        model_state: EasyDeLState | str,
        ref_model_state: Optional[EasyDeLState | str] = None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
        label_pad_token_id: int = -100,
        padding_value: int = None,
        train_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
        eval_dataset: Optional[
            Union["Dataset", Dict[str, "Dataset"]]  # noqa #type:ignore
        ] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        data_collator: Optional[Callable] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        reference_free: bool = False,
        is_encoder_decoder: Optional[bool] = False,
        dataset_map_arguments: Optional[dict] = None,
        low_mem_usage: bool = True,
        auto_fix_data: bool = True,
        _do_init_fns: bool = True,
    ):
        assert arguments is not None, (
            "You Have to pass arguments that will be used for training but you have passed"
            "`arguments=None`"
        )
        assert isinstance(
            arguments, TrainArguments
        ), f"arguments type must be `TrainArguments` but got {type(arguments)}"
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model_state, str):
            raise ValueError(
                "You passed model_kwargs to the DPOTrainer. But your model is already instantiated."
            )

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model_state, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model_state, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDeLModelForCausalLM` for you."
            )
            model_state = EasyDeLState.from_pretrained(model_state, **model_init_kwargs)
        if isinstance(ref_model_state, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDeLModelForCausalLM`"
            )
            ref_model_state = EasyDeLState.from_pretrained(
                ref_model_state, **ref_model_init_kwargs
            )

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        self.auto_fix_data = auto_fix_data

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the "
                "DPOTrainer's init it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        padding_value = (
            padding_value if padding_value is not None else tokenizer.pad_token_id
        )  # type: ignore
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = arguments.truncation_mode

        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self.reference_free = reference_free
        self.is_encoder_decoder = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.low_mem_usage = low_mem_usage
        data_collator = (
            DPODataCollatorWithPadding(
                max_prompt_length=self.max_prompt_length,
                max_target_length=self.max_target_length,  # type: ignore
                pad_token_id=tokenizer.pad_token_id,  # type: ignore
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=False,
            )
            if data_collator is None
            else data_collator
        )
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if dataset_map_arguments is None:
            dataset_map_arguments = {}
        with jax.default_device(jax.devices("cpu")[0]):
            train_dataset = train_dataset.map(
                self.tokenize_row,
                **dataset_map_arguments,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    **dataset_map_arguments,
                )

        self.arguments = arguments
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.ref_model_state = ref_model_state
        self.model_state = model_state
        self._loggers_initialized = False
        self.mesh = self.arguments.get_mesh()
        assert (
            padding_value is not None
        ), "`padding_value` can not be set as `None` it must be an integer."

        self.concatenated_forward = jax.jit(
            create_dpo_concatenated_forward(
                is_encoder_decoder=self.is_encoder_decoder,
                padding_value=padding_value,
                label_pad_token_id=label_pad_token_id,
            ),
            static_argnums=[0],
        )

        self._cached_p_l_s = None
        self._cached_c_l_s = None
        self._cached_r_l_s = None
        super().__init__(
            arguments=arguments,
            dataset_train=train_dataset,
            dataset_eval=eval_dataset,
            finetune=True,
            checkpoint_path=None,
            _do_init_fns=_do_init_fns,
        )

    def initialize_trainer_utils(self):
        """
        Initializes various utilities used by the trainer.

        This includes setting up Weights & Biases, initializing the training timer,
        configuring dataloaders, configuring the model and optimizer, sharding the
        model and reference model states, and configuring the training and evaluation functions.
        """
        self._initialize_wandb()
        self._initialize_timer()
        self._configure_dataloaders()
        self._configure_model()
        self._shard_states()
        self._configure_functions()

    def _configure_dataloaders(self):
        """
        Configures the dataloaders for training and evaluation.

        This method retrieves the dataloaders from the `configure_dataloaders` method,
        sets the maximum training and evaluation steps, and logs the time taken for
        this configuration.
        """
        operation_name = "configure dataloaders"
        with self.timer(operation_name):
            dataset_configurations = self.configure_dataloaders()
            self.dataloader_train = dataset_configurations.dataloader_train
            self.max_training_steps = dataset_configurations.max_training_steps
            self.dataloader_eval = dataset_configurations.dataloader_eval
            self.max_evaluation_steps = dataset_configurations.max_evaluation_steps
        self.timer.log(operation_name)

    def _configure_model(self):
        """
        Configures the model, optimizer, scheduler, and configuration.

        This method retrieves the model, optimizer, scheduler, and configuration from
        the `configure_model` method and configures LoRA (if enabled). It also logs
        the time taken for this configuration.
        """
        with self.timer("configure Model, Optimizer, Scheduler and Config"):
            model_configurations = self.configure_model()
            self.model = model_configurations.model
            self.tx = model_configurations.tx
            self.scheduler = model_configurations.scheduler
            self.config = model_configurations.config
            self._configure_lora()
        self.timer.log("configure Model, Optimizer, Scheduler and Config")

    def _configure_functions(self):
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method retrieves the configured functions from the `configure_functions`
        method, sets up the mesh, checkpoint manager, and state initialization
        function, and logs the time taken for this configuration.
        """
        operation_name = "configure functions and sharding them"
        with self.timer(operation_name):
            functions = self.configure_functions()

            self.create_sharded_state_from_params_function = (
                functions.create_sharded_state_from_params_function
            )
            self.sharded_train_step_function = functions.sharded_train_step_function
            self.sharded_eval_step_function = functions.sharded_eval_step_function
            self.mesh = functions.mesh
            self.checkpoint_manager = functions.checkpoint_manager
            self.initialize_state_function = functions.initialize_state_function
        self.timer.log(operation_name)

    def _configure_lora(self):
        """
        Configures LoRA (Low-Rank Adaptation) if enabled in the training arguments.

        This method applies LoRA to the model, sets up the LoRA parameters, apply function,
        optimizer state, model, and optimizer, and logs the time taken for this configuration.
        """
        if self.rapture is not None:
            lora_modules = self.rapture.apply_lora(
                module=self.model,
                parameters=self.arguments.rapture_config.parameters,
                tx=self.tx,
            )
            self.lora_parameters = lora_modules.lora_parameters
            self.lora_apply_fn = lora_modules.lora_module.__call__
            self.lora_opt_state = lora_modules.lora_opt_state
            self.lora_model = lora_modules.lora_module
            self.lora_tx = lora_modules.lora_tx

    def configure_dataloaders(self) -> TrainerConfigureDataloaderOutput:
        """
        Configures the dataloaders for training and evaluation.

        This method creates the training and evaluation dataloaders using the provided
        datasets and data collator. It also determines the maximum number of training
        and evaluation steps based on the dataset sizes and training arguments.

        Returns:
            TrainerConfigureDataloaderOutput: An object containing the configured dataloaders and the
                                            maximum number of training and evaluation steps.
        """
        dataloader_train = self.get_train_dataloader()
        max_evaluation_steps = None
        dataloader_eval = None

        max_training_steps = (
            self.arguments.num_train_epochs * len(dataloader_train)
            if self.arguments.max_training_steps is None
            else self.arguments.max_training_steps
        )
        if self.eval_dataset is not None:
            dataloader_eval = self.get_eval_dataloader(self.eval_dataset)
            max_evaluation_steps = len(dataloader_eval)
        return TrainerConfigureDataloaderOutput(
            dataloader_eval=dataloader_eval,
            dataloader_train=dataloader_train,
            max_evaluation_steps=max_evaluation_steps,
            max_training_steps=max_training_steps,
        )

    def configure_model(self) -> TrainerConfigureModelOutput:
        """
        Configures the model, optimizer, scheduler, and configuration.

        This method retrieves the model configuration from the model state, creates
        the optimizer and scheduler using the training arguments, and returns an
        object containing the configured model, optimizer, scheduler, and configuration.

        Returns:
            TrainerConfigureModelOutput: An object containing the configured model, optimizer, scheduler, and configuration.
        """
        config = self.model_state.module.config
        tx, scheduler = self.arguments.get_optimizer_and_scheduler(
            self.max_training_steps
        )
        model = (self.model_state.module,)
        return TrainerConfigureModelOutput(
            model=model, tx=tx, scheduler=scheduler, config=config
        )

    def configure_functions(self) -> TrainerConfigureFunctionOutput:
        """
        Configures and JIT-compiles the training and evaluation step functions.

        This method sets up the necessary functions for training and evaluation, including:
            - Initialization of the model state.
            - Sharding of the model parameters and optimizer state.
            - JIT-compilation of the training and evaluation step functions.

        Returns:
            TrainerConfigureFunctionOutput: An object containing the configured functions and other relevant information.
        """

        def initialize_state_function():
            """
            Initializes the EasyDeLState object, which holds model parameters, optimizer state, and other training information.

            Returns:
                EasyDeLState: The initialized EasyDeLState object.
            """
            initialized_parameters = self.model.init_weights(
                jax.random.PRNGKey(0), self.arguments.init_input_shape
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
                    hyperparameters=EasyDeLState.create_hyperparameters(
                        self.model.config.model_type
                    ),
                    module=self.lora_model,
                    module_config=self.model_state.module.config,
                    module_config_args=None,
                )
            else:
                return EasyDeLState.create(
                    tx=tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model_state.module.config),
                    tx_init=tx_init,
                    hyperparameters=EasyDeLState.create_hyperparameters(
                        self.model.config.model_type
                    ),
                    module=self.model,
                    module_config_args=None,
                )

        def create_state_from_params_function(parameters):
            """
            Creates an EasyDeLState object from given parameters.

            This function is used when loading a model from pretrained parameters
            or a checkpoint.

            Args:
                parameters (FrozenDict): The model parameters.

            Returns:
                EasyDeLState: The EasyDeLState object initialized with the provided parameters.
            """
            if self.rapture is None:
                return EasyDeLState.create(
                    tx=self.tx,
                    params=parameters,
                    apply_fn=self.model.__call__,
                    module_config=copy.deepcopy(self.model_state.module.config),
                    tx_init=copy.deepcopy(self.arguments.optimizer_kwargs),
                    hyperparameters=EasyDeLState.create_hyperparameters(
                        self.model.config.model_type
                    ),
                    module=self.model,
                    module_config_args=None,
                )
            else:
                return EasyDeLState(
                    step=0,
                    apply_fn=self.lora_apply_fn,
                    params=parameters,
                    tx=self.lora_tx,
                    opt_state=self.lora_opt_state,
                    tx_init=EasyDeLState.safe_dict(
                        copy.deepcopy(self.arguments.optimizer_kwargs)
                    ),
                    hyperparameters=EasyDeLState.create_hyperparameters(
                        self.model.config.model_type
                    ),
                    module=self.lora_model,
                    module_config=self.model_state.module.config,
                    module_config_args=None,
                )

        state_shape = jax.eval_shape(lambda: self.model_state)

        state_partition_spec = match_partition_rules(
            (
                self.config.get_partition_rules(
                    fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
                )
                if self.arguments.custom_rule is None
                else self.arguments.custom_rule
            ),
            state_shape,
        )
        spec_named_sharding = self.specs_to_name_sharding(state_partition_spec)
        empty_sharding = jax.sharding.NamedSharding(
            spec=PartitionSpec(),
            mesh=self.arguments.get_mesh(),
        )
        create_sharded_state_from_params_function = jax.jit(
            create_state_from_params_function,
            in_shardings=(spec_named_sharding.params,),
            out_shardings=spec_named_sharding,
            donate_argnums=(0,),
        )
        train_function = create_dpo_train_function(
            concatenated_forward=self.concatenated_forward,
            ref_state=self.ref_model_state,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
            label_smoothing=self.label_smoothing,
            beta=self.beta,
        )
        sharded_train_step_function = jax.jit(
            train_function,
            in_shardings=(
                spec_named_sharding,
                jax.sharding.NamedSharding(
                    spec=self.arguments.step_partition_spec,
                    mesh=self.mesh,
                ),
            ),
            out_shardings=(spec_named_sharding, empty_sharding),
        )

        eval_function = create_dpo_eval_function(
            concatenated_forward=self.concatenated_forward,
            ref_state=self.ref_model_state,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
            label_smoothing=self.label_smoothing,
            beta=self.beta,
        )

        sharded_eval_step_function = jax.jit(
            eval_function,
            in_shardings=(
                spec_named_sharding,
                jax.sharding.NamedSharding(
                    spec=self.arguments.step_partition_spec,
                    mesh=self.mesh,
                ),
            ),
            out_shardings=(spec_named_sharding, empty_sharding),
        )

        self.arguments.ensure_checkpoint_path()
        self.state_partition_spec = state_partition_spec
        self.state_named_sharding = spec_named_sharding
        self.state_shape = state_shape
        checkpoint_manager = self.arguments.get_streaming_checkpointer()
        mesh = self.arguments.get_mesh()
        return TrainerConfigureFunctionOutput(
            create_sharded_state_from_params_function=create_sharded_state_from_params_function,
            sharded_train_step_function=sharded_train_step_function,
            sharded_eval_step_function=sharded_eval_step_function,
            mesh=mesh,
            checkpoint_manager=checkpoint_manager,
            initialize_state_function=initialize_state_function,
        )

    def _shard_states(self):
        """
        Shards the model and reference model states if automatic sharding is enabled.

        This method shards the `model_state` and `ref_model_state` using the sharding rules
        defined in the model configuration. It also initializes the optimizer and scheduler
        for the sharded model state.
        """
        if self.model_state.tx is None or self.model_state.opt_state is None:
            inner_module_operation_name = (
                "initializing TX and Schedulers for `model_state`"
            )
            with self.timer(inner_module_operation_name):
                params_with_opt = (
                    self.model_state.params["params"]
                    if "_overwrite_with_gradient" in self.model_state.params
                    else self.model_state.params
                )
                opt_state = self.tx.init(params_with_opt)

                self.model_state = self.model_state.replace(
                    opt_state=opt_state,
                    tx=self.tx,
                )
            self.timer.log(inner_module_operation_name)
        else:
            logger.info(
                "Found an existing TX and OptimizerState for "
                "model_state (ignore sharding and tx_init)."
            )

    def create_collect_function(
        self,
        max_sequence_length: int,
        truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end",
    ) -> Callable:
        """
        Creates a data collection function for batching.

        For DPO training, this method simply returns the pre-configured `data_collator`.

        Args:
            max_sequence_length (int): The maximum sequence length (not used in this implementation).
            truncation_mode (typing.Literal["keep_end", "keep_start"], optional):
                The truncation mode (not used in this implementation). Defaults to "keep_end".

        Returns:
            Callable: The data collator function.
        """
        return self.data_collator

    def _get_train_dataloader(self) -> "tensorflow.data.Dataset":  # noqa #type:ignore
        """
        Creates the training dataloader as a TensorFlow Dataset.

        This method retrieves the training dataset, applies the data collator, and converts
        it into a TensorFlow Dataset for efficient batching and data loading during training.

        Returns:
            tensorflow.data.Dataset: The training dataloader.

        Raises:
            ValueError: If the training dataset is not set.
        """
        import tensorflow_datasets

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        return tensorflow_datasets.as_numpy(
            train_dataset.to_tf_dataset(
                batch_size=self.arguments.total_batch_size,
                collate_fn=data_collator,
                num_workers=self.arguments.dataloader_num_workers,
                shuffle=True,
                drop_remainder=True,
            )
        )

    def _get_eval_dataloader(
        self,
        eval_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
    ) -> "tensorflow.data.Dataset":  # noqa #type:ignore
        """
        Creates the evaluation dataloader as a TensorFlow Dataset.

        This method retrieves the evaluation dataset (either provided as an argument or
        from the `self.eval_dataset` attribute), applies the data collator, and converts
        it into a TensorFlow Dataset for efficient batching and data loading during evaluation.

        Args:
            eval_dataset (Optional[Dataset], optional):
                An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

        Returns:
            tensorflow.data.Dataset: The evaluation dataloader.

        Raises:
            ValueError: If no evaluation dataset is provided or set.
        """
        import tensorflow_datasets

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return tensorflow_datasets.as_numpy(
            eval_dataset.to_tf_dataset(
                batch_size=self.arguments.total_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.arguments.dataloader_num_workers,
                shuffle=False,
                drop_remainder=True,
            )
        )

    def get_train_dataloader(self) -> "tensorflow.data.Dataset":  # noqa #type:ignore
        """
        Returns the training dataloader, potentially with precomputed reference log probabilities.

        If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
        probabilities for the chosen and rejected responses in the training dataset and adds
        them as columns to the dataset.

        Returns:
            tensorflow.data.Dataset: The training dataloader.
        """

        import tensorflow_datasets

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            data_loader = tensorflow_datasets.as_numpy(
                self.train_dataset.to_tf_dataset(
                    batch_size=self.arguments.total_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.arguments.dataloader_num_workers,
                    shuffle=False,
                    drop_remainder=True,
                )
            )
            reference_chosen_log_probs = []
            reference_rejected_log_probs = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Train dataset reference log probs"
            ):
                reference_chosen_logp, reference_rejected_logp = (
                    self.compute_reference_log_probs(
                        self.model_state,
                        padded_batch,
                    )
                )
                reference_chosen_log_probs.append(reference_chosen_logp)
                reference_rejected_log_probs.append(reference_rejected_logp)

            all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
            all_reference_rejected_log_probs = jnp.concatenate(
                reference_rejected_log_probs
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_log_probs",
                column=all_reference_rejected_log_probs,
            )

            self._precomputed_train_ref_log_probs = True
        return self._get_train_dataloader()

    def get_eval_dataloader(
        self,
        eval_dataset: Optional["Dataset"] = None,  # noqa #type:ignore
    ) -> "tensorflow.data.Dataset":  # noqa #type:ignore
        """
        Returns the evaluation dataloader, potentially with precomputed reference log probabilities.

        If `precompute_ref_log_probs` is enabled, this method computes the reference model's log
        probabilities for the chosen and rejected responses in the evaluation dataset and adds
        them as columns to the dataset.

        Args:
            eval_dataset (Optional[Dataset], optional):
                An optional evaluation dataset to use. If None, `self.eval_dataset` is used. Defaults to None.

        Returns:
            tensorflow.data.Dataset: The evaluation dataloader.
        """

        import tensorflow_datasets

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            # prepare dataloader
            data_loader = tensorflow_datasets.as_numpy(
                eval_dataset.to_tf_dataset(
                    batch_size=self.arguments.total_batch_size,
                    collate_fn=self.data_collator,
                    num_workers=self.arguments.dataloader_num_workers,
                    shuffle=False,
                    drop_remainder=True,
                )
            )

            reference_chosen_log_probs = []
            reference_rejected_log_probs = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Eval dataset reference log probs"
            ):
                reference_chosen_logp, reference_rejected_logp = (
                    self.compute_reference_log_probs(self.model_state, padded_batch)
                )
                reference_chosen_log_probs.append(reference_chosen_logp.cpu())
                reference_rejected_log_probs.append(reference_rejected_logp.cpu())

            all_reference_chosen_log_probs = jnp.concatenate(reference_chosen_log_probs)
            all_reference_rejected_log_probs = jnp.concatenate(
                reference_rejected_log_probs
            )

            eval_dataset = eval_dataset.add_column(
                name="reference_chosen_log_probs", column=all_reference_chosen_log_probs
            )
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_log_probs",
                column=all_reference_rejected_log_probs,
            )

            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return self._get_eval_dataloader(eval_dataset=eval_dataset)

    def tokenize_row(self, feature: Dict, state: EasyDeLState = None) -> Dict:
        """
        Tokenizes a single row of data from the DPO dataset.

        Args:
            feature (Dict): A dictionary containing the "prompt", "chosen", and "rejected" texts.
            state (EasyDeLState, optional): Not used in this implementation. Defaults to None.

        Returns:
            Dict: A dictionary containing the tokenized prompt, chosen response, and rejected response,
                along with attention masks and labels.

        Raises:
            ValueError: If the input data types are incorrect.
        """

        def validate_input(text: str, name: str) -> None:
            if not isinstance(text, str):
                raise ValueError(
                    f"{name} should be a string but got {type(text)}: {text}"
                )

        validate_input(feature["prompt"], "prompt")
        validate_input(feature["chosen"], "chosen")
        validate_input(feature["rejected"], "rejected")

        prompt_tokens = self._tokenize_prompt(feature["prompt"])
        chosen_tokens = self._tokenize_answer(feature["prompt"], feature["chosen"])
        rejected_tokens = self._tokenize_answer(feature["prompt"], feature["rejected"])

        chosen_sequence_tokens = self._create_sequence_tokens(chosen_tokens)
        rejected_sequence_tokens = self._create_sequence_tokens(rejected_tokens)

        batch = {}
        self._add_to_batch(batch, "", prompt_tokens)
        self._add_to_batch(batch, "chosen_", chosen_sequence_tokens)
        self._add_to_batch(batch, "rejected_", rejected_sequence_tokens)

        return batch

    def _tokenize_prompt(self, prompt: str) -> Dict:
        tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
        tokens = {f"prompt_{k}": v for k, v in tokens.items()}
        tokens["prompt_input_ids"] = self._add_special_tokens(
            tokens["prompt_input_ids"]
        )
        tokens["prompt_attention_mask"] = self._add_special_tokens(
            tokens["prompt_attention_mask"], token=1
        )
        return tokens

    def _tokenize_answer(self, prompt: str, answer: str) -> Dict:
        tokens = self.build_tokenized_answer(prompt, answer)
        tokens["prompt_input_ids"] = self._add_special_tokens(
            tokens["prompt_input_ids"]
        )
        tokens["prompt_attention_mask"] = self._add_special_tokens(
            tokens["prompt_attention_mask"], token=1
        )
        tokens["input_ids"] = self._add_special_tokens(tokens["input_ids"], end=True)
        tokens["attention_mask"] = self._add_special_tokens(
            tokens["attention_mask"], token=1, end=True
        )
        return tokens

    def _create_sequence_tokens(self, tokens: Dict) -> Dict:
        sequence_tokens = {
            k: np.concatenate((tokens[f"prompt_{k}"], tokens[k]), axis=-1)
            for k in ["input_ids", "attention_mask"]
        }
        sequence_tokens["labels"] = sequence_tokens["input_ids"]
        sequence_tokens["labels"][
            : len(tokens["prompt_input_ids"])
        ] = self.label_pad_token_id
        return sequence_tokens

    def _add_special_tokens(
        self, array: np.ndarray, token: int = None, end: bool = False
    ) -> np.ndarray:
        token = (
            token
            if token is not None
            else (self.tokenizer.eos_token_id if end else self.tokenizer.bos_token_id)
        )
        special_token = np.array(token).reshape(1, 1)
        array = np.atleast_2d(array)
        return (
            np.concatenate((special_token, array), axis=-1)
            if not end
            else np.concatenate((array, special_token), axis=-1)
        )

    def _add_to_batch(self, batch: Dict, prefix: str, tokens: Dict) -> None:
        for type_key, token_array in tokens.items():
            if type_key == "token_type_ids":
                continue

            max_length = self.max_target_length if prefix else self.max_prompt_length
            pad_value = 0 if type_key == "attention_mask" else self.padding_value

            token_array = pad_to_length(
                token_array.astype("i4"),
                max_length,
                pad_value=pad_value,
                axis=-1,
            )
            token_array = token_array[..., :max_length]

            if token_array.shape[-1] != max_length:
                raise ValueError(
                    f"Padding error for {prefix}{type_key}. Expected length "
                    f"{max_length}, got {token_array.shape[-1]}"
                )

            batch[f"{prefix}{type_key}"] = token_array

    def build_tokenized_answer(self, prompt, answer):
        """
        Tokenizes a prompt and answer pair, handling special tokens and padding/truncation.
        This method tokenizes the prompt and answer separately, then concatenates them
        while ensuring correct token alignment. It also handles adding special tokens
        (BOS and EOS) and padding/truncating sequences to the appropriate lengths.
        Args:
            prompt (str): The prompt text.
            answer (str): The answer text.
        Returns:
            Dict: A dictionary containing the tokenized prompt and answer, along with attention masks.
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]
        prompt_input_ids = np.asarray(prompt_input_ids, dtype="i4")
        answer_input_ids = np.asarray(answer_input_ids, dtype="i4")
        full_concat_input_ids = np.concatenate((prompt_input_ids, answer_input_ids))

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        response_token_ids_start_idx = len(prompt_input_ids)
        if (
            prompt_input_ids.tolist()
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][
            response_token_ids_start_idx:
        ]

        return dict(
            prompt_input_ids=np.array(prompt_input_ids, dtype="i4"),
            prompt_attention_mask=np.array(prompt_attention_mask, dtype="i4"),
            input_ids=np.array(answer_input_ids, dtype="i4"),
            attention_mask=np.array(answer_attention_mask, dtype="i4"),
        )

    def compute_reference_log_probs(
        self,
        state: EasyDeLState,
        padded_batch: Dict,
    ) -> tuple[Any, Any]:
        """
        Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.

        Args:
            state (EasyDeLState): The EasyDeLState object of the model (used if no reference model is provided).
            padded_batch (Dict): The padded batch of data.

        Returns:
            tuple[Any, Any]: A tuple containing the log probabilities for the chosen and rejected responses.
        """

        if self.ref_model_state is None:
            (
                reference_chosen_log_probs,
                reference_rejected_log_probs,
                _,
                _,
            ) = self.concatenated_forward(
                apply_fn=state.apply_fn,
                params=state.params,
                batch=padded_batch,
            )
        else:
            (
                reference_chosen_log_probs,
                reference_rejected_log_probs,
                _,
                _,
            ) = self.concatenated_forward(
                apply_fn=self.ref_model_state.apply_fn,
                params=self.ref_model_state.params,
                batch=padded_batch,
            )

        return reference_chosen_log_probs, reference_rejected_log_probs

    def _save_state(
        self,
        state: EasyDeLState,
        gather_fns: Optional[Any | Mapping[str, Callable] | dict[Callable]],
        milestone: bool = False,
    ) -> str:
        """
        Saves the model state to a checkpoint file.

        This method constructs the checkpoint file name, prints a message indicating the save operation,
        and uses the `save_state` method of the `EasyDeLState` object to save the state to disk.

        Args:
            state (EasyDeLState): The EasyDeLState object to be saved.
            gather_fns (Optional[Any | Mapping[str, Callable] | dict[Callable]]):
                Gather functions used to collect sharded data before saving.
            milestone (bool, optional): Whether this save is a milestone (e.g., end of epoch). Defaults to False.

        Returns:
            str: The filename of the saved checkpoint.
        """
        step = (
            int(jax.device_get(state.step)) + self.arguments.step_start_point
            if self.arguments.step_start_point is not None
            else int(jax.device_get(state.step))
        )
        checkpoint_name = f"{self.arguments.model_name}-S{step}"
        filename = f"{checkpoint_name}_{step}" if milestone else f"{checkpoint_name}"
        filename += ".easy"
        termcolor.cprint(f"Saving Model {filename}.", color="red", force_color=True)
        state.save_state(
            filename=filename,
            checkpoint_dir=os.path.join(
                self.arguments.save_dir, self.arguments.model_name
            ),
            gather_fns=gather_fns,
            float_dtype=self.dtype,
            verbose=self.arguments.verbose,
            save_optimizer=self.arguments.save_optimizer_state,
        )
        return filename

    def train(self) -> DPOTrainerOutput:
        """
        Trains the DPO model.

        This method orchestrates the training process, iterating over epochs and batches,
        performing training steps, logging metrics, saving checkpoints, handling keyboard
        interrupts and timeouts, and optionally evaluating the model.

        Returns:
            DPOTrainerOutput: An object containing the trained model state and other training information.

        Raises:
            AssertionError: If the model state is None.
        """
        assert (
            self.model_state is not None
        ), "model_state can not be None for training purpose"
        with self.mesh:
            with (
                jax.default_device(jax.devices("cpu")[0])
                if self.low_mem_usage
                else leave_alone_context_manager
            ):
                checkpoint_path = "SAVING_SKIPPED"
                flops_per_device = (
                    self.calculate_number_total_flops_per_device(
                        params=self.model_state.params
                    )
                    / 1e12
                )
                pbar = tqdm(total=self.max_training_steps)
                pbar.set_description("Training")
                current_step = (
                    self.model_state.step.tolist()
                    if isinstance(self.model_state.step, jax.Array)
                    else self.model_state.step
                )

                loss_sum = None
                chosen_rewards_sum = None
                rejected_rewards_sum = None
                filename = None

                try:
                    for epoch_index in range(self.arguments.num_train_epochs):
                        for batch in self.dataloader_train:
                            if self.arguments.step_start_point > current_step:
                                ...
                            elif current_step < self.max_training_steps:
                                time_start = time.time()
                                self.model_state, metrics = (
                                    self.sharded_train_step_function(
                                        self.model_state, batch
                                    )
                                )
                                total_time = time.time() - time_start
                                flops = flops_per_device / total_time
                                (loss, chosen_rewards, rejected_rewards) = (
                                    metrics.loss,
                                    metrics.chosen_rewards[0],
                                    metrics.rejected_rewards[0],
                                )
                                loss.block_until_ready()
                                loss_sum = (
                                    loss.tolist()
                                    if loss_sum is None
                                    else loss_sum + loss
                                )

                                rejected_rewards_sum = (
                                    rejected_rewards.tolist()
                                    if (rejected_rewards_sum is None)
                                    else rejected_rewards_sum + rejected_rewards
                                )
                                chosen_rewards_sum = (
                                    chosen_rewards.tolist()
                                    if (chosen_rewards_sum is None)
                                    else chosen_rewards_sum + chosen_rewards
                                )
                                train_metrics = {
                                    "train/loss": loss.tolist(),
                                    "train/mean_loss": loss_sum
                                    / (
                                        (current_step + 1)
                                        - self.arguments.step_start_point
                                    ),
                                    "train/mean_rejected_rewards": rejected_rewards_sum
                                    / (
                                        (current_step + 1)
                                        - self.arguments.step_start_point
                                    ),
                                    "train/mean_chosen_rewards": chosen_rewards_sum
                                    / (
                                        (current_step + 1)
                                        - self.arguments.step_start_point
                                    ),
                                    "train/learning_rate": self.scheduler(
                                        jax.device_get(self.model_state.step)
                                    ).tolist(),
                                    "train/step": current_step,
                                    "train/step_time": total_time,
                                    "train/perplexity": jnp.exp(loss).tolist(),
                                    "train/epoch": epoch_index,
                                    "train/TFLOPs": flops,
                                }
                                log_metrics = copy.deepcopy(train_metrics)
                                train_metrics.update(self.arguments._captured_memory)
                                self.arguments.log_metrics(
                                    metrics=train_metrics,
                                    step=current_step,
                                )
                                pbar.update(1)
                                pbar.set_postfix(
                                    **{
                                        k.replace("train/", ""): v
                                        for k, v in log_metrics.items()
                                    }
                                )
                            else:
                                break

                            current_step += 1
                except KeyboardInterrupt:
                    termcolor.cprint(
                        "KeyboardInterrupt At training model Will return Current State of the Model with Parameters.",
                        color="cyan",
                        force_color=True,
                    )

                except EasyDeLTimerError:
                    termcolor.cprint(
                        "Training reached out maximum training Time Killing training Process "
                        "and Will return Current State of the Model with Parameters.",
                        color="cyan",
                        force_color=True,
                    )

                if (
                    self.arguments.merge_lora_rapture_parameters
                    and self.rapture is not None
                ):
                    print(
                        termcolor.colored("Info : ", color="red", force_color=True),
                        termcolor.colored(
                            "Merging LoRA Parameters.", color="white", force_color=True
                        ),
                    )
                    self.model_state = self.model_state.replace(
                        params=self.rapture.merge_parameters(self.model_state.params)
                    )

                shard_fns, gather_fns = make_shard_and_gather_fns(
                    partition_specs=match_partition_rules(
                        rules=self.model_state.module.config.get_partition_rules(
                            self.arguments.fully_sharded_data_parallel
                        ),
                        params=jax.eval_shape(lambda: self.model_state),
                    ),
                    mesh=self.mesh,
                )
                output = DPOTrainerOutput(
                    state=self.model_state,
                    mesh=self.mesh,
                    shard_fns=shard_fns,
                    gather_fns=gather_fns,
                    checkpoint_manager=self.checkpoint_manager,
                )
                if self.arguments.save_steps is None and self.arguments.do_last_save:
                    shard_fns, gather_fns = make_shard_and_gather_fns(
                        match_partition_rules(
                            (
                                self.config.get_partition_rules(
                                    fully_sharded_data_parallel=self.arguments.fully_sharded_data_parallel
                                )
                                if self.arguments.custom_rule is None
                                else self.arguments.custom_rule
                            ),
                            jax.eval_shape(lambda: self.model_state),
                        ),
                        mesh=self.mesh,
                    )  # You have to re-init the new shard and gather functions in order to be able to skip LoRA weight
                    # crashing errors and saving errors
                    filename = self._save_state(
                        state=self.model_state, gather_fns=gather_fns
                    )
                    checkpoint_path = f"{str(self.arguments.get_path())}/{filename}"

                if self.arguments.do_eval:
                    for _ in self.eval(self.model_state):
                        ...

                output.checkpoint_path = checkpoint_path
                output.last_save_file_name = filename
                self.finish()

        return output

    def eval(self, model_state: EasyDeLState) -> typing.Iterator[dict]:
        """
        Evaluates the DPO model using the provided model state.

        This method iterates over the evaluation dataset, performs evaluation steps,
        calculates metrics, logs metrics, and yields a dictionary of metrics for each step.

        Args:
            model_state (EasyDeLState): The EasyDeLState object containing the model parameters
                                        and other relevant information.

        Yields:
            Iterator[dict]: An iterator that yields a dictionary of evaluation metrics for each step.

        Raises:
            AssertionError: If the evaluation dataset is not set.
        """
        assert (
            self.eval_dataset is not None
        ), "`dataloader_eval` is required by evaluator function."
        with self.mesh:
            pbar = tqdm(total=self.max_evaluation_steps)
            pbar.set_description("Evaluating")
            current_step = 0
            loss_sum = None
            chosen_rewards_sum = None
            rejected_rewards_sum = None
            flops_per_device = (
                self.calculate_number_total_flops_per_device(
                    params=self.model_state.params
                )
                / 1e12
            )

            try:
                for batch in self.dataloader_eval:
                    time_start = time.time()
                    for key in self.arguments.ids_to_pop_from_dataset:
                        _ = batch.pop(key, None)
                    for key in list(batch.keys()):
                        if not (
                            key.endswith("_input_ids")
                            or key.endswith("_attention_mask")
                            or key.endswith("_labels")
                        ):
                            _ = batch.pop(key, None)

                    metrics = self.sharded_eval_step_function(model_state, batch)
                    total_time = time.time() - time_start
                    flops = flops_per_device / total_time
                    (loss, chosen_rewards, rejected_rewards) = (
                        metrics.loss,
                        metrics.chosen_rewards[0],
                        metrics.rejected_rewards[0],
                    )

                    loss_sum = loss.tolist() if loss_sum is None else loss_sum + loss
                    rejected_rewards_sum = (
                        rejected_rewards.tolist()
                        if (rejected_rewards_sum is None)
                        else rejected_rewards_sum + rejected_rewards
                    )
                    chosen_rewards_sum = (
                        chosen_rewards.tolist()
                        if (chosen_rewards_sum is None)
                        else chosen_rewards_sum + chosen_rewards
                    )

                    eval_metrics = {
                        "eval/loss": loss.tolist(),
                        "eval/mean_loss": loss_sum
                        / ((current_step + 1) - self.arguments.step_start_point),
                        "eval/mean_rejected_rewards": rejected_rewards_sum
                        / ((current_step + 1) - self.arguments.step_start_point),
                        "eval/mean_chosen_rewards": chosen_rewards_sum
                        / ((current_step + 1) - self.arguments.step_start_point),
                        "eval/step": current_step,
                        "eval/step_time": total_time,
                        "eval/perplexity": jnp.exp(loss).tolist(),
                        "eval/TFLOPs": flops,
                    }
                    log_metrics = copy.deepcopy(eval_metrics)
                    eval_metrics.update(self.arguments._captured_memory)
                    self.arguments.log_metrics(metrics=eval_metrics, step=current_step)

                    pbar.update(1)
                    pbar.set_postfix(
                        **{k.replace("eval/", ""): v for k, v in log_metrics.items()}
                    )
                    yield eval_metrics
                    current_step += 1
            except KeyboardInterrupt:
                termcolor.cprint(
                    "KeyboardInterrupt At Evaluation model Will return Nothing and just pass.",
                    color="cyan",
                    force_color=True,
                )

    def __repr__(self):
        """
        The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        :param self: Refer to the instance of the class
        :return: A string representation of the object
        """
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                try:
                    repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                    string += (
                        repr_src
                        if len(repr_src) < 350
                        else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
                    )
                except TypeError:
                    repr_src = f"\t{k} : " + "EasyDeLReadingError" + "\n"
                    string += (
                        repr_src
                        if len(repr_src) < 350
                        else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
                    )

        return string + ")"

    def __str__(self):
        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
