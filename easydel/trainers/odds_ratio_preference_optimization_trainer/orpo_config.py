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

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "orpo")
@dataclass
class ORPOConfig(TrainingArguments):
    """Configuration class for Odds Ratio Preference Optimization training.

    ORPO is a reference-free preference optimization method that uses odds ratios
    to model preferences between chosen and rejected responses. Unlike DPO, ORPO
    doesn't require a reference model, making it more memory-efficient and simpler
    to implement while achieving comparable or better performance.

    The key innovation of ORPO is formulating preference learning through log-odds
    differences: log(p/(1-p)), which provides better gradient properties than raw
    probabilities and eliminates the need for KL regularization with a reference model.

    Attributes:
        trainer_prefix (str | None): Prefix for trainer logs and checkpoints.
            Default: "orpotrainer"
        learning_rate (float): Learning rate for the optimizer.
            Default: 1e-6
        max_length (int | None): Maximum total sequence length (prompt + completion).
            Default: 1024
        max_prompt_length (int | None): Maximum length for prompt sequences.
            Default: 512
        max_completion_length (int | None): Maximum length for completion sequences.
            Automatically calculated as max_length - max_prompt_length if None.
        beta (float): Temperature parameter controlling the strength of preference
            optimization. Higher values make the model more selective between
            chosen and rejected responses. Default: 0.1
        disable_dropout (bool): Whether to disable dropout during training for
            deterministic behavior. Default: True
        label_pad_token_id (int): Token ID used for padding labels in loss computation.
            Default: -100 (ignored by PyTorch/JAX loss functions)
        padding_value (int | None): Value used for padding input sequences.
            If None, uses the tokenizer's pad_token_id.
        generate_during_eval (bool): Whether to generate sample outputs during
            evaluation for qualitative assessment. Default: False
        is_encoder_decoder (bool | None): Whether the model is encoder-decoder
            architecture. Auto-detected if None.
        model_init_kwargs (dict | None): Additional keyword arguments for model
            initialization.
        dataset_num_proc (int | None): Number of processes for parallel dataset
            preprocessing. None uses sequential processing.

    Example:
        >>> config = ORPOConfig(
        ...     beta=0.2,
        ...     max_length=2048,
        ...     learning_rate=2e-6,
        ...     num_train_epochs=3
        ... )

    Note:
        ORPO loss = -log_sigmoid(beta * (log_odds_chosen - log_odds_rejected))
        where log_odds = log(p/(1-p)) for each response.
    """

    trainer_prefix: str | None = field(
        default="orpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The learning rate used during training."},
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "The maximum allowed sequence length for the input."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "The maximum allowed length of the prompt portion of the input."},
    )
    max_completion_length: int | None = field(
        default=None,
        metadata={
            "help": "The maximum allowed length of the completion. If not provided, it is set to "
            "max_length - max_prompt_length."
        },
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "A hyperparameter beta."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Flag to disable dropout during training."},
    )
    label_pad_token_id: int = field(
        default=-100,
        metadata={"help": "The token id used for padding labels."},
    )
    padding_value: int | None = field(
        default=None,
        metadata={"help": "The value used for padding sequences."},
    )
    generate_during_eval: bool = field(
        default=False,
        metadata={"help": "Flag indicating whether to generate sequences during evaluation."},
    )
    is_encoder_decoder: bool | None = field(
        default=None,
        metadata={"help": "Flag to indicate if the model is encoder-decoder."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for dataset processing."},
    )

    def __post_init__(self, max_sequence_length: int | None):
        """
        Post-initialization processing.

        This method is automatically called after the dataclass __init__ method.
        It sets the 'max_completion_length' if it is not provided by subtracting the
        'max_prompt_length' from 'max_length'.

        Returns:
            The result of the superclass __post_init__ method.
        """
        # If max_completion_length is not provided, derive it from max_length and max_prompt_length.
        self._handle_deprecated_max_sequence_length(max_sequence_length)

        if self.max_completion_length is None and self.max_length is not None and self.max_prompt_length is not None:
            self.max_completion_length = self.max_length - self.max_prompt_length

        # Call the post_init of the parent class if it exists.
        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
