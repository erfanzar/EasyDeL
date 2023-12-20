from curses import version
import inspect
from jax import grad, jit, numpy as jnp, lax
from ..models import FlaxAutoModelForCausalLMWithValueHead, FlaxPreTrainedModelWrapper
from chex import Array
from typing import List, Optional, Union, Callable
from transformers import PreTrainedTokenizerBase
import datasets
import torch
from .ppo_config import PPOConfig

Dataset = datasets.Dataset


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig = None,
        model: FlaxPreTrainedModelWrapper = None,
        ref_model: Optional[FlaxPreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer=None,
        data_collator: Optional[Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler=None,
    ):
        ...

    def step(
        self,
        queries: List[Array],
        responses: List[Array],
        scores: List[Array],
        response_masks: Optional[List[Array]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`Array`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`Array`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`Array`]):
                List of tensors containing the scores.
            response_masks (List[`Array`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """

        ...

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            
            self._signature_columns += ["label", "query", "response"]

    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(
            set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
