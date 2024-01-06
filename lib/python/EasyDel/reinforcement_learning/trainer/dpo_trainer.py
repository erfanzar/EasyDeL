import warnings
from collections import defaultdict

import chex
import jax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.collectors import DPODataCollatorWithPadding
from typing import Optional, Literal, Dict, Union, Any, Tuple, List

from ...modules.auto_easydel_model import AutoEasyDelModelForCausalLM
from datasets import Dataset
from jax import numpy as jnp
from ...modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel
from ...trainer.training_configurations import TrainArguments
from transformers import PreTrainedTokenizerBase


class DPOTrainer:
    def __init__(
            self,
            model: EasyDelFlaxPretrainedModel | str = None,
            ref_model: Optional[EasyDelFlaxPretrainedModel | str] = None,
            beta: float = 0.1,
            label_smoothing: float = 0,
            loss_type: Literal["sigmoid", "hinge", "ipo", "kto"] = "sigmoid",
            arguments: TrainArguments = None,
            label_pad_token_id: int = -100,
            padding_value: int = None,
            truncation_mode: str = "keep_end",
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            max_length: Optional[int] = None,
            max_prompt_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            precompute_ref_log_probs: bool = False,
            model_init_kwarguments: Optional[Dict] = None,
            ref_model_init_kwarguments: Optional[Dict] = None,
    ):
        if model_init_kwarguments is None:
            model_init_kwarguments = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwarguments to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwarguments is None:
            ref_model_init_kwarguments = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwarguments to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM` for you."
            )
            model = AutoEasyDelModelForCausalLM.from_pretrained(model, **model_init_kwarguments)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM`"
            )
            ref_model = AutoEasyDelModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwarguments)
        data_collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=False,
        )

        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        train_dataset = train_dataset.map(self.tokenize_row)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row)

        self.arguments = arguments
        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.model = model
        self._loggers_initialized = False
        self.mesh = self.arguments.get_mesh()

    def _get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_dataset = self._remove_unused_columns(train_dataset, description="training")

        dataloader_params = {
            "batch_size": self.arguments.total_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.arguments.dataloader_num_workers,
            "pin_memory": self.arguments.dataloader_pin_memory,
        }

        return DataLoader(train_dataset, **dataloader_params)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.arguments.total_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.arguments.dataloader_num_workers,
                "pin_memory": self.arguments.dataloader_pin_memory,
                "shuffle": False,
            }

            data_loader = DataLoader(self.train_dataset, **dataloader_params)

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = jnp.concatenate(reference_chosen_logps)
            all_reference_rejected_logps = jnp.concatenate(reference_rejected_logps)

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True
        return self._get_train_dataloader()

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        full_concat_input_ids = jnp.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = jnp.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: EasyDelFlaxPretrainedModel = None) -> Dict:
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        # add BOS token to head of prompt
        prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
        rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                                                                                         self.label_pad_token_id
                                                                                     ] * len(
            chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                                                                                             self.label_pad_token_id
                                                                                         ] * len(
            rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch

    def compute_reference_log_probs(self, padded_batch: Dict) -> tuple[Any, Any]:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""

        if self.ref_model is None:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.model, padded_batch)
        else:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    def get_mesh(self) -> jax.sharding.Mesh:
        return self.mesh

    def concatenated_forward(
            self, model: EasyDelFlaxPretrainedModel, batch: Dict[str, Union[List, chex.Array]]
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """
        # TODO : Complete concatenated_forward
        ...

    @staticmethod
    def get_batch_logps(
            logits: chex.Array,
            labels: chex.Array,
            average_log_prob: bool = False,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> chex.Array:
        """
        sudo code
        (per_token_logps * loss_mask).sum(-1)
        or
        (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        """

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]

        batch, seq_len, dim = logits.shape
        loss_mask = labels != label_pad_token_id
        labels = jax.lax.select(
            labels == label_pad_token_id,
            jnp.zeros(labels.shape, dtype=labels.dtype),
            labels
        )
        logits_log_s = jax.nn.log_softmax(
            logits, -1
        )
        per_token_logps = jnp.take_along_axis(
            logits_log_s,
            axis=2,
            indices=labels[:, :, None]
        ).reshape(batch, seq_len)

        if average_log_prob:
            log_prob = jnp.sum((per_token_logps * loss_mask), axis=-1) / jnp.sum(loss_mask, axis=-1)
        else:
            log_prob = jnp.sum((per_token_logps * loss_mask), axis=-1)

        return log_prob
