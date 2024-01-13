import warnings
from collections import defaultdict

import chex
import flax.core
import jax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.collectors import DPODataCollatorWithPadding
from typing import Optional, Literal, Dict, Union, Any, Tuple, List
from .utils import pad_to_length
from ...modules.auto_easydel_model import AutoEasyDelModelForCausalLM
from datasets import Dataset
from jax import numpy as jnp
from ...modules.easydel_modelling_utils import EasyDelFlaxPretrainedModel
from ...trainer.training_configurations import TrainArguments

from transformers import PreTrainedTokenizerBase
from .partitioner_config import PartitionerConfig


class DPOTrainer:
    def __init__(
            self,
            model: EasyDelFlaxPretrainedModel | str = None,
            ref_model: Optional[EasyDelFlaxPretrainedModel | str] = None,
            partitioner_config: Optional[PartitionerConfig] = PartitionerConfig(),
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
        if partitioner_config is None:
            partitioner_config = PartitionerConfig()
        self.partitioner_config = partitioner_config
        assert arguments is not None, (
            "You Have to pass arguments that will be used for training but you have passed"
            "`arguments=None`"
        )
        assert isinstance(arguments, TrainArguments), (
            f"arguments type must be `TrainArguments` but got {type(arguments)}"
        )
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
            model, model_params = AutoEasyDelModelForCausalLM.from_pretrained(
                model,
                **model_init_kwarguments
            )
        else:
            model_params = None
        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoEasyDelModelForCausalLM`"
            )
            ref_model, ref_model_params = AutoEasyDelModelForCausalLM.from_pretrained(
                ref_model,
                **ref_model_init_kwarguments
            )

        else:
            ref_model_params = None
        data_collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=False,
        )
        self.ref_model_params = ref_model_params
        self.model_params = model_params
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs
        self.is_encoder_decoder = False
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

        train_dataset = train_dataset.map(self.tokenize_row, )
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
        prompt_input_ids = jnp.asarray(prompt_input_ids, dtype="i4")
        answer_input_ids = jnp.asarray(answer_input_ids, dtype="i4")
        full_concat_input_ids = jnp.concatenate(
            (
                prompt_input_ids,
                answer_input_ids
            )
        )

        # Prepare input tokens for token by token comparison
        full_input_ids = jnp.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids.tolist() != full_tokenized["input_ids"][:response_token_ids_start_idx]:
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
            chosen_tokens["prompt_input_ids"]
        )
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
            self,
            model: EasyDelFlaxPretrainedModel,
            batch: Dict[str, Union[List, chex.Array]],
            params: flax.core.FrozenDict | dict
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            params=params,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

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

    @staticmethod
    def concatenated_inputs(
            batch: Dict[str, Union[List, chex.Array]],
            is_encoder_decoder: bool = False,
            label_pad_token_id: int = -100,
            padding_value: int = 0,
    ) -> Dict[str, chex.Array]:
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], jax.Array):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    raise KeyError("couldn't find pad_value [Dataset Issue]")
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], jax.Array):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    raise KeyError("couldn't find pad_value [Dataset Issue]")
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = jnp.concatenate(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    axis=0,
                )

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1)
            )

        return concatenated_batch

    def dpo_loss(
            self,
            policy_chosen_logps: chex.Array,
            policy_rejected_logps: chex.Array,
            reference_chosen_logps: chex.Array,
            reference_rejected_logps: chex.Array,
            reference_free: bool = False,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = (
                    -jax.nn.log_sigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - jax.nn.log_sigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = jax.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            chosen_KL = jax.lax.clamp(min=0, x=jnp.mean(policy_chosen_logps - reference_chosen_logps))
            rejected_KL = jax.lax.clamp(min=0, x=jnp.mean(policy_rejected_logps - reference_rejected_logps))

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = jnp.concatenate(
                (
                    1 - jax.nn.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - jax.nn.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
                self.beta
                * (
                        policy_chosen_logps - reference_chosen_logps
                )
        )
        rejected_rewards = (
                self.beta
                * (
                        policy_rejected_logps
                        - reference_rejected_logps
                )
        )

        return losses, chosen_rewards, rejected_rewards

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                string += repr_src if len(repr_src) < 350 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
        return string + ")"

    def __str__(self):
        return self.__repr__()
