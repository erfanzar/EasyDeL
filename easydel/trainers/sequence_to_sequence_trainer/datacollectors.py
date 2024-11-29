from dataclasses import dataclass
from typing import Any, Dict, List, Union

from jax import numpy as jnp


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	max_length: int
	processor: Any
	decoder_start_token_id: int

	def __call__(
		self,
		features: List[Dict[str, Union[List[int], jnp.ndarray]]],
	) -> Dict[str, jnp.ndarray]:
		if not isinstance(features, list):
			features = [features]
		input_features = [
			{"input_features": feature["input_features"]} for feature in features
		]
		batch = self.processor.feature_extractor.pad(
			input_features,
			return_tensors="np",
			max_length=self.max_length,
			padding="max_length",
		)

		label_features = [{"input_ids": feature["labels"]} for feature in features]
		labels_batch = self.processor.tokenizer.pad(
			label_features,
			return_tensors="np",
			max_length=self.max_length,
			padding="max_length",
		)
		labels = jnp.where(
			jnp.not_equal(labels_batch.attention_mask, 1),
			-100,
			labels_batch["input_ids"],
		)
		inp = labels_batch["input_ids"]
		if labels[:, 0] == self.decoder_start_token_id:
			labels = labels[:, 1:]
			inp = inp[:, :-1]

		batch["labels"] = labels
		batch["decoder_input_ids"] = inp

		return batch
