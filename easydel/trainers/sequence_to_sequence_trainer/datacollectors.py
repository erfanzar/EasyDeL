from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np


def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
	"""
	Shift label ids one token to the right.
	"""
	shifted_label_ids = np.zeros_like(label_ids)
	shifted_label_ids[:, 1:] = label_ids[:, :-1]
	shifted_label_ids[:, 0] = decoder_start_token_id

	return shifted_label_ids


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	processor: Any
	decoder_start_token_id: int
	input_padding: Union[bool, str] = "do_not_pad"
	target_padding: Union[bool, str] = "max_length"
	max_input_length: Optional[float] = None
	max_target_length: Optional[int] = None

	def __call__(
		self, features: List[Dict[str, Union[List[int], np.ndarray]]]
	) -> Dict[str, np.ndarray]:
		model_input_name = self.processor.model_input_names[0]

		input_features = {
			model_input_name: [feature[model_input_name] for feature in features]
		}
		label_features = {"input_ids": [feature["labels"] for feature in features]}

		batch = self.processor.feature_extractor.pad(
			input_features,
			max_length=self.max_input_length,
			padding=self.input_padding,
			return_tensors="np",
		)

		labels_batch = self.processor.tokenizer.pad(
			label_features,
			max_length=self.max_target_length,
			padding=self.target_padding,
			return_tensors="np",
		)

		labels = labels_batch["input_ids"][:, : self.max_target_length]
		labels_batch.attention_mask = labels_batch.attention_mask[
			:, : self.max_target_length
		]
		if (labels[:, 0] == self.decoder_start_token_id).all().item():
			labels = labels[:, 1:]
			labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]

		decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

		labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
		labels = labels.filled(fill_value=-100)

		batch["labels"] = labels
		batch["decoder_input_ids"] = decoder_input_ids
		batch["decoder_attention_mask"] = labels_batch.attention_mask
		# print(
		# 	"|".join(f"{k} - {v.shape}" for k, v in batch.items())
		# 	+ str(self.max_target_length)
		# )

		return batch
