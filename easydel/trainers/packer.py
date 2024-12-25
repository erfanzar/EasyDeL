from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
	from datasets import Dataset
else:
	Dataset = tp.Any


def pack_sequences(
	dataset: Dataset,
	max_length: int = 512,
	pad_token_id: int = 0,
	reset_position_ids: bool = False,
	num_proc: tp.Optional[int] = None,
):
	"""
	Pack sequences together with their attention masks and position IDs

	# With continuous position IDs
	packed_dataset = pack_sequences(
			dataset,
			max_length=512,
			pad_token_id=0,
			reset_position_ids=False
	)

	# With reset position IDs for each sequence
	packed_dataset = pack_sequences(
			dataset,
			max_length=512,
			pad_token_id=0,
			reset_position_ids=True
	)

	# Example output format for a packed sequence with two sequences:
	# reset_position_ids=False:
	{
			'input_ids': [seq1_tokens + [PAD] + seq2_tokens + [PAD] + padding],
			'attention_mask': [1,1,1,0,1,1,1,0,0,0],
			'position_ids': [0,1,2,3,4,5,6,7,0,0]
	}

	# reset_position_ids=True:
	{
			'input_ids': [seq1_tokens + [PAD] + seq2_tokens + [PAD] + padding],
			'attention_mask': [1,1,1,0,1,1,1,0,0,0],
			'position_ids': [0,1,2,0,0,1,2,0,0,0]
	}

	Args:
	    dataset: Dataset containing 'input_ids' and 'attention_mask'
	    max_length: Maximum length of packed sequence
	    pad_token_id: Token ID used for padding
	    reset_position_ids: If True, reset position IDs for each sequence in the pack

	Returns:
	    packed_dataset: Dataset with packed sequences, attention masks, and position IDs
	"""

	def pack_examples(examples):
		current_packed_input_ids = []
		current_packed_attention_mask = []
		current_packed_position_ids = []
		current_length = 0

		packed_input_ids = []
		packed_attention_mask = []
		packed_position_ids = []

		def get_position_ids(length, start_position=0):
			if reset_position_ids:
				return list(range(length))
			else:
				return list(range(start_position, start_position + length))

		# Iterate through all examples
		for input_ids, attention_mask in zip(
			examples["input_ids"], examples["attention_mask"]
		):
			seq_length = len(input_ids)

			# If adding this sequence would exceed max_length, start a new packed sequence
			if current_length + seq_length + 1 > max_length:
				# Pad the current packed sequence if needed
				if current_length < max_length:
					padding_length = max_length - current_length
					current_packed_input_ids.extend([pad_token_id] * padding_length)
					current_packed_attention_mask.extend([0] * padding_length)
					current_packed_position_ids.extend([0] * padding_length)

				# Add the completed packed sequence to results
				packed_input_ids.append(current_packed_input_ids)
				packed_attention_mask.append(current_packed_attention_mask)
				packed_position_ids.append(current_packed_position_ids)

				# Start new packed sequence
				current_packed_input_ids = []
				current_packed_attention_mask = []
				current_packed_position_ids = []
				current_length = 0

			# Generate position IDs for current sequence
			position_ids = get_position_ids(seq_length, start_position=current_length)

			# Add current sequence
			current_packed_input_ids.extend(input_ids)
			current_packed_attention_mask.extend(attention_mask)
			current_packed_position_ids.extend(position_ids)

			# Add separator token
			current_packed_input_ids.append(pad_token_id)
			current_packed_attention_mask.append(0)
			current_packed_position_ids.append(
				position_ids[-1] + 1 if not reset_position_ids else 0
			)

			current_length += seq_length + 1

		# Handle the last packed sequence
		if current_packed_input_ids:
			# Pad if needed
			if current_length < max_length:
				padding_length = max_length - current_length
				current_packed_input_ids.extend([pad_token_id] * padding_length)
				current_packed_attention_mask.extend([0] * padding_length)
				current_packed_position_ids.extend([0] * padding_length)

			packed_input_ids.append(current_packed_input_ids)
			packed_attention_mask.append(current_packed_attention_mask)
			packed_position_ids.append(current_packed_position_ids)

		return {
			"input_ids": packed_input_ids,
			"attention_mask": packed_attention_mask,
			"position_ids": packed_position_ids,
		}

	# Process the dataset in batches
	packed_dataset = dataset.map(
		pack_examples,
		batched=True,
		remove_columns=dataset.column_names,
		desc="Packing sequences",
		num_proc=num_proc,
	)

	return packed_dataset
