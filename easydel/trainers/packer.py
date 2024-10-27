import math
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


class SequencePacker:
	def __init__(
		self,
		max_length: int = 512,
		pad_token_id: int = 0,
		num_workers: Optional[int] = None,
	):
		"""
		Initialize the sequence packer.

		Args:
		    max_length: Maximum length of packed sequences
		    pad_token_id: Token ID to use for padding
		    num_workers: Number of CPU workers to use. Defaults to cpu_count() - 1
		"""
		self.max_length = max_length
		self.pad_token_id = pad_token_id
		self.num_workers = (
			num_workers if num_workers is not None else max(1, cpu_count() - 1)
		)

	@staticmethod
	def _standardize_length(
		sequences: List[List[int]], target_length: int, pad_value: int
	) -> List[List[int]]:
		"""
		Ensure all sequences have the same length through padding.
		"""
		return [seq + [pad_value] * (target_length - len(seq)) for seq in sequences]

	@staticmethod
	def _process_chunk(args):
		"""
		Process a chunk of sequences in parallel.

		Args:
		    args: Tuple of (chunk_sequences, max_length, pad_token_id)

		Returns:
		    List of packed sequences for this chunk
		"""
		chunk_sequences, chunk_attention_masks, max_length, pad_token_id = args
		chunk_lengths = [len(seq) for seq in chunk_sequences]
		packed_groups = []
		current_group = []
		current_length = 0

		for i, length in enumerate(chunk_lengths):
			if current_length + length <= max_length:
				current_group.append(i)
				current_length += length
			else:
				if current_group:
					packed_groups.append(current_group)
				current_group = [i]
				current_length = length

		if current_group:
			packed_groups.append(current_group)

		# Pack sequences in this chunk
		chunk_packed_input_ids = []
		chunk_packed_attention_mask = []

		for group in packed_groups:
			current_input_ids = []
			current_attention_mask = []

			for idx in group:
				current_input_ids.extend(chunk_sequences[idx])
				current_attention_mask.extend(chunk_attention_masks[idx])

			# Pad to max_length
			padding_length = max_length - len(current_input_ids)
			if padding_length > 0:
				current_input_ids.extend([pad_token_id] * padding_length)
				current_attention_mask.extend([0] * padding_length)
			else:
				# Truncate if somehow longer than max_length
				current_input_ids = current_input_ids[:max_length]
				current_attention_mask = current_attention_mask[:max_length]

			chunk_packed_input_ids.append(current_input_ids)
			chunk_packed_attention_mask.append(current_attention_mask)

		return chunk_packed_input_ids, chunk_packed_attention_mask

	def pack_sequences(
		self, input_ids: List[List[int]], attention_mask: List[List[int]]
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Pack sequences together using multiprocessing.

		Args:
		    input_ids: List of input ID sequences
		    attention_mask: List of attention mask sequences

		Returns:
		    Tuple of (packed_input_ids, packed_attention_mask) as numpy arrays
		"""
		total_sequences = len(input_ids)
		chunk_size = math.ceil(total_sequences / self.num_workers)

		# Split data into chunks
		chunks = []
		for i in range(0, total_sequences, chunk_size):
			end_idx = min(i + chunk_size, total_sequences)
			chunks.append(
				(
					input_ids[i:end_idx],
					attention_mask[i:end_idx],
					self.max_length,
					self.pad_token_id,
				)
			)

		# Process chunks in parallel
		with Pool(self.num_workers) as pool:
			results = list(
				tqdm(
					pool.imap(self._process_chunk, chunks),
					total=len(chunks),
					desc="Processing chunks",
				)
			)

		# Combine results
		all_packed_input_ids = []
		all_packed_attention_mask = []

		for chunk_input_ids, chunk_attention_mask in results:
			all_packed_input_ids.extend(chunk_input_ids)
			all_packed_attention_mask.extend(chunk_attention_mask)

		# Convert to numpy arrays first to ensure shapes are correct
		np_input_ids = np.array(all_packed_input_ids, dtype=np.int32)
		np_attention_mask = np.array(all_packed_attention_mask, dtype=np.int32)

		# Convert to JAX arrays
		return (jnp.array(np_input_ids), jnp.array(np_attention_mask))

	def __call__(self, dataset: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
		"""
		Process a dataset with input_ids and attention_mask.

		Args:
		    dataset: Dictionary containing 'input_ids' and 'attention_mask'

		Returns:
		    Tuple of (packed_input_ids, packed_attention_mask) as JAX arrays
		"""
		print(
			f"Processing dataset with {len(dataset['input_ids'])} sequences using {self.num_workers} workers"
		)
		return self.pack_sequences(dataset["input_ids"], dataset["attention_mask"])

	@staticmethod
	def get_boundaries(packed_attention_mask: jnp.ndarray) -> jnp.ndarray:
		"""
		Get the boundaries between packed sequences.

		Args:
		    packed_attention_mask: Packed attention mask array

		Returns:
		    Array of indices where sequences start
		"""
		transitions = jnp.diff(packed_attention_mask, axis=1)
		sequence_starts = jnp.where(transitions == 1)[1] + 1
		return jnp.concatenate([jnp.array([0]), sequence_starts])
