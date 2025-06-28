# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from easydel.inference.vsurge.managers.block_manager import BlockAllocator, BlockManager
from easydel.inference.vsurge.managers.model_io import ModelIOProcessor
from easydel.inference.vsurge.managers.scheduler import BatchInfo, PriorityScheduler, SchedulerConfig
from easydel.inference.vsurge.managers.sequence import SequenceMetadata, SequenceState, SequenceStatus

BLOCK_SIZE = 4
NUM_BLOCKS = 32
EOS_ID = 2
MAX_BATCH_SIZE = 16
MAX_SEQ_LENGTH = 1024


@pytest.fixture
def sequence_counter():
    """Reset sequence ID counter for each test."""
    counter = {"value": 0}

    def get_next():
        val = counter["value"]
        counter["value"] += 1
        return val

    return get_next


@pytest.fixture
def create_sequence(sequence_counter):
    """Factory for creating test sequences."""

    def _create(prompt_tokens: list[int], max_tokens: int = 10) -> SequenceState:
        metadata = SequenceMetadata(
            seq_id=sequence_counter(), block_size=BLOCK_SIZE, max_tokens=max_tokens, temperature=1.0, top_p=1.0, top_k=-1
        )
        return SequenceState(metadata=metadata, prompt_tokens=np.array(prompt_tokens, dtype=np.int32))

    return _create


class TestBlockAllocator:
    def test_allocate_deallocate(self):
        allocator = BlockAllocator(NUM_BLOCKS)
        blocks = []
        for _ in range(5):
            blocks.append(allocator.allocate())

        assert allocator.num_free == NUM_BLOCKS - 5
        assert len(allocator.allocated) == 5
        for block in blocks:
            allocator.deallocate(block)

        assert allocator.num_free == NUM_BLOCKS
        assert len(allocator.allocated) == 0

    def test_out_of_memory(self):
        allocator = BlockAllocator(2)
        allocator.allocate()
        allocator.allocate()

        with pytest.raises(MemoryError):
            allocator.allocate()


class TestBlockManager:
    def test_allocate_free_blocks(self, create_sequence):
        bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        blocks = bm.allocate_blocks(2)
        assert len(blocks) == 2
        assert bm.allocator.num_free == NUM_BLOCKS - 2
        for bid in blocks:
            assert bm.blocks[bid].ref_count == 1
        for bid in blocks:
            bm.decrement_ref(bid)

        assert bm.allocator.num_free == NUM_BLOCKS

    def test_copy_on_write(self):
        bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        blocks = bm.allocate_blocks(1)
        original_block = blocks[0]
        bm.increment_ref(original_block)
        assert bm.blocks[original_block].ref_count == 2
        new_block, copied = bm.copy_on_write(original_block)
        assert copied
        assert new_block != original_block
        assert bm.blocks[original_block].ref_count == 1
        assert bm.blocks[new_block].ref_count == 1
        same_block, copied = bm.copy_on_write(new_block)
        assert not copied
        assert same_block == new_block

    def test_insufficient_blocks(self):
        bm = BlockManager(2, BLOCK_SIZE)

        with pytest.raises(MemoryError):
            bm.allocate_blocks(3)


class TestSequenceState:
    def test_token_management(self, create_sequence):
        seq = create_sequence([1, 2, 3, 4, 5])

        assert seq.num_tokens == 5
        assert seq.num_blocks == 2
        assert seq.last_block_usage == 1
        seq.append_token(6)
        assert seq.num_tokens == 6
        assert len(seq.output_tokens) == 1
        assert seq.all_tokens[-1] == 6

    def test_block_tokens(self, create_sequence):
        seq = create_sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])

        block0_tokens = seq.get_block_tokens(0)
        assert_array_equal(block0_tokens, [1, 2, 3, 4])

        block1_tokens = seq.get_block_tokens(1)
        assert_array_equal(block1_tokens, [5, 6, 7, 8])

        block2_tokens = seq.get_block_tokens(2)
        assert_array_equal(block2_tokens, [9])

    def test_fork(self, create_sequence):
        seq = create_sequence([1, 2, 3])
        seq.append_token(4)
        seq.block_table = [0, 1]

        forked = seq.fork()

        assert forked.prompt_tokens is seq.prompt_tokens
        assert forked.output_tokens == seq.output_tokens
        assert forked.output_tokens is not seq.output_tokens
        assert forked.block_table == seq.block_table
        assert forked.block_table is not seq.block_table


class TestScheduler:
    @pytest.fixture
    def scheduler_config(self):
        return SchedulerConfig(
            max_batch_size=MAX_BATCH_SIZE,
            max_sequence_length=MAX_SEQ_LENGTH,
            block_size=BLOCK_SIZE,
            num_blocks=NUM_BLOCKS,
            eos_token_ids={EOS_ID},
        )

    def test_preemption(self, create_sequence, scheduler_config):
        bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        scheduler = PriorityScheduler(scheduler_config, bm)

        tokens_needed = (NUM_BLOCKS - 3) * BLOCK_SIZE
        long_seq = create_sequence([1] * tokens_needed)
        scheduler.add_sequence(long_seq)
        batch = scheduler.schedule()
        assert batch is not None, "Expected to schedule long sequence"
        assert batch.is_prefill
        assert long_seq in batch.sequences

        assert long_seq.metadata.seq_id in scheduler.running_sequences

        free_blocks = bm.allocator.num_free
        assert free_blocks < 4, f"Expected < 4 free blocks, got {free_blocks}"

        new_seq = create_sequence([2] * 20)
        scheduler.add_sequence(new_seq, priority=0.0)

        batch = scheduler.schedule()
        assert batch is not None, "Expected to schedule new sequence after preemption"

        assert batch.is_prefill
        assert new_seq in batch.sequences
        assert long_seq.status == SequenceStatus.WAITING
        assert new_seq.status == SequenceStatus.RUNNING
        assert long_seq.metadata.seq_id not in scheduler.running_sequences
        assert any(seq.metadata.seq_id == long_seq.metadata.seq_id for _, _, seq in scheduler.waiting_queue)

    def test_finish_sequences(self, create_sequence, scheduler_config):
        bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        scheduler = PriorityScheduler(scheduler_config, bm)
        seq1 = create_sequence([1], max_tokens=2)
        seq2 = create_sequence([2], max_tokens=10)
        scheduler.add_sequence(seq1)
        scheduler.add_sequence(seq2)
        batch = scheduler.schedule()
        assert batch is not None
        assert batch.is_prefill
        assert len(batch.sequences) == 2

        finished = scheduler.finish_sequences(batch, [3, 4])
        assert len(finished) == 0
        assert len(seq1.output_tokens) == 1
        assert seq1.status == SequenceStatus.RUNNING
        batch = scheduler.schedule()
        assert batch is not None
        assert not batch.is_prefill
        finished = scheduler.finish_sequences(batch, [5, EOS_ID])

        assert len(finished) == 2
        assert seq1 in finished
        assert seq2 in finished
        assert seq1.status == SequenceStatus.FINISHED
        assert seq2.status == SequenceStatus.FINISHED
        assert len(seq1.output_tokens) == 2
        assert seq2.output_tokens[-1] == EOS_ID

    def test_empty_schedule(self, scheduler_config):
        """Test that schedule returns None when no sequences are available."""
        bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        scheduler = PriorityScheduler(scheduler_config, bm)
        batch = scheduler.schedule()
        assert batch is None

    def test_schedule_decode_only(self, create_sequence, scheduler_config):
        """Test decode scheduling when no prefill sequences are waiting."""
        bm = BlockManager(NUM_BLOCKS, BLOCK_SIZE)
        scheduler = PriorityScheduler(scheduler_config, bm)
        seq = create_sequence([1, 2, 3])
        scheduler.add_sequence(seq)
        batch = scheduler.schedule()
        assert batch is not None
        assert batch.is_prefill
        batch = scheduler.schedule()
        assert batch is not None
        assert not batch.is_prefill
        assert len(batch.sequences) == 1
        assert seq in batch.sequences


class TestModelIO:
    @pytest.fixture
    def io_processor(self):
        return ModelIOProcessor(BLOCK_SIZE, MAX_BATCH_SIZE, NUM_BLOCKS)

    def test_prefill_input(self, create_sequence, io_processor):
        seq1 = create_sequence([10, 20, 30, 40, 50])
        seq1.block_table = [0, 1]

        seq2 = create_sequence([100, 200])
        seq2.block_table = [2]

        batch = BatchInfo([seq1, seq2], is_prefill=True, total_tokens=7)
        model_input = io_processor.create_model_input(batch)

        expected_tokens = [10, 20, 30, 40, 50, 100, 200]
        assert_array_equal(model_input.input_ids, expected_tokens)

        expected_positions = [0, 1, 2, 3, 4, 0, 1]
        assert_array_equal(model_input.position_ids, expected_positions)

        expected_slots = [0, 1, 2, 3, 4, 8, 9]
        assert_array_equal(model_input.attention_metadata.slot_mapping, expected_slots)
        expected_cu_seqlens = [0, 5, 7]
        assert_array_equal(model_input.attention_metadata.cu_seqlens_q, expected_cu_seqlens)

    def test_decode_input(self, create_sequence, io_processor):
        seq1 = create_sequence([10])
        seq1.append_token(20)
        seq1.append_token(30)
        seq1.block_table = [5]

        seq2 = create_sequence([100, 200])
        seq2.append_token(300)
        seq2.block_table = [6]

        batch = BatchInfo([seq1, seq2], is_prefill=False, total_tokens=2)
        model_input = io_processor.create_model_input(batch)
        expected_tokens = [30, 300]
        assert_array_equal(model_input.input_ids, expected_tokens)
        expected_positions = [2, 2]
        assert_array_equal(model_input.position_ids, expected_positions)
        expected_slots = [22, 26]
        assert_array_equal(model_input.attention_metadata.slot_mapping, expected_slots)
        expected_context_lens = [3, 3]
        assert_array_equal(model_input.attention_metadata.context_lens, expected_context_lens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
