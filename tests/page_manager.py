import unittest

import jax.numpy as jnp

from easydel.inference.vsurge.scheduler import PageManager


class TestPageManager(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.num_pages = 100
        self.max_sequences = 10
        self.page_size = 16
        self.max_num_pages_per_req = 5

    def create_manager(self):
        """Helper to create a fresh PageManager instance."""
        return PageManager.create(
            num_pages=self.num_pages,
            max_sequences=self.max_sequences,
            page_size=self.page_size,
            max_num_pages_per_req=self.max_num_pages_per_req,
        )

    def test_create(self):
        """Test PageManager creation and initial state."""
        manager = self.create_manager()
        self.assertEqual(manager.page_size, self.page_size)
        self.assertEqual(manager.max_sequences, self.max_sequences)
        self.assertTrue(jnp.all(manager.sequence_page_table == -1))
        self.assertTrue(jnp.all(manager.page_ownership == -1))
        self.assertTrue(jnp.all(manager.sequence_lengths == -1))
        # Check initial properties
        self.assertEqual(manager.free_pages, self.num_pages)

    def test_allocate_sequence_slot_success(self):
        """Test successful sequence slot allocation."""
        manager = self.create_manager()
        manager, seq_id = manager.allocate_sequence_slot()
        self.assertEqual(seq_id, 0)  # First slot should be 0
        self.assertEqual(manager.sequence_lengths[0], 0)  # Length should be initialized to 0
        # Allocate another
        manager, seq_id2 = manager.allocate_sequence_slot()
        self.assertEqual(seq_id2, 1)
        self.assertEqual(manager.sequence_lengths[1], 0)

    def test_allocate_sequence_slot_full(self):
        """Test sequence slot allocation when full."""
        manager = self.create_manager()
        # Fill all slots
        for i in range(self.max_sequences):
            manager, seq_id = manager.allocate_sequence_slot()
            self.assertEqual(seq_id, i)
        # Try to allocate one more
        manager, seq_id = manager.allocate_sequence_slot()
        self.assertEqual(seq_id, -1)  # Should indicate failure
        # State should be unchanged
        self.assertTrue(jnp.all(manager.sequence_lengths[: self.max_sequences] >= 0))
        # All previously allocated are active

    def test_release_sequence_slot(self):
        """Test releasing a sequence."""
        manager = self.create_manager()
        manager, seq_id = manager.allocate_sequence_slot()
        self.assertEqual(seq_id, 0)
        self.assertEqual(manager.sequence_lengths[0], 0)
        # Simulate allocating a page to sequence 0
        # We'll directly modify the state for simplicity in this test
        # In a real scenario, you'd use allocate_pages_for_tokens
        # Let's assume page 5 is allocated to sequence 0
        page_num = 5
        manager = PageManager(
            sequence_page_table=manager.sequence_page_table.at[0, 0].set(page_num),
            page_ownership=manager.page_ownership.at[page_num].set(0),
            sequence_lengths=manager.sequence_lengths,
            page_size=manager.page_size,
        )
        self.assertEqual(manager.page_ownership[page_num], 0)
        self.assertEqual(manager.sequence_page_table[0, 0], page_num)
        # Now release the sequence
        manager = manager.release_sequence_slot(0)
        # Check state
        self.assertEqual(manager.sequence_lengths[0], -1)  # Marked inactive
        self.assertEqual(manager.page_ownership[page_num], -1)  # Page ownership released
        self.assertEqual(manager.sequence_page_table[0, 0], -1)  # Page reference cleared

    def test_allocate_pages_for_tokens_single_seq_new(self):
        """Test allocating pages for tokens of a new sequence."""
        manager = self.create_manager()
        manager, seq_id = manager.allocate_sequence_slot()
        self.assertEqual(seq_id, 0)
        num_tokens = 5
        token_sequence_ids = jnp.full((num_tokens,), seq_id, dtype=jnp.int32)
        manager, metadata = manager.allocate_pages_for_tokens(token_sequence_ids)
        # Sequence length should be updated
        self.assertEqual(manager.sequence_lengths[seq_id], num_tokens)
        # One page should be allocated (5 tokens, page size 16)
        pages_allocated = (manager.sequence_lengths[seq_id] + self.page_size - 1) // self.page_size
        self.assertEqual(pages_allocated, 1)
        # Check that one page is now owned
        owned_pages = jnp.sum(manager.page_ownership == seq_id)
        self.assertEqual(owned_pages, 1)
        # Check metadata basics
        self.assertEqual(metadata.num_seqs, 1)
        # The first page allocated should be page 0 (argmin of -1s)
        expected_page_num = jnp.argmin(jnp.full((self.num_pages,), -1, dtype=jnp.int32))
        self.assertEqual(manager.sequence_page_table[seq_id, 0], expected_page_num)
        self.assertEqual(manager.page_ownership[expected_page_num], seq_id)
        # Check slot mapping (assuming page 0 is allocated)
        expected_slots = jnp.arange(num_tokens, dtype=jnp.int32) + expected_page_num * self.page_size
        self.assertTrue(jnp.array_equal(metadata.slot_mapping, expected_slots))
        # Check position IDs
        expected_positions = jnp.arange(num_tokens, dtype=jnp.int32)
        self.assertTrue(jnp.array_equal(metadata.position_ids, expected_positions))
        # Check query_start_loc
        num_valid_seqs = metadata.num_seqs  # Should be 1 in this test case
        self.assertEqual(num_valid_seqs, 1)
        expected_query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
        actual_query_start_loc = metadata.query_start_loc[: num_valid_seqs + 1]
        self.assertTrue(jnp.array_equal(actual_query_start_loc, expected_query_start_loc))

        expected_context_lens = jnp.array([num_tokens], dtype=jnp.int32)
        actual_context_lens = metadata.context_lens[:num_valid_seqs]
        self.assertTrue(jnp.array_equal(actual_context_lens, expected_context_lens))

    def test_allocate_pages_for_tokens_multiple_seqs(self):
        """Test allocating pages for tokens of multiple sequences."""
        manager = self.create_manager()
        manager, seq_id_0 = manager.allocate_sequence_slot()
        manager, seq_id_1 = manager.allocate_sequence_slot()
        self.assertEqual(seq_id_0, 0)
        self.assertEqual(seq_id_1, 1)
        tokens_seq_0 = 10
        tokens_seq_1 = 7
        total_tokens = tokens_seq_0 + tokens_seq_1
        token_sequence_ids = jnp.concatenate(
            [
                jnp.full((tokens_seq_0,), seq_id_0, dtype=jnp.int32),
                jnp.full((tokens_seq_1,), seq_id_1, dtype=jnp.int32),
            ]
        )
        manager, metadata = manager.allocate_pages_for_tokens(token_sequence_ids)
        self.assertEqual(manager.sequence_lengths[seq_id_0], tokens_seq_0)
        self.assertEqual(manager.sequence_lengths[seq_id_1], tokens_seq_1)
        # Check pages allocated
        pages_seq_0 = (tokens_seq_0 + self.page_size - 1) // self.page_size  # 10/16 -> 1
        pages_seq_1 = (tokens_seq_1 + self.page_size - 1) // self.page_size  # 7/16 -> 1
        self.assertEqual(pages_seq_0, 1)
        self.assertEqual(pages_seq_1, 1)
        owned_pages_0 = jnp.sum(manager.page_ownership == seq_id_0)
        owned_pages_1 = jnp.sum(manager.page_ownership == seq_id_1)
        self.assertEqual(owned_pages_0, pages_seq_0)
        self.assertEqual(owned_pages_1, pages_seq_1)
        # Check metadata
        self.assertEqual(metadata.num_seqs, 2)  # Two active sequences involved
        # Check query_start_loc [0, 10, 17]
        # jnp.unique on [0,0,...0, 1,1,...1] finds [0, 1].
        # token_counts becomes [10, 7] for seqs [0, 1].
        # cumsum([0, 10, 7]) = [0, 10, 17].
        num_valid_seqs = metadata.num_seqs  # Should be 2 in this test case
        self.assertEqual(num_valid_seqs, 2)  # Sanity check

        # Check query_start_loc: slice using num_seqs
        expected_query_start_loc = jnp.array([0, tokens_seq_0, total_tokens], dtype=jnp.int32)
        actual_query_start_loc = metadata.query_start_loc[: num_valid_seqs + 1]
        self.assertTrue(jnp.array_equal(actual_query_start_loc, expected_query_start_loc))

        expected_context_lens = jnp.array([tokens_seq_0, tokens_seq_1], dtype=jnp.int32)
        actual_context_lens = metadata.context_lens[:num_valid_seqs]  # Slice to (2,)
        self.assertTrue(jnp.array_equal(actual_context_lens, expected_context_lens))

    def test_allocate_pages_for_tokens_extend_seq(self):
        """Test allocating more pages to extend an existing sequence."""
        manager = self.create_manager()
        manager, seq_id = manager.allocate_sequence_slot()
        self.assertEqual(seq_id, 0)
        # Allocate initial tokens (e.g., 10)
        initial_tokens = 10
        token_sequence_ids_init = jnp.full((initial_tokens,), seq_id, dtype=jnp.int32)
        manager, _ = manager.allocate_pages_for_tokens(token_sequence_ids_init)
        self.assertEqual(manager.sequence_lengths[seq_id], initial_tokens)
        initial_pages = (initial_tokens + self.page_size - 1) // self.page_size
        self.assertEqual(initial_pages, 1)
        self.assertEqual(jnp.sum(manager.page_ownership == seq_id), initial_pages)
        # Allocate more tokens (e.g., 10 more, total 20)
        more_tokens = 10
        total_tokens = initial_tokens + more_tokens
        token_sequence_ids_more = jnp.full((more_tokens,), seq_id, dtype=jnp.int32)
        manager, metadata = manager.allocate_pages_for_tokens(token_sequence_ids_more)
        # Check final state
        self.assertEqual(manager.sequence_lengths[seq_id], total_tokens)
        final_pages = (total_tokens + self.page_size - 1) // self.page_size
        self.assertEqual(final_pages, 2)  # 20/16 = 1.25 -> 2
        self.assertEqual(jnp.sum(manager.page_ownership == seq_id), final_pages)
        self.assertEqual(metadata.num_seqs, 1)
        num_valid_seqs = int(metadata.num_seqs)
        self.assertEqual(num_valid_seqs, 1)
        expected_context_lens = jnp.array([total_tokens], dtype=jnp.int32)
        actual_context_lens = metadata.context_lens[:num_valid_seqs]
        self.assertTrue(jnp.array_equal(actual_context_lens, expected_context_lens))

    def test_get_memory_stats(self):
        """Test the get_memory_stats method."""
        manager = self.create_manager()
        stats_initial = manager.get_memory_stats()
        self.assertEqual(stats_initial["num_pages"], self.num_pages)
        self.assertEqual(stats_initial["used_pages"], 0)
        self.assertEqual(stats_initial["free_pages"], self.num_pages)
        self.assertEqual(stats_initial["utilization"], 0.0)
        self.assertEqual(stats_initial["max_sequences"], self.max_sequences)
        manager, seq_id = manager.allocate_sequence_slot()
        num_tokens = 20
        token_sequence_ids = jnp.full((num_tokens,), seq_id, dtype=jnp.int32)
        manager, _ = manager.allocate_pages_for_tokens(token_sequence_ids)
        stats_after = manager.get_memory_stats()
        expected_pages_used = (num_tokens + self.page_size - 1) // self.page_size  # 20/16 -> 2
        self.assertEqual(stats_after["used_pages"], expected_pages_used)
        self.assertEqual(stats_after["free_pages"], self.num_pages - expected_pages_used)
        self.assertAlmostEqual(stats_after["utilization"], expected_pages_used / self.num_pages)

    # --- Edge Case Tests ---
    def test_allocate_pages_invalid_seq_id(self):
        """Test allocating pages with invalid sequence IDs."""
        manager = self.create_manager()
        # Use seq_id -1 (invalid) for all tokens
        num_tokens = 5
        token_sequence_ids = jnp.full((num_tokens,), -1, dtype=jnp.int32)
        manager_new, metadata = manager.allocate_pages_for_tokens(token_sequence_ids)
        # State should be largely unchanged
        self.assertTrue(jnp.array_equal(manager.sequence_page_table, manager_new.sequence_page_table))
        self.assertTrue(jnp.array_equal(manager.page_ownership, manager_new.page_ownership))
        self.assertTrue(jnp.array_equal(manager.sequence_lengths, manager_new.sequence_lengths))
        # Metadata should reflect no valid sequences processed significantly
        # unique_sequences finds [-1], maps to [max_sequences=10].
        # is_valid = seq_id (-1) >= 0 and < 10 -> False.
        # safe_sequences = 0.
        # page_tables = page_table[0] = [-1, ...] -> because is_valid is False, final page_tables = [-1, ...]
        # num_seqs = sum(is_valid) = 0.
        self.assertEqual(metadata.num_seqs, 0)
        self.assertTrue(jnp.all(metadata.slot_mapping == -1))

    def test_compute_position_ids_complex(self):
        """Test _compute_position_ids with more complex token sequences."""
        # This test targets the internal `_compute_position_ids` logic.
        # Scenario: Tokens for sequences [0, 1, 0, 2, 1, 1]
        # Mock current lengths for sequences [0, 1, 2, ...]: [5, 3, 0, -1, ...]
        # The logic is:
        # relative_positions = position within the contiguous block of the same seq_id in the token_sequence_ids array.
        # sequence_start_positions = sequence_lengths[token_seq_id] - count_of_that_seq_id_in_token_sequence_ids_array
        #                           (This finds the starting global position within the sequence for this batch)
        # position_ids = sequence_start_positions + relative_positions
        manager = self.create_manager()
        mock_lengths = jnp.array([5, 3, 0, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32)
        token_sequence_ids = jnp.array([0, 1, 0, 2, 1, 1], dtype=jnp.int32)
        position_ids = manager._compute_position_ids(token_sequence_ids, mock_lengths)
        expected_positions = jnp.array([3, 0, 3, -1, 0, 1], dtype=jnp.int32)
        self.assertTrue(jnp.array_equal(position_ids, expected_positions))


if __name__ == "__main__":
    unittest.main()
