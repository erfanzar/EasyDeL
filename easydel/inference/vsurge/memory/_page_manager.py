from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from easydel.layers.caching import PagedAttentionCacheMetaData, PagedAttentionMetadata


@jax.jit
def _find_next_free_page_index(page_status: jax.Array) -> jax.Array:
    """Finds the index of the next free page in the page status array.

    Args:
        page_status: A JAX array representing the status of each page (0 for free, 1 for used).

    Returns:
        The index of the next free page, or -1 if no free pages are available.
    """
    search_status = page_status[1:]
    overall_free_mask = search_status == 0

    next_free_relative = jnp.argmax(overall_free_mask)
    next_free_overall = next_free_relative + 1
    has_free_overall = jnp.any(overall_free_mask)
    return jnp.where(has_free_overall, next_free_overall, -1)


@partial(jax.jit, static_argnames=("max_pages_per_group",))
def _release_pages_for_group(
    page_state: PagedAttentionMetadata,
    page_group_id: jax.Array,
    max_pages_per_group: int,
) -> PagedAttentionMetadata:
    """Releases pages associated with a specific page group.

    Args:
        page_state: The current page state.
        page_group_id: The ID of the page group to release pages for.
        max_pages_per_group: The maximum number of pages allowed per group.

    Returns:
        The updated page state with released pages.
    """
    current_page_status = page_state.page_status
    current_page_map = page_state.page_map
    num_valid_pages = page_state.num_pages_used[page_group_id]

    def release_page(i: int, status: jax.Array) -> jax.Array:
        is_valid = i < num_valid_pages
        page_idx = current_page_map[page_group_id, i]
        should_release = jnp.logical_and(is_valid, page_idx > 0)

        return jax.lax.cond(
            should_release,
            lambda s: s.at[page_idx].set(0),
            lambda s: s,
            status,
        )

    new_page_status = jax.lax.fori_loop(
        0,
        max_pages_per_group,
        release_page,
        current_page_status,
    )

    return page_state.replace(
        page_status=new_page_status,
        num_pages_used=page_state.num_pages_used.at[page_group_id].set(0),
        sequence_lengths=page_state.sequence_lengths.at[page_group_id].set(0),
        active_page=page_state.active_page.at[page_group_id].set(0),
        has_active_page=page_state.has_active_page.at[page_group_id].set(False),
        active_page_position=page_state.active_page_position.at[page_group_id].set(0),
    )


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _reserve_pages_for_group(
    released_state: PagedAttentionMetadata,
    page_group_id: jax.Array,
    true_length: jax.Array,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PagedAttentionMetadata:
    """Reserves pages for a specific page group.

    Args:
        released_state: The page state after releasing pages.
        page_group_id: The ID of the page group to reserve pages for.
        true_length: The true length of the sequence.
        tokens_per_page: The number of tokens per page.
        max_pages_per_group: The maximum number of pages allowed per group.

    Returns:
        The updated page state with reserved pages.
    """
    num_pages_needed = (true_length + tokens_per_page - 1) // tokens_per_page
    last_token_abs_idx = true_length - 1
    last_page_position_idx = last_token_abs_idx % tokens_per_page
    next_write_position = (last_page_position_idx + 1) % tokens_per_page

    current_page_status = released_state.page_status
    current_page_map = released_state.page_map
    current_num_pages_used = released_state.num_pages_used

    num_free_pages = jnp.sum(current_page_status == 0)
    group_has_capacity = jax.lax.le(num_pages_needed, max_pages_per_group)
    sufficient_free_pages = jax.lax.ge(num_free_pages, num_pages_needed)
    has_enough_resources = jnp.logical_and(sufficient_free_pages, group_has_capacity)

    def allocate_and_update_state(
        initial_state_tuple: tuple[jax.Array, jax.Array, jax.Array],
    ) -> PagedAttentionMetadata:
        """Allocates pages iteratively if resources are sufficient."""
        initial_status, initial_map, initial_num_used = initial_state_tuple

        def allocate_one_page(
            page_idx_in_group: jax.Array,
            loop_state_tuple: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Allocates a single page within the fori_loop."""
            current_loop_status, current_loop_map, current_loop_num_used = loop_state_tuple
            next_free_page_global = _find_next_free_page_index(current_loop_status)
            page_allocated = jax.lax.ge(next_free_page_global, 0)

            new_loop_status = jax.lax.cond(
                page_allocated,
                lambda s: s.at[next_free_page_global].set(1),
                lambda s: s,
                current_loop_status,
            )
            new_loop_map = jax.lax.cond(
                page_allocated,
                lambda m: m.at[page_group_id, page_idx_in_group].set(next_free_page_global),
                lambda m: m,
                current_loop_map,
            )
            new_loop_num_used = jax.lax.cond(
                page_allocated,
                lambda n: n.at[page_group_id].add(1),
                lambda n: n,
                current_loop_num_used,
            )
            return new_loop_status, new_loop_map, new_loop_num_used

        final_page_status, final_page_map, final_num_pages_used = jax.lax.fori_loop(
            0,
            num_pages_needed,
            allocate_one_page,
            (initial_status, initial_map, initial_num_used),
        )
        active_page_global_index = final_page_map[page_group_id, num_pages_needed - 1]

        return released_state.replace(
            page_status=final_page_status,
            page_map=final_page_map,
            num_pages_used=final_num_pages_used,
            sequence_lengths=released_state.sequence_lengths.at[page_group_id].set(true_length),
            active_page=released_state.active_page.at[page_group_id].set(active_page_global_index),
            has_active_page=released_state.has_active_page.at[page_group_id].set(True),
            active_page_position=released_state.active_page_position.at[page_group_id].set(next_write_position),
        )

    final_state = jax.lax.cond(
        has_enough_resources,
        allocate_and_update_state,
        lambda _: released_state,
        operand=(current_page_status, current_page_map, current_num_pages_used),
    )
    return final_state


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _release_and_reserve_for_group(
    page_state: PagedAttentionMetadata,
    page_group_id: jax.Array,
    true_length: jax.Array,
    tokens_per_page: int,
    max_pages_per_group: int,
) -> PagedAttentionMetadata:
    """Releases and reserves pages for a specific page group.

    Args:
        page_state: The current page state.
        page_group_id: The ID of the page group to release and reserve pages for.
        true_length: The true length of the sequence.
        tokens_per_page: The number of tokens per page.
        max_pages_per_group: The maximum number of pages allowed per group.

    Returns:
        The updated page state with released and reserved pages.
    """
    released_state = _release_pages_for_group(
        page_state,
        page_group_id,
        max_pages_per_group,
    )
    final_state = _reserve_pages_for_group(
        released_state,
        page_group_id,
        true_length,
        tokens_per_page,
        max_pages_per_group,
    )
    return final_state


@partial(jax.jit, static_argnames=("tokens_per_page", "max_pages_per_group"))
def _update_decode_pages_global(
    page_state: PagedAttentionMetadata,
    tokens_per_page: jax.Array,
    max_pages_per_group: jax.Array,
) -> PagedAttentionMetadata:
    """Updates page state during decoding.

    Args:
        page_state: The current page state.
        tokens_per_page: The number of tokens per page.
        max_pages_per_group: The maximum number of pages allowed per group.

    Returns:
        The updated page state after decoding.
    """
    max_page_groups = page_state.sequence_lengths.shape[0]

    seq_len_increment = jnp.where(page_state.has_active_page, 1, 0)
    new_sequence_lengths = page_state.sequence_lengths + seq_len_increment

    new_active_page_position = jnp.where(
        page_state.has_active_page,
        (new_sequence_lengths - 1) % tokens_per_page,
        page_state.active_page_position,
    )

    required_pages_per_group = (new_sequence_lengths + tokens_per_page - 1) // tokens_per_page
    needs_new_page_mask = jnp.logical_and(
        page_state.has_active_page, required_pages_per_group > page_state.num_pages_used
    )
    has_capacity_mask = required_pages_per_group <= max_pages_per_group
    needs_allocation_mask = jnp.logical_and(needs_new_page_mask, has_capacity_mask)

    def allocate_for_group_if_needed(
        group_idx: jax.Array,
        current_state: PagedAttentionMetadata,
    ) -> PagedAttentionMetadata:
        current_status = current_state.page_status
        current_map = current_state.page_map
        current_num_used = current_state.num_pages_used
        current_active_page = current_state.active_page

        needs_alloc = needs_allocation_mask[group_idx]
        next_free_page_global = _find_next_free_page_index(current_status)
        can_allocate = jnp.logical_and(needs_alloc, next_free_page_global >= 0)

        new_status = jax.lax.cond(
            can_allocate,
            lambda s: s.at[next_free_page_global].set(1),
            lambda s: s,
            current_status,
        )

        page_map_index = current_num_used[group_idx]
        new_map = jax.lax.cond(
            can_allocate,
            lambda m: m.at[group_idx, page_map_index].set(next_free_page_global),
            lambda m: m,
            current_map,
        )
        new_num_used = jax.lax.cond(can_allocate, lambda n: n.at[group_idx].add(1), lambda n: n, current_num_used)
        new_active_page = jax.lax.cond(
            can_allocate,
            lambda a: a.at[group_idx].set(next_free_page_global),
            lambda a: a,
            current_active_page,
        )

        return current_state.replace(
            page_status=new_status,
            page_map=new_map,
            num_pages_used=new_num_used,
            active_page=new_active_page,
        )

    initial_loop_state = page_state.replace(
        sequence_lengths=new_sequence_lengths,
        active_page_position=new_active_page_position,
    )

    final_state = jax.lax.fori_loop(
        0,
        max_page_groups,
        allocate_for_group_if_needed,
        initial_loop_state,
    )
    return final_state


class PageManager:
    """Manages the allocation and release of pages for paged attention."""

    def __init__(self, metadata: PagedAttentionCacheMetaData):
        """Initializes the PageManager with the given metadata.

        Args:
            metadata: The metadata containing page management parameters.
        """
        self.metadata = metadata
        self.num_pages: int = metadata.num_pages
        self.tokens_per_page: int = metadata.tokens_per_page
        self.max_decodes_length: int = metadata.max_decodes_length
        self.max_page_groups: int = metadata.batch_size
        self.max_pages_per_group: int = metadata.max_pages_per_group

        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """Validates initialization parameters for logical consistency."""
        if self.max_pages_per_group <= 0:
            raise ValueError("`max_pages_per_group` must be positive.")
        min_required = (self.max_decodes_length + self.tokens_per_page - 1) // self.tokens_per_page
        if self.max_pages_per_group < min_required:
            raise ValueError(
                f"`max_pages_per_group` ({self.max_pages_per_group}) is insufficient for `max_decodes_length` "
                f"({self.max_decodes_length}). Needs {min_required}."
            )

        if self.num_pages <= 1:
            raise ValueError("`num_pages` must be greater than 1.")
        if self.tokens_per_page <= 0:
            raise ValueError("`tokens_per_page` must be positive.")
        if self.max_page_groups <= 0:
            raise ValueError("`max_page_groups` must be positive.")

    def update_prefill_pages(
        self,
        page_state: PagedAttentionMetadata,
        page_group_id: int,
        true_length: int,
    ) -> PagedAttentionMetadata:
        """Updates pages during the prefill stage.

        Args:
            page_state: The current page state.
            page_group_id: The ID of the page group to update pages for.
            true_length: The true length of the sequence.

        Returns:
            The updated page state after prefill.
        """
        if page_group_id < 0 or page_group_id >= self.max_page_groups:
            raise ValueError(f"PageManager: page_group_id ({page_group_id}) out of range [0, {self.max_page_groups})")
        if true_length <= 0 or true_length > self.max_decodes_length:
            raise ValueError(f"PageManager: true_length ({true_length}) out of range (0, {self.max_decodes_length}]")

        return _release_and_reserve_for_group(
            page_state,
            page_group_id,
            true_length,
            self.tokens_per_page,
            self.max_pages_per_group,
        )

    def update_decode_pages(
        self,
        page_state: PagedAttentionMetadata,
    ) -> PagedAttentionMetadata:
        """Updates pages during the decode stage.

        Args:
            page_state: The current page state.

        Returns:
            The updated page state after decode.
        """
        return _update_decode_pages_global(page_state, self.tokens_per_page, self.max_pages_per_group)

    def release_pages(
        self,
        page_state: PagedAttentionMetadata,
        page_group_id: int,
    ) -> PagedAttentionMetadata:
        """Releases pages associated with a specific page group.

        Args:
            page_state: The current page state.
            page_group_id: The ID of the page group to release pages for.

        Returns:
            The updated page state with released pages.
        """
        if page_group_id < 0 or page_group_id >= self.max_page_groups:
            raise ValueError(f"PageManager: page_group_id ({page_group_id}) out of range [0, {self.max_page_groups})")
        return _release_pages_for_group(page_state, page_group_id, self.max_pages_per_group)

    def get_initial_page_state(self) -> PagedAttentionMetadata:
        """Gets the initial page state.

        Returns:
            The initial page state.
        """
        return PagedAttentionMetadata.create(
            num_pages=self.num_pages,
            max_page_groups=self.max_page_groups,
            max_pages_per_group=self.max_pages_per_group,
        )
