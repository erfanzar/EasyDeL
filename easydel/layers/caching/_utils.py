from enum import Enum

from eformer.pytree import auto_pytree


@auto_pytree
class AttnMaskDetail:
    """Configuration for attention masking patterns.

    Defines the type and parameters of attention masking to apply
    during cache operations. Supports various masking strategies
    including sliding windows, chunks, and custom patterns.

    Attributes:
        mask_type (Enum): Type of attention mask (e.g., FULL, SLIDING, CHUNKED).
        size (int): Primary size parameter for the mask (window size, chunk size, etc.).
        offset (int | None): Optional offset for mask positioning.
        chunks (int | None): Number of chunks for chunked attention.
        bricks (int | None): Number of bricks for blocked attention patterns.
    """

    mask_type: Enum
    size: int
    offset: int | None = None
    chunks: int | None = None
    bricks: int | None = None
