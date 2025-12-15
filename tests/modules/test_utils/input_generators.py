"""Input generation utilities for EasyDeL model testing.

This module provides functions to generate test inputs for various model types,
including text-only models, vision-language models, and audio models.
"""

import jax.numpy as jnp
import numpy as np
import torch


def make_text_inputs(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
) -> dict:
    """Generate text inputs for both PyTorch and JAX.

    Args:
        vocab_size: Size of the vocabulary
        batch_size: Batch size
        seq_len: Sequence length
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'torch' and 'jax' keys containing input tensors
    """
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, vocab_size, (batch_size, seq_len))
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

    return {
        "torch": {
            "input_ids": torch.from_numpy(input_ids).to(torch.long),
            "attention_mask": torch.from_numpy(attention_mask).to(torch.long),
        },
        "jax": {
            "input_ids": jnp.asarray(input_ids, dtype="i4"),
            "attention_mask": jnp.asarray(attention_mask, dtype="bool"),
        },
    }


def make_input_ids(
    vocab_size: int,
    input_shape: tuple[int, int],
    seed: int = 42,
) -> tuple[torch.Tensor, jnp.ndarray]:
    """Generate random input IDs for testing.

    Args:
        vocab_size: Size of the vocabulary
        input_shape: (batch_size, seq_len) shape
        seed: Random seed

    Returns:
        Tuple of (torch_input_ids, jax_input_ids)
    """
    rng = np.random.default_rng(seed)
    np_input_ids = rng.integers(0, vocab_size, input_shape)
    return (
        torch.from_numpy(np_input_ids).to(torch.long),
        jnp.asarray(np_input_ids, dtype="i4"),
    )


def make_vlm_inputs(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    image_token_id: int,
    num_image_tokens: int,
    pixel_values_shape: tuple,
    num_images: int = 1,
    token_type_ids: bool = False,
    image_grid_hws: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    """Generate Vision-Language Model inputs with image token placeholders.

    Args:
        vocab_size: Size of the vocabulary
        batch_size: Batch size
        seq_len: Sequence length
        image_token_id: Token ID used as image placeholder
        num_image_tokens: Number of image tokens per image
        pixel_values_shape: Shape of pixel_values tensor (batch, channels, height, width)
        num_images: Number of images to include
        token_type_ids: Whether to generate token_type_ids (for Gemma3)
        seed: Random seed

    Returns:
        Dictionary with 'torch' and 'jax' keys containing all inputs
    """
    rng = np.random.default_rng(seed)
    input_shape = (batch_size, seq_len)

    # Generate random input IDs
    np_input_ids = rng.integers(0, vocab_size, input_shape)

    # Insert image token placeholders
    total_image_tokens = num_images * num_image_tokens
    if total_image_tokens < seq_len - 1:
        start_pos = 1  # Leave position 0 for BOS if needed
        for img_idx in range(num_images):
            img_start = start_pos + img_idx * num_image_tokens
            img_end = img_start + num_image_tokens
            if img_end <= seq_len:
                np_input_ids[:, img_start:img_end] = image_token_id

    # Create pixel values (random values in reasonable range)
    np_pixel_values = rng.standard_normal(pixel_values_shape).astype(np.float32) * 0.5

    # Create attention mask (all ones)
    np_attention_mask = np.ones(input_shape, dtype=np.int64)

    result = {
        "torch": {
            "input_ids": torch.from_numpy(np_input_ids).to(torch.long),
            "pixel_values": torch.from_numpy(np_pixel_values).to(torch.float32),
            "attention_mask": torch.from_numpy(np_attention_mask).to(torch.long),
        },
        "jax": {
            "input_ids": jnp.asarray(np_input_ids, dtype="i4"),
            "pixel_values": jnp.asarray(np_pixel_values, dtype="f4"),
            "attention_mask": jnp.asarray(np_attention_mask, dtype="bool"),
        },
    }

    if token_type_ids:
        # For Gemma3: token_type_ids distinguishes image (1) vs text (0) tokens
        # HuggingFace convention: 1 = image tokens, 0 = text tokens
        np_token_type_ids = np.zeros(input_shape, dtype=np.int64)
        # Mark image token positions as 1
        for img_idx in range(num_images):
            img_start = 1 + img_idx * num_image_tokens
            img_end = img_start + num_image_tokens
            if img_end <= seq_len:
                np_token_type_ids[:, img_start:img_end] = 1

        result["torch"]["token_type_ids"] = torch.from_numpy(np_token_type_ids).to(torch.long)
        result["jax"]["token_type_ids"] = jnp.asarray(np_token_type_ids, dtype="i4")

    if image_grid_hws is not None:
        image_grid_hws = np.asarray(image_grid_hws, dtype=np.int64)
        result["torch"]["image_grid_hws"] = torch.from_numpy(image_grid_hws).to(torch.long)
        result["jax"]["image_grid_hws"] = jnp.asarray(image_grid_hws, dtype="i4")

    return result


def make_qwen_vlm_inputs(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    num_image_tokens: int,
    pixel_values_shape: tuple,
    image_grid_thw: np.ndarray,
    num_images: int = 1,
    seed: int = 42,
) -> dict:
    """Generate inputs for Qwen2-VL/Qwen3-VL models with mRoPE support.

    Args:
        vocab_size: Size of the vocabulary
        batch_size: Batch size
        seq_len: Sequence length
        image_token_id: Token ID used as image placeholder
        vision_start_token_id: Token ID for vision start marker
        vision_end_token_id: Token ID for vision end marker
        num_image_tokens: Number of image tokens per image
        pixel_values_shape: Shape of pixel_values tensor
        image_grid_thw: Grid shape array (num_images, 3) with [T, H, W] per image
        num_images: Number of images
        seed: Random seed

    Returns:
        Dictionary with 'torch' and 'jax' keys containing all inputs
    """
    rng = np.random.default_rng(seed)
    input_shape = (batch_size, seq_len)

    # Generate random input IDs
    np_input_ids = rng.integers(0, vocab_size, input_shape)

    # Insert vision tokens: <vision_start> + image_tokens + <vision_end>
    tokens_per_image = num_image_tokens + 2  # start + tokens + end
    total_vision_tokens = num_images * tokens_per_image

    if total_vision_tokens < seq_len - 1:
        start_pos = 1
        for img_idx in range(num_images):
            base_pos = start_pos + img_idx * tokens_per_image
            if base_pos + tokens_per_image <= seq_len:
                # Vision start token
                np_input_ids[:, base_pos] = vision_start_token_id
                # Image tokens
                np_input_ids[:, base_pos + 1 : base_pos + 1 + num_image_tokens] = image_token_id
                # Vision end token
                np_input_ids[:, base_pos + 1 + num_image_tokens] = vision_end_token_id

    # Create pixel values
    np_pixel_values = rng.standard_normal(pixel_values_shape).astype(np.float32) * 0.5

    # Create attention mask
    np_attention_mask = np.ones(input_shape, dtype=np.int64)

    # NOTE: Do NOT pass position_ids for Qwen VL models.
    # Both HuggingFace and EasyDeL compute 3D position_ids internally
    # using get_rope_index for proper mRoPE.

    return {
        "torch": {
            "input_ids": torch.from_numpy(np_input_ids).to(torch.long),
            "pixel_values": torch.from_numpy(np_pixel_values).to(torch.float32),
            "attention_mask": torch.from_numpy(np_attention_mask).to(torch.long),
            "image_grid_thw": torch.from_numpy(image_grid_thw).to(torch.long),
        },
        "jax": {
            "input_ids": jnp.asarray(np_input_ids, dtype="i4"),
            "pixel_values": jnp.asarray(np_pixel_values, dtype="f4"),
            "attention_mask": jnp.asarray(np_attention_mask, dtype="bool"),
            "image_grid_thw": jnp.asarray(image_grid_thw, dtype="i4"),
        },
    }


def make_seq2seq_inputs(
    vocab_size: int,
    batch_size: int,
    src_len: int,
    tgt_len: int,
    seed: int = 42,
) -> dict:
    """Generate encoder-decoder inputs for sequence-to-sequence models.

    Args:
        vocab_size: Size of the vocabulary
        batch_size: Batch size
        src_len: Source sequence length (encoder input)
        tgt_len: Target sequence length (decoder input)
        seed: Random seed

    Returns:
        Dictionary with 'torch' and 'jax' keys containing encoder/decoder inputs
    """
    rng = np.random.default_rng(seed)

    # Encoder inputs
    np_input_ids = rng.integers(0, vocab_size, (batch_size, src_len))
    np_attention_mask = np.ones((batch_size, src_len), dtype=np.int64)

    # Decoder inputs
    np_decoder_input_ids = rng.integers(0, vocab_size, (batch_size, tgt_len))
    np_decoder_attention_mask = np.ones((batch_size, tgt_len), dtype=np.int64)

    return {
        "torch": {
            "input_ids": torch.from_numpy(np_input_ids).to(torch.long),
            "attention_mask": torch.from_numpy(np_attention_mask).to(torch.long),
            "decoder_input_ids": torch.from_numpy(np_decoder_input_ids).to(torch.long),
            "decoder_attention_mask": torch.from_numpy(np_decoder_attention_mask).to(torch.long),
        },
        "jax": {
            "input_ids": jnp.asarray(np_input_ids, dtype="i4"),
            "attention_mask": jnp.asarray(np_attention_mask, dtype="bool"),
            "decoder_input_ids": jnp.asarray(np_decoder_input_ids, dtype="i4"),
            "decoder_attention_mask": jnp.asarray(np_decoder_attention_mask, dtype="bool"),
        },
    }


def make_audio_inputs(
    batch_size: int,
    audio_length: int,
    num_mel_bins: int = 80,
    seed: int = 42,
) -> dict:
    """Generate audio inputs for speech models (e.g., Whisper).

    Args:
        batch_size: Batch size
        audio_length: Length of audio sequence (in mel frames)
        num_mel_bins: Number of mel frequency bins
        seed: Random seed

    Returns:
        Dictionary with 'torch' and 'jax' keys containing audio features
    """
    rng = np.random.default_rng(seed)

    # Input features shape: (batch_size, num_mel_bins, audio_length)
    np_input_features = rng.standard_normal((batch_size, num_mel_bins, audio_length)).astype(np.float32)

    return {
        "torch": {
            "input_features": torch.from_numpy(np_input_features).to(torch.float32),
        },
        "jax": {
            "input_features": jnp.asarray(np_input_features, dtype="f4"),
        },
    }


def make_classification_inputs(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    num_labels: int,
    seed: int = 42,
) -> dict:
    """Generate inputs for sequence classification models.

    Args:
        vocab_size: Size of the vocabulary
        batch_size: Batch size
        seq_len: Sequence length
        num_labels: Number of classification labels
        seed: Random seed

    Returns:
        Dictionary with 'torch' and 'jax' keys containing inputs and labels
    """
    rng = np.random.default_rng(seed)

    np_input_ids = rng.integers(0, vocab_size, (batch_size, seq_len))
    np_attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    np_labels = rng.integers(0, num_labels, (batch_size,))

    return {
        "torch": {
            "input_ids": torch.from_numpy(np_input_ids).to(torch.long),
            "attention_mask": torch.from_numpy(np_attention_mask).to(torch.long),
            "labels": torch.from_numpy(np_labels).to(torch.long),
        },
        "jax": {
            "input_ids": jnp.asarray(np_input_ids, dtype="i4"),
            "attention_mask": jnp.asarray(np_attention_mask, dtype="bool"),
            "labels": jnp.asarray(np_labels, dtype="i4"),
        },
    }


def make_image_classification_inputs(
    batch_size: int,
    image_size: int = 224,
    num_channels: int = 3,
    num_labels: int = 1000,
    seed: int = 42,
) -> dict:
    """Generate inputs for image classification models.

    Args:
        batch_size: Batch size
        image_size: Image dimension (height = width)
        num_channels: Number of image channels
        num_labels: Number of classification labels
        seed: Random seed

    Returns:
        Dictionary with 'torch' and 'jax' keys containing pixel values and labels
    """
    rng = np.random.default_rng(seed)

    np_pixel_values = rng.standard_normal((batch_size, num_channels, image_size, image_size)).astype(np.float32)
    np_labels = rng.integers(0, num_labels, (batch_size,))

    return {
        "torch": {
            "pixel_values": torch.from_numpy(np_pixel_values).to(torch.float32),
            "labels": torch.from_numpy(np_labels).to(torch.long),
        },
        "jax": {
            "pixel_values": jnp.asarray(np_pixel_values, dtype="f4"),
            "labels": jnp.asarray(np_labels, dtype="i4"),
        },
    }
