"""Host-side position metadata for Qwen3-Omni image/video/audio prompts."""

import jax
import numpy as np
from jax import numpy as jnp


def build_omni_rope_positions(
    config,
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    use_audio_in_video=False,
    audio_seqlens=None,
    second_per_grids=None,
):
    """Build HF-compatible temporal/spatial positions before compiled decoding.

    Inputs are host token/grid metadata. Audio lengths count mel frames; video
    intervals count seconds per temporal grid (one second when unspecified).
    Returns positions ``[3, batch, seq]`` and continuation deltas ``[batch, 1]``.
    Audio-in-video streams interleave video and audio positions in temporal order,
    choosing video first on ties, as in the processor and pretrained model.
    """
    ids = np.asarray(input_ids)
    valid = np.ones_like(ids, dtype=bool) if attention_mask is None else np.asarray(attention_mask, dtype=bool)
    images = [] if image_grid_thw is None else np.asarray(image_grid_thw).tolist()
    videos = [] if video_grid_thw is None else np.asarray(video_grid_thw).tolist()
    seconds = [1.0] * len(videos) if second_per_grids is None else np.asarray(second_per_grids).tolist()
    audio_lengths = None if audio_seqlens is None else np.asarray(audio_seqlens).tolist()
    merge = config.vision_config.spatial_merge_size
    image_idx = video_idx = audio_idx = 0
    # Video timing may produce fractional positions (e.g. 0.5s * 25/s).
    positions = np.zeros((3, *ids.shape), dtype=np.float32)
    deltas = np.zeros((ids.shape[0], 1), dtype=np.float32)

    def vision_positions(grid, interval, start):
        temporal, height, width = (int(x) for x in grid)
        height, width = height // merge, width // merge
        t, h, w = np.meshgrid(
            np.arange(temporal) * interval * config.position_id_per_seconds,
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        return np.stack((t.ravel(), h.ravel(), w.ravel())).astype(np.float32) + start

    for row, tokens in enumerate(ids):
        tokens = tokens[valid[row]].tolist()
        parts = []
        cursor = offset = 0

        def text(count, parts=parts):
            nonlocal offset
            if count:
                parts.append(np.broadcast_to(np.arange(count) + offset, (3, count)))
                offset += count

        while cursor < len(tokens):
            token = tokens[cursor]
            is_vision = token == config.vision_start_token_id and cursor + 1 < len(tokens)
            next_token = tokens[cursor + 1] if is_vision else None
            joined = is_vision and use_audio_in_video and next_token == config.audio_start_token_id
            image = is_vision and next_token == config.image_token_id
            video = is_vision and next_token == config.video_token_id
            audio = token == config.audio_start_token_id
            if not (image or video or audio or joined):
                text(1)
                cursor += 1
                continue
            bos = 2 if joined else 1
            text(bos)
            cursor += bos
            if image:
                if image_idx >= len(images):
                    raise ValueError("Image placeholders require matching image_grid_thw metadata.")
                media = vision_positions(images[image_idx], 1.0, offset)
                image_idx += 1
            elif video or joined:
                if video_idx >= len(videos):
                    raise ValueError("Video placeholders require matching video_grid_thw metadata.")
                media = vision_positions(videos[video_idx], seconds[video_idx], offset)
                video_idx += 1
            if audio or joined:
                if audio_lengths is None:
                    end = cursor
                    while end < len(tokens) and tokens[end] in (config.audio_token_id, config.video_token_id):
                        end += 1
                    length = tokens[cursor:end].count(config.audio_token_id)
                else:
                    frames = int(audio_lengths[audio_idx])
                    chunk = config.audio_config.n_window * 2
                    length = (frames // chunk) * ((chunk + 7) // 8) + (frames % chunk + 7) // 8
                audio_idx += 1
                audio_positions = np.broadcast_to(np.arange(length) + offset, (3, length))
                if joined:
                    # Stable sort preserves video-first ordering for temporal ties.
                    combined = np.concatenate((media, audio_positions), axis=1)
                    media = combined[:, np.argsort(combined[0], kind="stable")]
                else:
                    media = audio_positions
            count = media.shape[1]
            if cursor + count > len(tokens):
                raise ValueError("Multimodal grid/audio lengths exceed the placeholder token sequence.")
            expected_tokens = (
                {config.video_token_id, config.audio_token_id}
                if joined
                else {config.image_token_id if image else config.video_token_id if video else config.audio_token_id}
            )
            if any(item not in expected_tokens for item in tokens[cursor : cursor + count]):
                raise ValueError("Multimodal grid/audio lengths do not match the placeholder token sequence.")
            parts.append(media)
            cursor += count
            offset = float(media.max()) + 1 if count else offset
            # End markers are ordinary sequential text positions. The next loop
            # consumes them, including both markers for audio-in-video prompts.
        packed = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0), dtype=np.int32)
        positions[:, row, valid[row]] = packed
        deltas[row, 0] = offset - len(tokens)
        if image_grid_thw is None and video_grid_thw is None:
            positions[:, row, ~valid[row]] = 1
    return positions, deltas


def build_omni_rope_positions_traced(
    config,
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    use_audio_in_video=False,
    audio_seqlens=None,
    second_per_grids=None,
):
    """JIT-compatible counterpart for raw-media training with traced metadata.

    A token scan tracks independent spatial/video/audio cursors per batch row.
    It needs no data-dependent shapes, host transfers, or padded visual tokens.
    Concrete generation preparation uses the validated host implementation above.
    """
    ids = jnp.asarray(input_ids)
    valid = jnp.ones_like(ids, dtype=jnp.bool_) if attention_mask is None else attention_mask.astype(jnp.bool_)
    next_ids = jnp.concatenate((ids[:, 1:], jnp.zeros_like(ids[:, :1])), axis=1)
    vision_start = valid & (ids == config.vision_start_token_id)
    image_start = vision_start & (next_ids == config.image_token_id)
    video_start = vision_start & (
        (next_ids == config.video_token_id) | (use_audio_in_video & (next_ids == config.audio_start_token_id))
    )
    # Flattened cumulative indices preserve processor media ordering across rows.
    image_indices = (jnp.cumsum(image_start.reshape(-1)) - 1).reshape(ids.shape)
    video_indices = (jnp.cumsum(video_start.reshape(-1)) - 1).reshape(ids.shape)
    images = jnp.ones((1, 3), jnp.int32) if image_grid_thw is None else jnp.asarray(image_grid_thw)
    videos = jnp.ones((1, 3), jnp.int32) if video_grid_thw is None else jnp.asarray(video_grid_thw)
    seconds = jnp.ones((videos.shape[0],), jnp.float32) if second_per_grids is None else jnp.asarray(second_per_grids)
    merge = config.vision_config.spatial_merge_size
    batch = ids.shape[0]
    scalar = jnp.zeros((batch,), jnp.float32)
    index = jnp.zeros((batch,), jnp.int32)
    flag = jnp.zeros((batch,), jnp.bool_)

    def step(carry, token_data):
        offset, maximum, base, image_cursor, video_cursor, audio_cursor, in_media, expect_audio_bos = carry
        token, following, keep, image_index, video_index = token_data
        is_vision_bos = keep & (token == config.vision_start_token_id)
        is_audio_bos = keep & (token == config.audio_start_token_id)
        bos = is_vision_bos | is_audio_bos
        second_bos = is_audio_bos & expect_audio_bos
        first_bos = bos & ~second_bos
        # Both BOS tokens precede the shared base in an audio/video stream.
        joined = is_vision_bos & use_audio_in_video & (following == config.audio_start_token_id)
        new_base = offset + jnp.where(joined, 2.0, 1.0)
        base = jnp.where(first_bos, new_base, base)
        image_cursor = jnp.where(first_bos, 0, image_cursor)
        video_cursor = jnp.where(first_bos, 0, video_cursor)
        audio_cursor = jnp.where(first_bos, 0, audio_cursor)
        image = keep & (token == config.image_token_id)
        video = keep & (token == config.video_token_id)
        audio = keep & (token == config.audio_token_id)
        media = image | video | audio
        image_grid = images[jnp.clip(image_index, 0, images.shape[0] - 1)]
        video_grid = videos[jnp.clip(video_index, 0, videos.shape[0] - 1)]
        grid = jnp.where(image[:, None], image_grid, video_grid)
        height = jnp.maximum(grid[:, 1] // merge, 1)
        width = jnp.maximum(grid[:, 2] // merge, 1)
        cursor = jnp.where(image, image_cursor, video_cursor)
        temporal = cursor // (height * width)
        temporal = temporal * jnp.where(image, 1.0, seconds[jnp.clip(video_index, 0, seconds.shape[0] - 1)])
        spatial = (
            jnp.stack(
                (
                    temporal * config.position_id_per_seconds,
                    (cursor // width) % height,
                    cursor % width,
                ),
                axis=-1,
            )
            + base[:, None]
        )
        audio_pos = jnp.broadcast_to((base + audio_cursor)[:, None], (batch, 3))
        media_pos = jnp.where(audio[:, None], audio_pos, spatial)
        text_pos = jnp.where(in_media, maximum + 1, offset)
        text_pos = jnp.where(second_bos, offset, text_pos)
        position = jnp.where(media[:, None], media_pos, text_pos[:, None])
        if image_grid_thw is None and video_grid_thw is None:
            padding_value = 1.0
        else:
            padding_value = 0.0
        position = jnp.where(keep[:, None], position, padding_value)
        maximum = jnp.where(first_bos, jnp.max(position, axis=-1), maximum)
        maximum = jnp.where(media, jnp.maximum(maximum, jnp.max(position, axis=-1)), maximum)
        offset = jnp.where(keep, jnp.max(position, axis=-1) + 1, offset)
        image_cursor += image.astype(jnp.int32)
        video_cursor += video.astype(jnp.int32)
        audio_cursor += audio.astype(jnp.int32)
        in_media = jnp.where(keep, media | bos, in_media)
        expect_audio_bos = jnp.where(keep, joined, expect_audio_bos)
        return (
            offset,
            maximum,
            base,
            image_cursor,
            video_cursor,
            audio_cursor,
            in_media,
            expect_audio_bos,
        ), position

    initial = (scalar, scalar, scalar, index, index, index, flag, flag)
    data = tuple(jnp.swapaxes(value, 0, 1) for value in (ids, next_ids, valid, image_indices, video_indices))
    carry, positions = jax.lax.scan(step, initial, data)
    next_offset = jnp.where(carry[6], carry[1] + 1, carry[0])
    return jnp.transpose(positions, (2, 1, 0)), (next_offset - jnp.sum(valid, axis=-1))[:, None]
