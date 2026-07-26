# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``process_encoder`` -- run a VLM's vision tower on the fly, outside the train step.

A vision-language batch normally carries ``pixel_values`` straight into the jitted
train step, where the model runs its vision tower, merges the resulting patch embeddings
into the text embeddings at placeholder positions, and backpropagates through the whole
thing. ``process_encoder`` lifts the tower out of the step: it runs it once under
``stop_gradient`` and hands the resulting patch embeddings to the step as a precomputed
tensor. Every VLM family in the zoo already accepts precomputed features -- their merge
sites read ``if image_features is None and pixel_values is not None:`` -- so no modeling
code changes; this module only has to find the right keyword per family.

What that buys, on the current data path:

1. **No vision backward.** Tower activations are not retained for a backward pass that is
   discarded whenever the tower is not the thing being trained. On a 300-400M tower this
   is the dominant saving, and it applies to every step.
2. **A fixed step signature.** ``pixel_values`` is image-major and dense (one row per real
   image), so its leading dimension moves with the number of images in the batch and the
   jitted step recompiles per distinct count. Bucketing the tower's input and zero-padding
   the emitted features to a constant width collapses that to one signature.

What it does *not* currently buy: skipping "dead" rows. That would need a batch carrying
images for only some of its sequences, and the collators do not produce one -- mixed
presence is either rejected outright (``"GRPO batches must not mix present and missing
generation kwargs"``) or the key is dropped for the whole batch. Row packing is therefore
implemented and tested but reachable only via an explicit ``presence_key``, for a data path
that can represent mixed batches; :class:`PixelLayout` records which case a batch is in.

Correctness rests on how :meth:`BaseVisionLanguageModule.merge_multimodal_embeddings`
places features. It is purely positional::

    gather_indices = jnp.cumsum(flat_mask)          # k-th placeholder <- features[k - 1]
    update_values = flattened_padded[gather_indices]

Two properties follow, and this module is built on both:

* **Dropping text-only rows is exact.** Such rows contain no placeholder tokens, so they
  consume no features. Preserving the relative order of the surviving rows is the only
  requirement.
* **Trailing features are never read.** ``gather_indices`` never exceeds the total
  placeholder count, so any feature rows past the last placeholder are inert.

The second property is what keeps compilation bounded. The tower runs on a *bucketed*
row count (few distinct shapes, real work saved), while the tensor handed to the train
step is zero-padded back to a *constant* shape, so the step itself never sees a new
signature and never recompiles. Padding costs a little bandwidth and no compute.

Gradient semantics: the tower is a frozen feature extractor for any step this runs on.
That is the point -- it is what buys the skipped backward -- but it means a run that
intends to *train* the vision tower must leave ``process_encoder`` disabled and use the
normal in-model path. :func:`validate_process_encoder` refuses the combination rather
than silently freezing a tower the optimizer was told to update.
"""

from __future__ import annotations

import inspect
import typing as tp
from dataclasses import asdict, dataclass
from enum import StrEnum

import jax
import numpy as np
from eformer.loggings import get_logger
from jax import numpy as jnp

logger = get_logger(__name__)

# Keyword names families use for already-computed vision features. The merge sites are
# uniform in shape (`if <name> is None and pixel_values is not None:`) but not in name:
# 9 families say `image_features`, 9 say `image_embeds`.
FEATURE_KWARG_CANDIDATES: tuple[str, ...] = ("image_features", "image_embeds", "visual_embeds")

# Attribute names under which a vision tower is stored. `BaseVisionLanguageModule`
# advertises its own via `_vision_tower_name`; these are the fallbacks.
VISION_TOWER_ATTR_CANDIDATES: tuple[str, ...] = ("vision_tower", "visual", "vision_model")

# Batch keys that describe images and must be packed in lockstep with `pixel_values`.
VISION_COMPANION_KEYS: tuple[str, ...] = (
    "pixel_attention_mask",
    "image_grid_thw",
    "image_grid_hws",
    "image_sizes",
    "image_position_ids",
)

PIXEL_KEY = "pixel_values"

# Sentinel ceiling for bucketing when no cap applies (image-major batches).
_NO_ROW_CEILING = 1 << 62


@dataclass
class ProcessEncoderConfig:
    """Configuration for on-the-fly vision-encoder processing.

    Attributes:
        enabled: Master switch; when ``False`` every entry point is a no-op.
        row_bucket_multiple: Packed vision-row counts are rounded up to a multiple of
            this before the tower runs, so the encoder compiles a handful of shapes
            instead of one per distinct image count. Coarser means fewer compilations
            and more wasted rows; ``1`` disables bucketing (and invites a compilation
            per distinct count). Ignored when the batch is fully vision-bearing.
        max_rows_per_step: Optional hard cap on packed rows per step. Rows past the cap
            keep their in-model path (``pixel_values`` is retained) rather than being
            silently dropped, so no image is lost.
        feature_dtype: Optional dtype string (e.g. ``"bfloat16"``) for the emitted
            features. ``None`` keeps the tower's output dtype, which avoids a cast.
        presence_key: Optional batch key holding a per-row boolean/int vision mask. When
            absent, presence is derived from ``pixel_values`` itself, which costs one
            small device-to-host transfer per step.
        only_buckets: Optional list of training-bucket indices this applies to; steps
            routed elsewhere keep the in-model path. ``None`` applies on every step.
        require_support: When ``True`` (default) a model that cannot accept precomputed
            features raises at trainer construction. When ``False`` the trainer logs once
            and falls back to the in-model vision path.
    """

    enabled: bool = False
    row_bucket_multiple: int = 8
    max_rows_per_step: int | None = None
    feature_dtype: str | None = None
    presence_key: str | None = None
    only_buckets: list[int] | None = None
    require_support: bool = True

    def __post_init__(self) -> None:
        if self.row_bucket_multiple < 1:
            raise ValueError(
                f"process_encoder.row_bucket_multiple must be >= 1, got {self.row_bucket_multiple}."
            )
        if self.max_rows_per_step is not None and self.max_rows_per_step < 1:
            raise ValueError(
                f"process_encoder.max_rows_per_step must be None or >= 1, got {self.max_rows_per_step}."
            )
        if self.feature_dtype is not None:
            try:
                jnp.dtype(self.feature_dtype)
            except TypeError as exc:
                raise ValueError(
                    f"process_encoder.feature_dtype must name a valid dtype, got {self.feature_dtype!r}."
                ) from exc
        if self.only_buckets is not None:
            buckets = list(self.only_buckets)
            if not buckets:
                raise ValueError(
                    "process_encoder.only_buckets must be None or a non-empty list of bucket indices, got []."
                )
            if any(not isinstance(b, int) or isinstance(b, bool) or b < 0 for b in buckets):
                raise ValueError(
                    f"process_encoder.only_buckets entries must be non-negative ints, got {self.only_buckets}."
                )
            self.only_buckets = buckets

    def applies_to_bucket(self, bucket_index: int | None) -> bool:
        """Whether processing should run on a step routed to ``bucket_index``.

        ``None`` ``only_buckets`` matches every step. When restricted, a step matches only
        if the trainer reports a bucket index in the list (``bucket_index=None`` -- no
        buckets configured -- never matches a restricted config).
        """
        if self.only_buckets is None:
            return True
        return bucket_index is not None and bucket_index in self.only_buckets

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any] | "ProcessEncoderConfig") -> "ProcessEncoderConfig":
        """Build a process-encoder config from a serialized dictionary."""
        if isinstance(data, cls):
            return data
        return cls(**data)


@dataclass(frozen=True)
class EncoderBinding:
    """How a specific model exposes its vision tower and accepts precomputed features.

    Attributes:
        tower_attr: Attribute path on the model holding the vision tower.
        feature_kwarg: Keyword the model's forward accepts precomputed features under.
        feature_fn_name: Name of the method that runs the tower (``get_image_features``).
        accepted_kwargs: Parameter names ``feature_fn_name`` accepts, so only the batch
            keys a family actually wants are forwarded.
    """

    tower_attr: str
    feature_kwarg: str
    feature_fn_name: str
    accepted_kwargs: frozenset[str]


def _unwrap(model: tp.Any) -> tp.Any:
    """Return the object that owns the VLM surface, descending one ``base_model`` hop."""
    return model


def find_vision_tower_attr(model: tp.Any) -> str | None:
    """Locate the attribute holding ``model``'s vision tower, or ``None``.

    Prefers the family's own advertised ``_vision_tower_name`` before falling back to the
    three names used across the zoo.
    """
    advertised = getattr(model, "_vision_tower_name", None)
    candidates: tuple[str, ...] = VISION_TOWER_ATTR_CANDIDATES
    if isinstance(advertised, str) and advertised:
        candidates = (advertised, *[c for c in VISION_TOWER_ATTR_CANDIDATES if c != advertised])

    for holder_name in ("", "base_model"):
        holder = model if not holder_name else getattr(model, holder_name, None)
        if holder is None:
            continue
        for name in candidates:
            if getattr(holder, name, None) is not None:
                return name if not holder_name else f"{holder_name}.{name}"
    return None


def _resolve_attr(obj: tp.Any, path: str) -> tp.Any:
    """Resolve a dotted attribute path."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _forward_accepts(model: tp.Any, name: str) -> bool:
    """Whether any forward-ish entry point of ``model`` names parameter ``name``.

    Families accept precomputed features on the base model's ``forward`` /
    ``compute_embedding`` rather than on the task head, and several accept it only via
    ``**kwargs`` pass-through, so the source of the merge site is the reliable signal.
    """
    targets = []
    for holder in (model, getattr(model, "base_model", None)):
        if holder is None:
            continue
        for attr in ("forward", "compute_embedding", "__call__"):
            fn = getattr(holder, attr, None)
            if fn is not None:
                targets.append(fn)

    for fn in targets:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if name in params:
            return True
    return False


def resolve_encoder_binding(model: tp.Any) -> EncoderBinding | None:
    """Discover how to drive ``model``'s vision tower out of band.

    Returns ``None`` when the model has no vision tower, exposes no
    ``get_image_features``, or accepts no precomputed-feature keyword -- all three are
    "fall back to the in-model path" conditions rather than errors.

    Args:
        model: A VLM task head (or its base model).

    Returns:
        The binding, or ``None`` when out-of-band processing is not possible.
    """
    tower_attr = find_vision_tower_attr(model)
    if tower_attr is None:
        return None

    feature_fn = getattr(model, "get_image_features", None)
    if feature_fn is None or not callable(feature_fn):
        return None

    feature_kwarg = next((name for name in FEATURE_KWARG_CANDIDATES if _forward_accepts(model, name)), None)
    if feature_kwarg is None:
        return None

    try:
        accepted = frozenset(inspect.signature(feature_fn).parameters)
    except (TypeError, ValueError):
        accepted = frozenset({PIXEL_KEY})

    return EncoderBinding(
        tower_attr=tower_attr,
        feature_kwarg=feature_kwarg,
        feature_fn_name="get_image_features",
        accepted_kwargs=accepted,
    )


class PixelLayout(StrEnum):
    """How a batch's ``pixel_values`` leading axis is organized.

    Attributes:
        IMAGE_MAJOR: One row per real image, densely packed. What every current collator
            emits; there are no dead rows to skip, so only bucketing applies.
        ROW_MAJOR: One row per batch sequence, so text-only sequences occupy rows that can
            be skipped. Requires a data path that represents mixed batches.
    """

    IMAGE_MAJOR = "image_major"
    ROW_MAJOR = "row_major"


def resolve_pixel_layout(batch: tp.Mapping[str, tp.Any], presence_key: str | None = None) -> PixelLayout:
    """Classify a batch's pixel layout.

    ``ROW_MAJOR`` is claimed only on positive evidence -- an explicit presence mask, or a
    leading dimension equal to the batch size *and* at least one all-zero row. A dense
    all-vision batch whose image count happens to equal the batch size is correctly left
    as ``IMAGE_MAJOR``, since packing it would be a no-op either way.
    """
    if presence_key is not None and presence_key in batch:
        return PixelLayout.ROW_MAJOR

    pixels = batch.get(PIXEL_KEY)
    sequences = batch.get("input_ids")
    if pixels is None or sequences is None or getattr(pixels, "ndim", 0) < 2:
        return PixelLayout.IMAGE_MAJOR
    if pixels.shape[0] != sequences.shape[0]:
        return PixelLayout.IMAGE_MAJOR

    reduce_axes = tuple(range(1, pixels.ndim))
    has_content = np.asarray(jax.device_get(jnp.any(pixels != 0, axis=reduce_axes)))
    return PixelLayout.ROW_MAJOR if not has_content.all() else PixelLayout.IMAGE_MAJOR


def grid_describes_rows(batch: tp.Mapping[str, tp.Any]) -> bool:
    """Whether ``pixel_values`` rows are grid-enumerated patches rather than images.

    Families like qwen2_vl flatten every image into patch rows and describe the layout in
    ``image_grid_thw``; the row count must then agree with the grid exactly, which rules
    out row-level padding.
    """
    return any(key in batch and batch[key] is not None for key in ("image_grid_thw", "image_grid_hws"))


def vision_row_presence(batch: tp.Mapping[str, tp.Any], presence_key: str | None = None) -> np.ndarray | None:
    """Return a host-side boolean mask of which pixel rows carry real image content.

    Uses ``presence_key`` when the data pipeline provides one; otherwise derives presence
    from ``pixel_values`` (a zeroed row carries no image). Deriving it costs one small
    device-to-host transfer of a ``[rows]`` bool, which is the price of knowing a *static*
    packed count -- jit cannot slice by a traced one.

    Returns ``None`` when the batch has no usable ``pixel_values``.
    """
    if presence_key is not None and presence_key in batch:
        mask = np.asarray(jax.device_get(batch[presence_key]))
        return mask.reshape(mask.shape[0], -1).any(axis=-1) if mask.ndim > 1 else mask.astype(bool)

    pixels = batch.get(PIXEL_KEY)
    if pixels is None or getattr(pixels, "ndim", 0) < 2:
        return None

    reduce_axes = tuple(range(1, pixels.ndim))
    return np.asarray(jax.device_get(jnp.any(pixels != 0, axis=reduce_axes))).astype(bool)


def bucket_row_count(count: int, multiple: int, ceiling: int) -> int:
    """Round ``count`` up to a multiple of ``multiple``, never past ``ceiling``.

    Bucketing keeps the encoder's compiled shape set small. The ceiling is the original
    batch row count -- padding past it would do more work than not packing at all.
    """
    if count <= 0:
        return 0
    if multiple <= 1:
        return min(count, ceiling)
    rounded = ((count + multiple - 1) // multiple) * multiple
    return min(rounded, ceiling)


@dataclass
class VisionSelection:
    """The outcome of choosing which rows to hand the tower.

    Attributes:
        indices: Original row indices of the selected vision-bearing rows, ascending, so
            the placeholder-order contract of the merge holds.
        padded_rows: Row count actually fed to the tower after bucketing.
        total_rows: Row count in the incoming batch.
        layout: Which pixel layout the selection was made under.
    """

    indices: np.ndarray
    padded_rows: int
    total_rows: int
    layout: PixelLayout = PixelLayout.IMAGE_MAJOR

    @property
    def selected_rows(self) -> int:
        """Number of genuine vision rows."""
        return int(self.indices.size)

    @property
    def saves_work(self) -> bool:
        """Whether packing actually shrinks the tower's input."""
        return self.padded_rows < self.total_rows


def select_vision_rows(
    batch: tp.Mapping[str, tp.Any],
    config: ProcessEncoderConfig,
    layout: PixelLayout | None = None,
) -> VisionSelection | None:
    """Choose which pixel rows to encode and the bucketed count to encode them at.

    On ``IMAGE_MAJOR`` batches -- every collator today -- all rows are real images, so no
    presence probe runs (and no device-to-host sync is paid); only bucketing applies. On
    ``ROW_MAJOR`` batches the text-only rows are dropped.

    Returns ``None`` when there is nothing to do: no ``pixel_values``, no row with image
    content, or a count above ``max_rows_per_step`` (in which case the caller keeps the
    in-model path so no image is dropped).
    """
    pixels = batch.get(PIXEL_KEY)
    if pixels is None or getattr(pixels, "ndim", 0) < 2:
        return None

    if layout is None:
        layout = resolve_pixel_layout(batch, config.presence_key)

    total_rows = int(pixels.shape[0])
    if layout is PixelLayout.IMAGE_MAJOR:
        indices = np.arange(total_rows)
    else:
        presence = vision_row_presence(batch, config.presence_key)
        if presence is None:
            return None
        indices = np.flatnonzero(presence)
        if indices.size == 0:
            return None

    if config.max_rows_per_step is not None and indices.size > config.max_rows_per_step:
        logger.warning(
            "process_encoder: %d vision rows exceeds max_rows_per_step=%d; "
            "keeping the in-model vision path for this step.",
            indices.size,
            config.max_rows_per_step,
        )
        return None

    if grid_describes_rows(batch):
        # Grid families (qwen2_vl and relatives) address `pixel_values` rows as *patches*
        # enumerated by `image_grid_thw`. Appending rows without extending the grid would
        # describe an image that does not exist, so bucketing is off until padding is done
        # in whole-image units (patches + a matching grid row).
        padded = int(indices.size)
    else:
        # ROW_MAJOR caps at the batch: padding past it would encode more rows than simply
        # not packing. IMAGE_MAJOR has no such cap -- rounding the image count up to the
        # next rung is what collapses the step's signature set to the ladder.
        ceiling = total_rows if layout is PixelLayout.ROW_MAJOR else _NO_ROW_CEILING
        padded = bucket_row_count(int(indices.size), config.row_bucket_multiple, ceiling)
    return VisionSelection(indices=indices, padded_rows=padded, total_rows=total_rows, layout=layout)


def pack_vision_inputs(
    batch: tp.Mapping[str, tp.Any],
    selection: VisionSelection,
) -> dict[str, tp.Any]:
    """Gather the vision-bearing rows of every image-describing key, bucket-padded.

    Rows are taken in ascending original order and the tail is padded by repeating the
    last selected row. Repetition rather than zeros keeps any per-row grid metadata
    self-consistent (a zero grid would describe an image of no patches); the features
    those padding rows produce are never gathered by the merge, so their content is
    irrelevant -- only their shape matters.
    """
    take = selection.indices
    if selection.padded_rows > take.size:
        pad_width = selection.padded_rows - take.size
        take = np.concatenate([take, np.repeat(take[-1:], pad_width)])

    packed: dict[str, tp.Any] = {}
    for key in (PIXEL_KEY, *VISION_COMPANION_KEYS):
        value = batch.get(key)
        if value is None or not hasattr(value, "shape") or value.ndim == 0:
            continue
        if value.shape[0] != selection.total_rows:
            # Already flat/per-image rather than per-row; the family will interpret it.
            packed[key] = value
            continue
        packed[key] = jnp.take(value, jnp.asarray(take), axis=0)
    return packed


def pad_features_to_constant(features: jnp.ndarray, target_rows: int) -> jnp.ndarray:
    """Zero-pad flattened features up to ``target_rows`` so the step's shape is constant.

    Safe because the merge gathers by ``cumsum`` over placeholder positions and therefore
    never indexes past the final placeholder: trailing rows are inert. This is what lets
    the tower run on a small bucketed shape while the jitted train step keeps a single
    input signature.
    """
    current = features.shape[0]
    if current >= target_rows:
        return features
    pad = jnp.zeros((target_rows - current, *features.shape[1:]), dtype=features.dtype)
    return jnp.concatenate([features, pad], axis=0)


def flatten_features(features: tp.Any) -> jnp.ndarray:
    """Normalize a family's feature output to a flat ``[num_tokens, hidden]`` array.

    Families return either a single array (``[rows, tokens, hidden]`` or already flat) or
    a per-image list/tuple, matching what their own merge sites concatenate.
    """
    if isinstance(features, list | tuple):
        parts = [jnp.asarray(part).reshape(-1, jnp.asarray(part).shape[-1]) for part in features]
        return jnp.concatenate(parts, axis=0)
    array = jnp.asarray(features)
    return array.reshape(-1, array.shape[-1])


class EncoderProcessor:
    """Runs a model's vision tower out of band and injects the result into batches.

    Holds the resolved :class:`EncoderBinding` and the per-shape compiled encode
    functions. One instance lives on the trainer; :meth:`process` is called once per
    microbatch, immediately before the jitted train step.
    """

    def __init__(self, model: tp.Any, config: ProcessEncoderConfig, binding: EncoderBinding | None = None):
        """Bind a processor to a model.

        Args:
            model: The VLM whose tower should be lifted out of the step.
            config: Behaviour knobs.
            binding: Pre-resolved binding; discovered from ``model`` when omitted.
        """
        self.config = config
        self.binding = binding if binding is not None else resolve_encoder_binding(model)
        self._model = model
        self._warned_unsupported = False
        self._feature_rows_target: int | None = None

    @property
    def supported(self) -> bool:
        """Whether this model can accept precomputed vision features."""
        return self.binding is not None

    def _encode(self, model: tp.Any, packed: dict[str, tp.Any]) -> tp.Any:
        """Run the tower on packed inputs under ``stop_gradient``.

        The call is eager (JAX dispatches asynchronously), matching how trainers already
        run out-of-step model forwards such as KTO's reference log-prob precompute.
        ``stop_gradient`` documents and enforces that nothing here contributes to the
        step's gradient.
        """
        binding = self.binding
        assert binding is not None
        feature_fn = getattr(model, binding.feature_fn_name)
        kwargs = {k: v for k, v in packed.items() if k in binding.accepted_kwargs and k != PIXEL_KEY}
        features = feature_fn(packed[PIXEL_KEY], **kwargs)
        return jax.tree_util.tree_map(jax.lax.stop_gradient, features)

    def process(
        self,
        batch: tp.Any,
        model: tp.Any | None = None,
        bucket_index: int | None = None,
    ) -> tp.Any:
        """Replace ``pixel_values`` with precomputed features, or return ``batch`` as-is.

        A no-op -- returning the input object unchanged -- when disabled, when the step's
        bucket is excluded, when the model cannot take precomputed features, or when the
        batch carries no vision. Never raises on an unsupported model; that is
        :func:`validate_process_encoder`'s job at construction time.

        Args:
            batch: The collated microbatch.
            model: Model to encode with; defaults to the one bound at construction.
            bucket_index: Training bucket this step is routed to, for ``only_buckets``.

        Returns:
            The batch, with ``pixel_values`` swapped for the family's feature keyword
            when processing ran.
        """
        config = self.config
        if not config.enabled or not isinstance(batch, dict):
            return batch
        if not config.applies_to_bucket(bucket_index):
            return batch
        if not self.supported:
            if config.require_support is False and not self._warned_unsupported:
                logger.warning(
                    "process_encoder is enabled but this model exposes no precomputed-feature "
                    "keyword; falling back to the in-model vision path."
                )
                self._warned_unsupported = True
            return batch

        layout = resolve_pixel_layout(batch, config.presence_key)
        selection = select_vision_rows(batch, config, layout=layout)
        if selection is None:
            return batch

        binding = self.binding
        assert binding is not None
        model = model if model is not None else self._model

        packed = pack_vision_inputs(batch, selection)
        features = flatten_features(self._encode(model, packed))

        if config.feature_dtype is not None:
            features = features.astype(jnp.dtype(config.feature_dtype))

        # Hold the step's input signature constant across steps: the first processed
        # batch fixes the target, and later batches (which encode a different bucketed
        # row count) are zero-padded up to it. Inert trailing rows make this free.
        per_row_tokens = max(1, features.shape[0] // max(1, selection.padded_rows))
        target_rows = selection.padded_rows * per_row_tokens
        if self._feature_rows_target is None or target_rows > self._feature_rows_target:
            self._feature_rows_target = target_rows
        features = pad_features_to_constant(features, self._feature_rows_target)

        processed = dict(batch)
        processed[binding.feature_kwarg] = features
        # Drop the pixels so the family's `if features is None and pixel_values is not
        # None` guard takes the precomputed branch and the tower does not run twice.
        processed.pop(PIXEL_KEY, None)
        return processed


def validate_process_encoder(
    config: ProcessEncoderConfig,
    model: tp.Any,
    trainable_selector: tp.Any = None,
) -> EncoderBinding | None:
    """Validate a ``process_encoder`` setup at trainer construction, returning the binding.

    Raises rather than silently degrading, because both failure modes are quiet in
    training: an unsupported model would keep paying the full vision cost the flag was
    set to avoid, and a tower listed as trainable would stop receiving gradients while
    the loss curve kept looking plausible.

    Args:
        config: The trainer's process-encoder config.
        model: The model to be trained.
        trainable_selector: The state's trainable selector, if any, used to detect a
            vision tower that the optimizer was told to update.

    Returns:
        The resolved binding, or ``None`` when disabled.

    Raises:
        ValueError: If enabled for a model that cannot accept precomputed features
            (and ``require_support``), or if the vision tower appears trainable.
    """
    if not config.enabled:
        return None

    binding = resolve_encoder_binding(model)
    if binding is None:
        message = (
            "process_encoder.enabled=True but this model does not expose an out-of-band "
            "vision path (needs a vision tower, `get_image_features`, and a forward that "
            f"accepts one of {FEATURE_KWARG_CANDIDATES}). Families whose merge site lacks "
            "the precomputed-feature guard -- notably qwen3_5 / qwen3_5_moe (deepstack), "
            "gemma4, glm46v, glm4v_moe -- need that guard added before this can be used. "
            "Set process_encoder.require_support=false to fall back to the in-model path."
        )
        if config.require_support:
            raise ValueError(message)
        logger.warning(message)
        return None

    if isinstance(trainable_selector, str) and trainable_selector not in ("parameters", ""):
        logger.info(
            "process_encoder: trainable selector %r -- the vision tower is a frozen "
            "feature extractor on processed steps and receives no gradient.",
            trainable_selector,
        )
    return binding
