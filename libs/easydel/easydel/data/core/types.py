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

"""Type definitions and configuration classes for data management.

This module defines the core data structures and configurations used throughout
the data management system, including dataset types, mixing configurations,
and dataset information containers.
"""

from __future__ import annotations

import json
import os
import typing as tp
from enum import StrEnum

from eformer.paths import ePath, ePathLike
from eformer.pytree import auto_pytree, field

from easydel.utils.helpers import get_cache_dir


class DatasetType(StrEnum):
    """Enumeration of dataset container formats supported by the data layer.

    Acts as the canonical tag carried on :class:`BaseDatasetInform` and
    its subclasses to select the correct reader in
    :mod:`easydel.data.sources`. Members inherit from :class:`StrEnum`
    so they compare equal to their wire-format strings (``"json"``,
    ``"parquet"``, …) in serialised configs.

    Members include the standard file containers (``JSON``, ``PARQUET``,
    ``CSV``, ``ARROW``, ``TSV``, ``TXT``) plus the HuggingFace virtual
    type (``HF``) used for ``datasets.load_dataset``-style sources.
    """

    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"
    ARROW = "arrow"
    HF = "huggingface"
    TSV = "tsv"
    TXT = "txt"

    @classmethod
    def from_string(cls, value: str) -> DatasetType | str:
        """Best-effort coercion of a free-form string into a :class:`DatasetType`.

        Performs a case-insensitive lookup. When the string does not
        match any member, the input is returned unchanged so callers can
        round-trip user-supplied values they want to keep verbatim
        (e.g. plugin formats added by downstream code).

        Args:
            value: Candidate format identifier such as ``"json"`` or
                ``"PARQUET"``.

        Returns:
            DatasetType | str: A :class:`DatasetType` member when
            ``value`` matches one (case-insensitive); otherwise the
            original string is returned.
        """
        try:
            return cls(value.lower())
        except ValueError:
            return value

    @classmethod
    def infer_from_path(cls, path: str) -> DatasetType | None:
        """Guess the dataset format from the suffix of a filesystem path.

        Recognises common compressed variants (``.json.gz``,
        ``.json.zst``, ``.tsv.gz``, …) so the extension table covers
        everything the readers in :mod:`easydel.data.sources` accept.

        Args:
            path: Filesystem path or URI whose suffix is examined. Only
                the trailing extension is consulted.

        Returns:
            DatasetType | None: The inferred type, or ``None`` when the
            extension does not match any known reader.
        """
        mapping = {
            (".json", ".jsonl", ".json.gz", ".jsonl.gz", ".json.zst", ".jsonl.zst"): cls.JSON,
            (".parquet",): cls.PARQUET,
            (".csv",): cls.CSV,
            (".arrow",): cls.ARROW,
            (".tsv", ".tsv.gz"): cls.TSV,
            (".txt",): cls.TXT,
        }
        for exts, dtype in mapping.items():
            if any(path.endswith(ext) for ext in exts):
                return dtype
        return None


@auto_pytree
class BaseDatasetInform:
    """Legacy per-dataset declaration for the :class:`DatasetMixture` API.

    The legacy ``*Inform`` family describes one constituent dataset of a
    :class:`DatasetMixture`, capturing its location, format, split, and
    optional row-level transformations. ``__post_init__`` runs auto-type
    inference from the data file extensions and emits a deprecation
    warning when ``dataset_split_name`` is set on file-based formats
    (it is only meaningful for HuggingFace datasets). Subclasses
    :class:`TextDatasetInform` and :class:`VisualDatasetInform` add
    text/multimodal-specific fields.

    The newer typed pipeline API (:class:`DatasetConfig`) is the
    preferred entry point; this class remains for backwards
    compatibility with existing :class:`DatasetMixture` users.

    Attributes:
        type (DatasetType | str | None): Container format tag. When
            ``None`` the format is inferred from the file extension in
            ``__post_init__`` and the constructor raises if inference
            fails. Stored as :class:`DatasetType` after normalisation.
        data_files (os.PathLike | str | list[...] | None): Path,
            list of paths, or glob/URI to the dataset files. ``None`` is
            allowed during construction but most downstream code
            requires this to be set.
        num_rows (int | None): Optional cap on the number of rows
            actually loaded; useful for smoke tests and debugging.
        dataset_split_name (str | None): Split name for HuggingFace
            datasets (the ``split=`` argument to ``load_dataset``).
            Ignored — and warned about — for file-based types.
        split (str): Generic split label used by file-based readers
            (defaults to ``"train"``).
        format_callback (Callable[[dict[str, Any]], dict[str, Any]] | None):
            Per-row transform applied immediately after reading. Useful
            for one-off schema massaging without writing a full
            transform.
        format_fields (dict[str, str] | None): Old-key -> new-key
            rename map applied to each row.
    """

    type: DatasetType | str | None = None
    data_files: os.PathLike | str | list[os.PathLike | str] | None = None
    num_rows: int | None = None
    dataset_split_name: str | None = None
    split: str = "train"
    format_callback: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]] | None = None
    format_fields: dict[str, str] | None = None

    def __post_init__(self):
        """Normalise ``type``, infer it from file extensions, and validate splits.

        Performs three operations after dataclass initialisation:

        1. If :attr:`type` is ``None``, the first entry of
           :attr:`data_files` is examined and
           :meth:`DatasetType.infer_from_path` is consulted to pick a
           reader. Failing that, ``ValueError`` is raised.
        2. String inputs to :attr:`type` are coerced to a
           :class:`DatasetType` member (silently leaving unknown strings
           in place, matching :meth:`DatasetType.from_string`).
        3. When :attr:`dataset_split_name` is set on a file-based
           container (i.e. anything that is not ``"huggingface"`` /
           ``"hf"``), a :class:`DeprecationWarning` is emitted because
           the field has no effect for those readers.

        Raises:
            ValueError: If :attr:`type` is ``None`` and cannot be
                inferred from :attr:`data_files`.
        """
        if self.type is None:
            # Convert PathLike to string for type inference
            inferred_type = None
            if self.data_files:
                first_file: os.PathLike | str | None
                if isinstance(self.data_files, list):
                    first_file = self.data_files[0] if self.data_files else None
                else:
                    first_file = self.data_files

                if first_file is not None:
                    inferred_type = DatasetType.infer_from_path(os.fspath(first_file))
            if inferred_type:
                self.type = inferred_type
            if self.type is None:
                raise ValueError(
                    "we couldn't automatically find type based on data files, "
                    "please provide correct type or format for data files"
                )
        if isinstance(self.type, str):
            try:
                self.type = DatasetType.from_string(self.type)
            except ValueError:
                pass

        if self.dataset_split_name and str(self.get_str_type()) not in {"huggingface", "hf"}:
            import warnings

            warnings.warn(
                "dataset_split_name is ignored for file-based dataset types; it will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

    def get_str_type(self):
        """Return the dataset type as a lowercase string for routing decisions.

        Mostly used by readers that branch on the format tag —
        :class:`DatasetType` enum members support string comparison via
        :class:`StrEnum`, but downstream code occasionally wants the
        bare value. Falls back to returning :attr:`type` unchanged when
        the field is not a :class:`DatasetType` (e.g. a raw plugin
        string).

        Returns:
            str: Lowercase dataset format identifier (``"json"``,
            ``"parquet"``, …) or the raw value of :attr:`type` when
            unrecognised.
        """
        try:
            return self.type.value.lower()
        except Exception:
            return self.type


@auto_pytree
class TextDatasetInform(BaseDatasetInform):
    """Per-dataset declaration for text-only constituents of a :class:`DatasetMixture`.

    Adds the fields the text-mixing path needs on top of
    :class:`BaseDatasetInform`: which row key holds the actual text
    content, which extra columns to preserve through the pipeline, and
    an optional pure-Python preprocessing hook.

    Attributes:
        content_field (str | None): Source row key holding the raw text;
            renamed to :attr:`DatasetMixture.text_target_field` during
            mixing. ``None`` indicates the dataset has no text body
            (rare; useful for label-only sources).
        additional_fields (list[str] | None): Extra row keys to keep
            (e.g. ``"label"``, ``"meta"``); everything else is dropped
            after mixing to keep batches small.
        preprocessing_fn (Callable[[dict[str, Any]], dict[str, Any]] | None):
            Optional per-row transform applied before mixing. Receives
            and returns the row dict.
    """

    content_field: str | None = "content"
    additional_fields: list[str] | None = None
    preprocessing_fn: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]] | None = None


@auto_pytree
class VisualDatasetInform(BaseDatasetInform):
    """Per-dataset declaration for image / multimodal constituents of a mixture.

    Specialises :class:`BaseDatasetInform` with multimodal fields.
    Carries the row key holding image data, an optional companion text
    field, and a target resize spec used by the image-handling code in
    the mixer.

    Attributes:
        pixel_field (str): Row key under which raw image bytes/arrays
            live in the source schema; renamed to
            :attr:`DatasetMixture.image_target_field` during mixing.
        content_field (str | None): Optional row key for accompanying
            text (captions, instructions). ``None`` indicates a
            text-free image dataset.
        image_size (tuple[int, int] | None): Target ``(width, height)``
            to which loaded images are resized. ``None`` keeps the
            native resolution.
        preprocessing_fn (Callable[[dict[str, Any]], dict[str, Any]] | None):
            Optional per-row transform applied before mixing.
    """

    pixel_field: str = "images"
    content_field: str | None = None
    image_size: tuple[int, int] | None = None
    preprocessing_fn: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]] | None = None


@auto_pytree
class DatasetMixture:
    """Top-level legacy declaration of a multi-dataset training mix.

    A :class:`DatasetMixture` aggregates one or more
    :class:`BaseDatasetInform` constituents together with all the
    settings needed to load, mix, optionally tokenize, optionally pack,
    and batch them for training. It is the original (pre-Pipeline) API
    surface for the data layer; see :class:`PipelineConfig` for the
    typed-stage replacement.

    The dataclass is registered as a pytree via ``@auto_pytree`` so it
    can be passed across JAX transforms / serialised consistently with
    other EasyDeL state. ``__post_init__`` ensures :attr:`cache_dir`
    exists on disk so caching can be turned on without manual setup.

    Attributes:
        informs (list[VisualDatasetInform | TextDatasetInform]):
            Constituent dataset declarations that make up the mix.
        cache_dir (str | ePathLike): Directory backing on-disk caches
            for dataset fragments and tokenization. Defaults to the
            EasyDeL user cache directory and is auto-created.
        streaming (bool): Iterate the inputs in streaming mode rather
            than fully materialising them in RAM.
        text_target_field (str): Canonical row key the mix re-projects
            text content onto regardless of source schema.
        image_target_field (str): Canonical row key the mix re-projects
            image content onto regardless of source schema.
        batch_size (int): Examples per yielded batch. ``1`` is the
            default to match consumer code that batches downstream.
        shuffle_buffer_size (int | None): Reservoir size for streaming
            shuffle. ``None`` disables the shuffle entirely.
        seed (int | None): RNG seed governing shuffling and mixing.
            Defaults to ``42`` for reproducibility.
        pack_tokens (bool): Enable pre-tokenized sequence packing — if
            ``True``, the rows are assumed to carry token id arrays
            under :attr:`tokens_field_name` and are packed into windows
            of length :attr:`pack_seq_length`.
        tokens_field_name (str): Row key consulted by the packer when
            :attr:`pack_tokens` is on.
        pack_seq_length (int | None): Target window length for packing.
            ``None`` indicates packing has not been configured.
        pack_eos_token_id (int): Token id used as the boundary between
            packed examples and as padding.
        pack_shuffle (bool): Shuffle the post-packed stream. Useful
            because consecutive packed rows tend to share underlying
            examples.
        pack_shuffle_buffer_factor (int): Multiplier on the natural
            packing batch governing the post-pack shuffle reservoir
            size.
        dask_storage_options (dict[str, Any] | None): Forwarded
            verbatim to ``fsspec``/``dask`` readers as
            ``storage_options``; carries credentials, request timeouts,
            etc.
        pack_on_the_fly (bool): When ``True``, raw rows are tokenized
            with :attr:`tokenize_callback` and packed in a single
            streaming pass — alternative to pre-tokenizing the dataset
            on disk.
        tokenize_callback (Callable[[dict[str, Any]], list[int]] | None):
            User-supplied tokenizer hook used by ``pack_on_the_fly``;
            takes a row dict and returns token ids.
        prefetch_workers (int): Background prefetch threads driving the
            output iterator.
        prefetch_buffer_size (int): Number of fully-formed batches kept
            in the prefetch queue.
        cloud_max_retries (int): Maximum retry attempts for transient
            cloud-IO failures before propagating.
        cloud_retry_delay (float): Initial backoff delay between retries
            in seconds; subsequent retries use exponential backoff.
        cache_remote_files (bool): When ``True``, files fetched from
            cloud storage are cached to ``cache_dir`` for reuse.
        cache_expiry_seconds (int): TTL for cached remote files.
        block_mixture (bool): Use the deterministic block mixer
            (consumes :attr:`mixture_block_size` examples from one
            dataset before switching) instead of round-robin
            interleave. Improves throughput.
        mixture_block_size (int): Number of examples per contiguous
            block when :attr:`block_mixture` is on.
        stop_strategy (str): What the mixer does when a dataset is
            exhausted — ``"restart"`` (recycle), ``"first_exhausted"``
            (terminate immediately), or ``"all_exhausted"``.
        mixture_weights (dict[str, float] | None): Per-dataset weights
            keyed by dataset identifier (the constituent's
            ``data_files`` path or explicit name). ``None`` falls back
            to uniform mixing.

    Example:
        >>> from easydel.data import DatasetMixture, TextDatasetInform
        >>>
        >>> # Simple mixture
        >>> mixture = DatasetMixture(
        ...     informs=[
        ...         TextDatasetInform(type="json", data_files="data1.json"),
        ...         TextDatasetInform(type="parquet", data_files="data2.parquet"),
        ...     ],
        ...     batch_size=32,
        ...     shuffle_buffer_size=10000,
        ... )
        >>>
        >>> # Advanced mixture with block mixing and token packing
        >>> mixture = DatasetMixture(
        ...     informs=[...],
        ...     block_mixture=True,
        ...     mixture_weights={"dataset1": 0.7, "dataset2": 0.3},
        ...     pack_tokens=True,
        ...     pack_seq_length=2048,
        ...     pack_eos_token_id=0,
        ... )
    """

    informs: list[VisualDatasetInform | TextDatasetInform]
    cache_dir: str | ePathLike = field(default_factory=get_cache_dir)
    streaming: bool = True
    text_target_field: str = "text"
    image_target_field: str = "image"
    batch_size: int = 1
    shuffle_buffer_size: int | None = None
    seed: int | None = 42

    pack_tokens: bool = False
    tokens_field_name: str = "tokens"
    pack_seq_length: int | None = None
    pack_eos_token_id: int = 0
    pack_shuffle: bool = True
    pack_shuffle_buffer_factor: int = 16
    dask_storage_options: dict[str, tp.Any] | None = None

    pack_on_the_fly: bool = False
    tokenize_callback: tp.Callable[[dict[str, tp.Any]], list[int]] | None = None

    # Prefetch configuration
    prefetch_workers: int = 2
    prefetch_buffer_size: int = 4

    # Cloud storage options
    cloud_max_retries: int = 3
    cloud_retry_delay: float = 0.1
    cache_remote_files: bool = True
    cache_expiry_seconds: int = 86400

    block_mixture: bool = True
    mixture_block_size: int = 2048
    stop_strategy: str = "restart"

    mixture_weights: dict[str, float] | None = None

    def __post_init__(self):
        """Normalise :attr:`cache_dir` to an :class:`ePath` and ensure it exists.

        Allows the user to supply ``cache_dir`` as a plain string for
        ergonomics — the dataclass coerces it into an :class:`ePath` so
        downstream code can rely on the rich path API regardless of how
        it was constructed. The directory is created with ``mkdir
        -p`` semantics so caching is safe to enable on a fresh machine.
        """
        if isinstance(self.cache_dir, str):
            self.cache_dir = ePath(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        """Read a JSON file and decode its content into a Python dict.

        Internal helper for :meth:`load_mixture`; assumes UTF-8
        encoding and a top-level JSON object.

        Args:
            json_file: Path to the JSON file to read.

        Returns:
            dict: The parsed top-level JSON object.
        """
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_string(self) -> str:
        """Serialise the mixture into a stable, deterministic JSON string.

        Uses ``sort_keys=True`` so byte-equal mixtures produce
        byte-equal JSON, which makes the output diffable and safe to
        use as a cache key. A trailing newline is appended so the
        result plays nicely with file editors.

        Returns:
            str: JSON-encoded representation of this mixture, including
            a trailing newline.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_mixture(cls, json_file: str | os.PathLike):
        """Reconstruct a :class:`DatasetMixture` from a previously saved JSON file.

        Decodes the file with :meth:`_dict_from_json_file`, instantiates
        the mixture with the top-level fields, then walks
        :attr:`informs` and re-hydrates each entry into either
        :class:`VisualDatasetInform` (if the dict carries a
        ``"pixel_field"``) or :class:`TextDatasetInform`. The
        reconstructed informs replace the raw dicts on the returned
        mixture.

        Args:
            json_file: Path to the JSON file written by
                :meth:`save_mixture`.

        Returns:
            DatasetMixture: A fully reconstructed mixture, ready to be
            used with :meth:`build`.
        """

        config_dict = cls._dict_from_json_file(json_file)
        mixture = cls(**config_dict)
        informs = []
        for inform in mixture.informs:
            if isinstance(inform, dict) and "pixel_field" in inform:
                informs.append(VisualDatasetInform(**inform))
            else:
                informs.append(TextDatasetInform(**inform))
        mixture.informs = informs
        return mixture

    def save_mixture(self, json_file_path: str | os.PathLike):
        """Persist the mixture as JSON for later round-tripping via :meth:`load_mixture`.

        The output is the same string returned by :meth:`to_json_string`
        — sorted keys, two-space indent, trailing newline.

        Args:
            json_file_path: Destination path. The file is written with
                UTF-8 encoding and overwrites any existing content.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def build(self):
        """Materialise this declaration into an iterable dataset object.

        Thin wrapper that defers to
        :func:`easydel.data.execution.pipeline.build_dataset`, kept on
        the class so users can write ``mixture.build()`` instead of
        importing the runner. The import is deferred to avoid a
        circular module dependency at import time.

        Returns:
            Dataset or IterableDataset fully configured according to
            the mixture (mixing, shuffling, batching, optional
            tokenization and packing all applied).

        Example:
            >>> mixture = DatasetMixture(
            ...     informs=[TextDatasetInform(type="json", data_files="data.json")],
            ...     batch_size=32,
            ...     shuffle_buffer_size=10000
            ... )
            >>> dataset = mixture.build()
            >>> for batch in dataset:
            ...     process(batch)
        """
        pipeline_mod = __import__("easydel.data.execution.pipeline", fromlist=["build_dataset"])
        return pipeline_mod.build_dataset(self)


class DatasetLoadError(Exception):
    """Domain-specific exception for unrecoverable dataset-loading errors.

    Raised by readers in :mod:`easydel.data.sources` and execution
    helpers in :mod:`easydel.data.execution` when a dataset cannot be
    materialised — e.g. the file is missing, the format does not match
    its declared :class:`DatasetType`, the underlying parser fails, or
    cloud storage access errors out after retries. Catch this in
    application code that needs to distinguish dataset issues from
    other runtime failures.
    """

    pass
