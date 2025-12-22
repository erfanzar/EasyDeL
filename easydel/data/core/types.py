# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from enum import Enum

from eformer.paths import ePath, ePathLike
from eformer.pytree import auto_pytree, field

from easydel.utils.helpers import get_cache_dir


class DatasetType(str, Enum):
    """Enumeration of supported dataset file formats.

    This enum defines all the file formats that can be automatically
    detected and loaded by the data management system.
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
        """Convert string to DatasetType enum value.

        Args:
            value: String representation of dataset type.

        Returns:
            DatasetType enum or original string if not found.
        """
        try:
            return cls(value.lower())
        except ValueError:
            return value

    @classmethod
    def infer_from_path(cls, path: str) -> DatasetType | None:
        """Infer dataset type from file path extension.

        Args:
            path: File path to analyze.

        Returns:
            Inferred DatasetType or None if cannot be determined.
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
    """Base configuration class for dataset information.

    This class provides the common configuration options for all dataset types.
    It handles automatic format detection and field normalization.

    Attributes:
        type: Dataset format type (auto-inferred from file extension if None).
        data_files: Path or paths to dataset files.
        num_rows: Optional limit on number of rows to load.
        dataset_split_name: Name of the dataset split (for HuggingFace datasets).
        split: Which split to load (default: "train").
        format_callback: Optional function to transform dataset examples.
        format_fields: Optional mapping for renaming fields.
    """

    type: DatasetType | str | None = None
    data_files: os.PathLike | str | list[os.PathLike | str] | None = None
    num_rows: int | None = None
    dataset_split_name: str | None = None
    split: str = "train"
    format_callback: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]] | None = None
    format_fields: dict[str, str] | None = None

    def __post_init__(self):
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
            assert self.type is not None, (
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
        """Get string representation of dataset type.

        Returns:
            Lowercase string representation of the dataset type.
        """
        try:
            return self.type.value.lower()
        except Exception:
            return self.type


@auto_pytree
class TextDatasetInform(BaseDatasetInform):
    """Configuration for text-only datasets.

    Extends BaseDatasetInform with text-specific configuration options.

    Attributes:
        content_field: Name of the field containing text content (default: "content").
        additional_fields: Optional list of additional fields to preserve.
        preprocessing_fn: Optional function to preprocess text data.
    """

    content_field: str = "content"
    additional_fields: list[str] | None = None
    preprocessing_fn: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]] | None = None


@auto_pytree
class VisualDatasetInform(BaseDatasetInform):
    """Configuration for visual/multimodal datasets.

    Extends BaseDatasetInform with image-specific configuration options.

    Attributes:
        pixel_field: Name of the field containing image data (default: "images").
        content_field: Optional field name for accompanying text content.
        image_size: Optional target image size as (width, height) tuple.
        preprocessing_fn: Optional function to preprocess image data.
    """

    pixel_field: str = "images"
    content_field: str | None = None
    image_size: tuple[int, int] | None = None
    preprocessing_fn: tp.Callable[[dict[str, tp.Any]], dict[str, tp.Any]] | None = None


@auto_pytree
class DatasetMixture:
    """Configuration for mixing multiple datasets with various strategies.

    Supports combining text and visual datasets with configurable sampling,
    shuffling, batching, and token packing strategies.

    Attributes:
        informs: List of dataset configurations to mix.
        cache_dir: Directory for caching datasets.
        streaming: Enable streaming mode (default: True).
        text_target_field: Target field name for text content (default: "text").
        image_target_field: Target field name for images (default: "image").
        batch_size: Batch size for data loading (default: 1).
        shuffle_buffer_size: Buffer size for shuffling (default: None, disabled).
        seed: Random seed for reproducibility (default: 42).

        # Token packing configuration (optional)
        pack_tokens: Enable pre-tokenized sequence packing (default: False).
        tokens_field_name: Field name containing token IDs (default: "tokens").
        pack_seq_length: Target sequence length for packing (default: None).
        pack_eos_token_id: EOS token ID for padding (default: 0).
        pack_shuffle: Shuffle packed sequences (default: True).
        pack_shuffle_buffer_factor: Buffer size multiplier for shuffle (default: 16).
        dask_storage_options: Storage options for dask/remote files (default: None).

        # On-the-fly tokenization and packing (optional)
        pack_on_the_fly: Enable on-the-fly tokenization and packing (default: False).
        tokenize_callback: Function to tokenize examples, returns token IDs (default: None).

        # Block-deterministic mixture configuration (optional)
        block_mixture: Use deterministic block mixing instead of standard interleave (default: True).
        mixture_block_size: Number of examples per block (default: 2048).
        stop_strategy: Strategy when dataset exhausted - "restart" or "first_exhausted" (default: "restart").
        mixture_weights: Per-dataset weights as dict mapping dataset identifier to weight (default: None).

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
        if isinstance(self.cache_dir, str):
            self.cache_dir = ePath(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        """Load dictionary from JSON file.

        Args:
            json_file: Path to JSON file.

        Returns:
            Parsed dictionary from JSON.
        """
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_string(self) -> str:
        """Serialize configuration to JSON string.

        Returns:
            JSON string representation of this configuration.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_mixture(cls, json_file: str | os.PathLike):
        """Load DatasetMixture configuration from JSON file.

        Args:
            json_file: Path to JSON file containing mixture configuration.

        Returns:
            DatasetMixture instance loaded from file.
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
        """Save DatasetMixture configuration to JSON file.

        Args:
            json_file_path: Path where JSON file will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def build(self):
        """Build the dataset using this mixture configuration.

        Convenience method that builds a dataset from the current mixture
        configuration by calling the pipeline's build_dataset function.

        Returns:
            Dataset or IterableDataset configured according to this mixture's
            settings, with all transformations and mixing strategies applied.

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
        from ..execution.pipeline import build_dataset

        return build_dataset(self)


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails.

    This exception is raised for various dataset loading failures including
    file not found, invalid format, parsing errors, or storage access issues.
    """

    pass
