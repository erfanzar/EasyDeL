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

Provides:
- DatasetType: Enum for supported dataset formats
- BaseDatasetInform: Base configuration for dataset information
- TextDatasetInform: Configuration for text datasets
- VisualDatasetInform: Configuration for visual/multimodal datasets
- DatasetMixture: Configuration for mixing multiple datasets
- DatasetLoadError: Exception for dataset loading failures
"""

from __future__ import annotations

import json
import os
import typing as tp
from enum import Enum

from eformer.paths import ePath, ePathLike
from eformer.pytree import auto_pytree, field

from ..helpers import get_cache_dir


class DatasetType(str, Enum):
    """Enumeration of supported dataset file formats.

    Attributes:
        JSON: JSON/JSONL format
        PARQUET: Apache Parquet format
        CSV: Comma-separated values
        ARROW: Apache Arrow format
        HF: HuggingFace dataset
        TSV: Tab-separated values
        TXT: Plain text files
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
            value: String representation of dataset type

        Returns:
            DatasetType enum or original string if not found
        """
        try:
            return cls(value.lower())
        except ValueError:
            return value

    @classmethod
    def infer_from_path(cls, path: str) -> DatasetType | None:
        """Infer dataset type from file path extension.

        Args:
            path: File path to analyze

        Returns:
            Inferred DatasetType or None if cannot be determined
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

    Stores common dataset metadata and handles automatic type inference.

    Attributes:
        type: Dataset format type (auto-inferred if None)
        data_files: Path to dataset files
        num_rows: Optional row limit for loading
        split: Dataset split name (default: "train")
        format_callback: Optional function to normalize dataset schema for interleaving.
            Should take a dataset example dict and return a normalized dict with
            consistent field names and types across all datasets.
        format_fields: Optional dict mapping to rename fields {'old_name': 'new_name'}.
            Simpler alternative to format_callback for basic field renaming.
    """

    type: DatasetType | str | None = None
    data_files: os.PathLike | str = None
    num_rows: int | None = None
    dataset_split_name: str | None = None
    split: str = "train"
    format_callback: tp.Callable[[dict], dict] | None = None
    format_fields: dict[str, str] | None = None

    def __post_init__(self):
        if self.type is None:
            inferred_type = DatasetType.infer_from_path(self.data_files)
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

    def get_str_type(self):
        """Get string representation of dataset type.

        Returns:
            Lowercase string representation of type
        """
        try:
            return self.type.value.lower()
        except Exception:
            return self.type


@auto_pytree
class TextDatasetInform(BaseDatasetInform):
    """Configuration for text-only datasets.

    Attributes:
        content_field: Field name containing text content (default: "content")
        additional_fields: Optional list of additional fields to include
        preprocessing_fn: Optional preprocessing function to apply
    """

    content_field: str = "content"
    additional_fields: list[str] | None = None
    preprocessing_fn: tp.Callable | None = None


@auto_pytree
class VisualDatasetInform(BaseDatasetInform):
    """Configuration for visual/multimodal datasets.

    Attributes:
        pixel_field: Field name containing image data (default: "images")
        content_field: Optional field name for text content
        image_size: Optional target image size as (width, height)
        preprocessing_fn: Optional preprocessing function to apply
    """

    pixel_field: str = "images"
    content_field: str | None = None
    image_size: tuple[int, int] | None = None
    preprocessing_fn: tp.Callable | None = None


@auto_pytree
class DatasetMixture:
    """Configuration for mixing multiple datasets with various strategies.

    Supports combining text and visual datasets with configurable sampling,
    shuffling, and batching strategies.

    Attributes:
        informs: List of dataset configurations to mix
        cache_dir: Directory for caching datasets
        streaming: Enable streaming mode (default: True)
        text_target_field: Target field name for text content (default: "text")
        image_target_field: Target field name for images (default: "image")
        batch_size: Batch size for data loading (default: 1)
        shuffle_buffer_size: Buffer size for shuffling (default: None, disabled)
        seed: Random seed for reproducibility (default: 42)

    Example:
        >>> from easydel.utils.data_managers import DatasetMixture, TextDatasetInform
        >>> mixture = DatasetMixture(
        ...     informs=[
        ...         TextDatasetInform(type="json", data_files="data1.json"),
        ...         TextDatasetInform(type="parquet", data_files="data2.parquet"),
        ...     ],
        ...     batch_size=32,
        ...     shuffle_buffer_size=10000,
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

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = ePath(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        """Load dictionary from JSON file.

        Args:
            json_file: Path to JSON file

        Returns:
            Parsed dictionary from JSON
        """
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_string(self) -> str:
        """Serialize configuration to JSON string.

        Returns:
            JSON string representation of this configuration
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_mixture(cls, json_file: str | os.PathLike):
        """Load DatasetMixture configuration from JSON file.

        Args:
            json_file: Path to JSON file containing mixture configuration

        Returns:
            DatasetMixture instance loaded from file
        """
        config_dict = cls._dict_from_json_file(json_file)
        mixture = cls(**config_dict)
        mixture.informs = [TextDatasetInform(**inform) for inform in mixture.informs]
        return mixture

    def save_mixture(self, json_file_path: str | os.PathLike):
        """Save DatasetMixture configuration to JSON file.

        Args:
            json_file_path: Path where JSON file will be saved
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails.

    Used to signal errors during dataset loading operations such as:
    - File not found
    - Invalid format
    - Parsing errors
    - Remote storage access failures
    """

    pass
