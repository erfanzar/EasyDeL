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
from __future__ import annotations

import json
import os
import typing as tp
from enum import Enum

from eformer.paths import ePath, ePathLike
from eformer.pytree import auto_pytree, field

from ..helpers import get_cache_dir


class DatasetType(str, Enum):
    """Enumeration of supported dataset types."""

    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"
    ARROW = "arrow"
    HF = "huggingface"
    TSV = "tsv"
    TXT = "txt"

    @classmethod
    def from_string(cls, value: str) -> DatasetType | str:
        """Convert string to DatasetType safely."""
        try:
            return cls(value.lower())
        except ValueError:
            return value

    @classmethod
    def infer_from_path(cls, path: str) -> DatasetType | None:
        """Infer dataset type from file extension."""
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
    """Base class for dataset information."""

    path: DatasetType | str | None = None
    data_files: os.PathLike | str = None
    num_rows: int | None = None
    split: str = "train"

    def __post_init__(self):
        if self.path is None:
            inferred_type = DatasetType.infer_from_path(self.data_files)
            if inferred_type:
                self.path = inferred_type
            assert self.path is not None, (
                "we couldn't automatically find path based on data files, "
                "please provide correct path or format for data files"
            )
        if isinstance(self.path, str):
            try:
                self.path = DatasetType.from_string(self.path)
            except ValueError:
                pass

    def get_str_path(self):
        try:
            return self.path.value.lower()
        except Exception:
            return self.path


@auto_pytree
class TextDatasetInform(BaseDatasetInform):
    """Dataset information specific to text datasets."""

    content_field: str = "content"
    additional_fields: list[str] | None = None
    preprocessing_fn: tp.Callable | None = None


@auto_pytree
class VisualDatasetInform(BaseDatasetInform):
    """Dataset information specific to visual datasets."""

    pixel_field: str = "images"
    content_field: str | None = None
    image_size: tuple[int, int] | None = None
    preprocessing_fn: tp.Callable | None = None


@auto_pytree
class DatasetMixture:
    """Configuration for a mixture of datasets."""

    informs: list[VisualDatasetInform | TextDatasetInform]
    cache_dir: str | ePathLike = field(default_factory=get_cache_dir)
    streaming: bool = True
    text_target_field: str = "text"
    image_target_field: str = "image"
    batch_size: int = 1
    shuffle_buffer_size: int | None = 1000
    seed: int | None = 42

    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = ePath(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    @classmethod
    def load_mixture(cls, json_file: str | os.PathLike):
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        mixture = cls(**config_dict)
        mixture.informs = [TextDatasetInform(**inform) for inform in mixture.informs]
        return mixture

    def save_mixture(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails."""

    pass
