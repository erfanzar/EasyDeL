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

import typing as tp

from eformer.paths import ePath, ePathLike

from .types import DatasetMixture, TextDatasetInform, VisualDatasetInform

if tp.TYPE_CHECKING:
    from datasets import Dataset as DS


class DataManager:
    """Manager for handling datasets and their transformations."""

    @staticmethod
    def create_dataset_from_mixture(mixture: DatasetMixture) -> DS:
        """
        Create a dataset from a mixture configuration.

        Args:
            mixture: DatasetMixture configuration

        Returns:
            A unified dataset object that contains samples from all sources
        """
        try:
            from datasets import concatenate_datasets, load_dataset
        except ImportError as e:
            raise ImportError("The 'datasets' library is required. Install it with 'pip install datasets'.") from e

        all_datasets = []

        for inform in mixture.informs:
            dataset_loaded = load_dataset(
                path=inform.get_str_path(),
                data_files=inform.data_files,
                split=inform.split,
                cache_dir=mixture.cache_dir,
                streaming=mixture.streaming,
            )
            if isinstance(inform, TextDatasetInform):
                ds_processed = DataManager._process_text_dataset(
                    dataset_loaded,
                    inform,
                    mixture.text_target_field,
                )
            elif isinstance(inform, VisualDatasetInform):
                ds_processed = DataManager._process_visual_dataset(
                    dataset_loaded,
                    inform,
                    mixture.text_target_field,
                    mixture.image_target_field,
                )
            else:
                raise ValueError(f"Unsupported dataset inform type: {type(inform)}")

            all_datasets.append(ds_processed)

        if mixture.streaming:
            combined_dataset = DataManager._interleave_datasets(
                all_datasets,
                mixture.seed,
                mixture.shuffle_buffer_size,
            )
        else:
            combined_dataset = concatenate_datasets(all_datasets)
            if mixture.shuffle_buffer_size:
                combined_dataset = combined_dataset.shuffle(seed=mixture.seed)

        if mixture.batch_size > 1:
            combined_dataset = combined_dataset.batch(mixture.batch_size)

        return combined_dataset

    @staticmethod
    def _process_text_dataset(
        dataset_loaded: DS,
        inform: TextDatasetInform,
        target_field: str,
    ) -> DS:
        """Process a text dataset according to the inform configuration."""
        if dataset_loaded.column_names is not None:
            if inform.content_field not in dataset_loaded.column_names:
                raise RuntimeError(
                    f"Couldnt find {inform.content_field} in available columns({dataset_loaded.column_names})."
                )

        def transform_fn(example):
            try:
                result = {target_field: example[inform.content_field]}
            except KeyError as e:
                raise KeyError(f"couldn't access {inform.content_field} Available Keys {example.keys()}.") from e

            if inform.additional_fields:
                for field in inform.additional_fields:
                    if field in example:
                        result[field] = example[field]

            return result

        ds_processed = dataset_loaded.map(transform_fn)

        if inform.preprocessing_fn:
            ds_processed = ds_processed.map(inform.preprocessing_fn)

        return ds_processed

    @staticmethod
    def _process_visual_dataset(
        dataset_loaded,
        inform: VisualDatasetInform,
        text_target_field: str,
        image_target_field: str,
    ):
        """Process a visual dataset according to the inform configuration."""
        try:
            from PIL import Image as PILImage  # type:ignore
        except ImportError as e:
            raise ImportError(
                "The 'pillow' library is required for visual datasets. Install it with 'pip install pillow'."
            ) from e

        def transform_fn(example):
            result = {image_target_field: example[inform.pixel_field]}

            if inform.content_field and inform.content_field in example:
                result[text_target_field] = example[inform.content_field]

            return result

        ds_processed = dataset_loaded.map(transform_fn)

        if inform.image_size:

            def resize_images(example):
                if isinstance(example[image_target_field], PILImage.Image):
                    example[image_target_field] = example[image_target_field].resize(
                        inform.image_size, PILImage.BILINEAR
                    )
                return example

            ds_processed = ds_processed.map(resize_images)

        if inform.preprocessing_fn:
            ds_processed = ds_processed.map(inform.preprocessing_fn)

        return ds_processed

    @staticmethod
    def _interleave_datasets(datasets, seed=None, shuffle_buffer_size=None):
        """Interleave multiple streaming datasets."""
        try:
            from datasets import interleave_datasets
        except ImportError as e:
            raise ImportError("The 'datasets' library is required. Install it with 'pip install datasets'.") from e

        interleaved = interleave_datasets(
            datasets,
            seed=seed,
            stopping_strategy="first_exhausted",
        )

        if shuffle_buffer_size:
            interleaved = interleaved.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

        return interleaved

    @classmethod
    def load_from_config(cls, config_path: str | ePathLike) -> tuple[DatasetMixture, tp.Any]:
        """
        Load dataset configuration from a JSON or YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Tuple of (DatasetMixture, dataset)
        """
        import json

        import yaml

        path = ePath(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in (".yaml", ".yml"):
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        mixture = cls._dict_to_mixture(config_dict)
        dataset = cls.create_dataset_from_mixture(mixture)

        return mixture, dataset

    @staticmethod
    def _dict_to_mixture(config_dict: dict) -> DatasetMixture:
        """Convert a dictionary to a DatasetMixture object."""
        informs = []

        for ds_config in config_dict.get("informs", []):
            if "pixel_field" in ds_config:
                inform = VisualDatasetInform(
                    path=ds_config["path"],
                    data_files=ds_config["data_files"],
                    pixel_field=ds_config["pixel_field"],
                    content_field=ds_config.get("content_field"),
                    split=ds_config.get("split", "train"),
                    image_size=tuple(ds_config["image_size"]) if "image_size" in ds_config else None,
                )
            else:
                inform = TextDatasetInform(
                    path=ds_config["path"],
                    data_files=ds_config["data_files"],
                    content_field=ds_config["content_field"],
                    split=ds_config.get("split", "train"),
                    additional_fields=ds_config.get("additional_fields"),
                )

            informs.append(inform)

        mixture = DatasetMixture(
            informs=informs,
            cache_dir=config_dict.get("cache_dir", "./cache"),
            streaming=config_dict.get("streaming", True),
            text_target_field=config_dict.get("text_target_field", "text"),
            image_target_field=config_dict.get("image_target_field", "image"),
            batch_size=config_dict.get("batch_size", 32),
            shuffle_buffer_size=config_dict.get("shuffle_buffer_size", 1000),
            seed=config_dict.get("seed", 42),
        )

        return mixture
