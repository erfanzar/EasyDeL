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
from concurrent.futures import ThreadPoolExecutor, as_completed

from eformer.paths import ePath

from .fast_loader import DataStreamOptimizer, FastDataLoader
from .types import DatasetMixture, TextDatasetInform, VisualDatasetInform

if tp.TYPE_CHECKING:
    from datasets import Dataset as DS


class FastDataManager:
    """Optimized data manager with fsspec support and parallel processing."""

    def __init__(
        self,
        cache_dir: str | None = None,
        num_workers: int = 4,
        prefetch_size: int = 10,
        buffer_size: int = 100,
        use_async: bool = True,
    ):
        self.cache_dir = cache_dir or str(ePath.home() / ".cache" / "easydel")
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.buffer_size = buffer_size
        self.use_async = use_async

        self.loader = FastDataLoader(
            cache_storage=self.cache_dir,
            use_async=use_async,
            num_workers=num_workers,
            buffer_size=buffer_size,
        )
        self.stream_optimizer = DataStreamOptimizer(
            prefetch_size=prefetch_size,
            buffer_size=buffer_size,
            num_workers=num_workers,
        )

    @staticmethod
    def create_dataset_from_mixture(
        mixture: DatasetMixture,
        fast_mode: bool = True,
        parallel_load: bool = True,
    ) -> DS:
        """
        Create a dataset from a mixture configuration with optimizations.

        Args:
            mixture: DatasetMixture configuration
            fast_mode: Use fast loading optimizations
            parallel_load: Load datasets in parallel

        Returns:
            A unified dataset object
        """
        if fast_mode:
            manager = FastDataManager(
                cache_dir=str(mixture.cache_dir),
                num_workers=4,
                prefetch_size=10,
                buffer_size=100,
            )
            return manager._create_fast_dataset(mixture, parallel_load)
        else:
            from .manager import DataManager

            return DataManager.create_dataset_from_mixture(mixture)

    def _create_fast_dataset(self, mixture: DatasetMixture, parallel_load: bool = True) -> DS:
        """Internal fast dataset creation."""
        try:
            from datasets import concatenate_datasets
        except ImportError as e:
            raise ImportError("The 'datasets' library is required. Install it with 'pip install datasets'.") from e

        if parallel_load and len(mixture.informs) > 1:
            all_datasets = self._parallel_load_datasets(mixture)
        else:
            all_datasets = self._sequential_load_datasets(mixture)

        if mixture.streaming:
            combined_dataset = self._optimized_interleave(
                all_datasets,
                mixture.seed,
                mixture.shuffle_buffer_size,
            )
        else:
            combined_dataset = concatenate_datasets(all_datasets)
            if mixture.shuffle_buffer_size:
                combined_dataset = combined_dataset.shuffle(seed=mixture.seed)

        if mixture.batch_size > 1:
            combined_dataset = self._optimized_batch(combined_dataset, mixture.batch_size)

        return combined_dataset

    def _parallel_load_datasets(self, mixture: DatasetMixture) -> list:
        """Load multiple datasets in parallel."""

        all_datasets = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_inform = {}

            for inform in mixture.informs:
                future = executor.submit(
                    self._load_single_dataset,
                    inform,
                    mixture,
                )
                future_to_inform[future] = inform

            for future in as_completed(future_to_inform):
                try:
                    dataset = future.result()
                    all_datasets.append(dataset)
                except Exception as e:
                    inform = future_to_inform[future]
                    print(f"Error loading dataset {inform.data_files}: {e}")
                    raise

        return all_datasets

    def _sequential_load_datasets(self, mixture: DatasetMixture) -> list:
        """Load datasets sequentially (fallback)."""
        all_datasets = []

        for inform in mixture.informs:
            dataset = self._load_single_dataset(inform, mixture)
            all_datasets.append(dataset)

        return all_datasets

    def _load_single_dataset(self, inform: TextDatasetInform | VisualDatasetInform, mixture: DatasetMixture):
        """Load and process a single dataset."""
        from datasets import load_dataset

        dataset_type = inform.get_str_type()

        if dataset_type in ["json", "jsonl", "parquet", "csv", "arrow"]:
            dataset_loaded = self._fast_load_dataset(inform, mixture)
        else:
            # For HuggingFace datasets, data_files should be the dataset ID
            # and we don't pass data_files parameter to load_dataset
            if dataset_type == "huggingface" or dataset_type == "hf":
                dataset_loaded = load_dataset(
                    path=inform.data_files if isinstance(inform.data_files, str) else inform.data_files[0],
                    split=inform.split,
                    cache_dir=mixture.cache_dir,
                    streaming=mixture.streaming,
                    num_proc=self.num_workers if not mixture.streaming else None,
                )
            else:
                # For other types, assume it's a HuggingFace dataset ID or local path
                dataset_loaded = load_dataset(
                    path=dataset_type,
                    data_files=inform.data_files,
                    split=inform.split,
                    cache_dir=mixture.cache_dir,
                    streaming=mixture.streaming,
                    num_proc=self.num_workers if not mixture.streaming else None,
                )

        if isinstance(inform, TextDatasetInform):
            return self._process_text_dataset_fast(
                dataset_loaded,
                inform,
                mixture.text_target_field,
            )
        elif isinstance(inform, VisualDatasetInform):
            return self._process_visual_dataset_fast(
                dataset_loaded,
                inform,
                mixture.text_target_field,
                mixture.image_target_field,
            )
        else:
            raise ValueError(f"Unsupported dataset inform type: {type(inform)}")

    def _fast_load_dataset(self, inform: TextDatasetInform | VisualDatasetInform, mixture: DatasetMixture):
        """Use fast loader for supported file types."""
        from datasets import Dataset

        file_type = inform.get_str_type()
        data_files = inform.data_files

        if isinstance(data_files, str):
            data_files = [data_files]
        elif not isinstance(data_files, list):
            data_files = self.loader.glob_files(str(data_files))

        if file_type == "json" or file_type == "jsonl":
            data = []
            for file in data_files:
                file_data = self.loader.load_json(file, lines=(file_type == "jsonl"))
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
            return Dataset.from_list(data)

        elif file_type == "parquet":
            import pandas as pd

            dfs = []
            for file in data_files:
                df = self.loader.load_parquet(file)
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            return Dataset.from_pandas(combined_df)

        elif file_type == "csv":
            import pandas as pd

            dfs = []
            for file in data_files:
                df = self.loader.load_csv(file)
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            return Dataset.from_pandas(combined_df)

        elif file_type == "arrow":
            import pandas as pd

            dfs = []
            for file in data_files:
                df = self.loader.load_arrow(file)
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            return Dataset.from_pandas(combined_df)

        else:
            from datasets import load_dataset

            return load_dataset(
                path=file_type,
                data_files=data_files,
                split=inform.split,
                cache_dir=mixture.cache_dir,
                streaming=mixture.streaming,
            )

    def _process_text_dataset_fast(
        self,
        dataset_loaded: DS,
        inform: TextDatasetInform,
        target_field: str,
    ) -> DS:
        """Process text dataset with optimizations."""
        if dataset_loaded.column_names is not None:
            if inform.content_field is not None and inform.content_field not in dataset_loaded.column_names:
                raise RuntimeError(
                    f"Couldn't find {inform.content_field} in available columns({dataset_loaded.column_names})."
                )

        def transform_fn(example):
            if inform.content_field is None:
                return example
            try:
                result = {target_field: example[inform.content_field]}
            except KeyError as e:
                raise KeyError(f"couldn't access {inform.content_field} Available Keys {example.keys()}.") from e

            if inform.additional_fields:
                for field in inform.additional_fields:
                    if field in example:
                        result[field] = example[field]

            return result

        # Check if dataset is streaming
        is_streaming = hasattr(dataset_loaded, "_is_streaming") or hasattr(dataset_loaded, "__iter__")

        if is_streaming:
            # For streaming datasets, don't use num_proc or load_from_cache_file
            ds_processed = dataset_loaded.map(transform_fn, batched=False)
            if inform.preprocessing_fn:
                ds_processed = ds_processed.map(inform.preprocessing_fn, batched=False)
        else:
            # For regular datasets, use optimizations
            ds_processed = dataset_loaded.map(
                transform_fn,
                num_proc=self.num_workers,
                batched=False,
                load_from_cache_file=True,
            )
            if inform.preprocessing_fn:
                ds_processed = ds_processed.map(
                    inform.preprocessing_fn,
                    num_proc=self.num_workers,
                    batched=False,
                    load_from_cache_file=True,
                )

        return ds_processed

    def _process_visual_dataset_fast(
        self,
        dataset_loaded,
        inform: VisualDatasetInform,
        text_target_field: str,
        image_target_field: str,
    ):
        """Process visual dataset with optimizations."""
        try:
            from PIL import Image as PILImage  # type:ignore
        except ImportError as e:
            raise ImportError(
                "The 'pillow' library is required for visual datasets. Install it with 'pip install pillow'."
            ) from e

        def transform_fn(example):
            if inform.content_field is None:
                return example
            result = {image_target_field: example[inform.pixel_field]}

            if inform.content_field and inform.content_field in example:
                result[text_target_field] = example[inform.content_field]

            return result

        # Check if dataset is streaming
        is_streaming = hasattr(dataset_loaded, "_is_streaming") or hasattr(dataset_loaded, "__iter__")

        if is_streaming:
            # For streaming datasets
            ds_processed = dataset_loaded.map(transform_fn, batched=False)

            if inform.image_size:

                def resize_images(examples):
                    if isinstance(examples[image_target_field], list):
                        resized = []
                        for img in examples[image_target_field]:
                            if isinstance(img, PILImage.Image):
                                resized.append(img.resize(inform.image_size, PILImage.BILINEAR))
                            else:
                                resized.append(img)
                        examples[image_target_field] = resized
                    else:
                        if isinstance(examples[image_target_field], PILImage.Image):
                            examples[image_target_field] = examples[image_target_field].resize(
                                inform.image_size, PILImage.BILINEAR
                            )
                    return examples

                ds_processed = ds_processed.map(resize_images, batched=True)

            if inform.preprocessing_fn:
                ds_processed = ds_processed.map(inform.preprocessing_fn, batched=False)
        else:
            # For regular datasets
            ds_processed = dataset_loaded.map(
                transform_fn,
                num_proc=self.num_workers,
                batched=False,
                load_from_cache_file=True,
            )

            if inform.image_size:

                def resize_images(examples):
                    if isinstance(examples[image_target_field], list):
                        resized = []
                        for img in examples[image_target_field]:
                            if isinstance(img, PILImage.Image):
                                resized.append(img.resize(inform.image_size, PILImage.BILINEAR))
                            else:
                                resized.append(img)
                        examples[image_target_field] = resized
                    else:
                        if isinstance(examples[image_target_field], PILImage.Image):
                            examples[image_target_field] = examples[image_target_field].resize(
                                inform.image_size, PILImage.BILINEAR
                            )
                    return examples

                ds_processed = ds_processed.map(
                    resize_images,
                    num_proc=self.num_workers,
                    batched=True,
                    load_from_cache_file=True,
                )

            if inform.preprocessing_fn:
                ds_processed = ds_processed.map(
                    inform.preprocessing_fn,
                    num_proc=self.num_workers,
                    batched=False,
                    load_from_cache_file=True,
                )

        return ds_processed

    def _optimized_interleave(self, datasets, seed=None, shuffle_buffer_size=None):
        """Optimized dataset interleaving with prefetching."""
        try:
            from datasets import interleave_datasets
        except ImportError as e:
            raise ImportError("The 'datasets' library is required. Install it with 'pip install datasets'.") from e

        # Don't wrap datasets with prefetch_stream for interleaving
        # The datasets library handles its own optimizations
        interleaved = interleave_datasets(
            datasets,
            seed=seed,
            stopping_strategy="first_exhausted",
        )

        if shuffle_buffer_size:
            interleaved = interleaved.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

        return interleaved

    def _optimized_batch(self, dataset, batch_size: int):
        """Optimized batching with prefetching."""
        # Don't wrap with prefetch_stream as datasets library handles batching
        return dataset.batch(batch_size)

    def create_data_iterator(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        prefetch: bool = True,
    ) -> tp.Iterator:
        """
        Create an optimized data iterator.

        Args:
            dataset: Input dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            drop_last: Drop last incomplete batch
            prefetch: Use prefetching

        Returns:
            Data iterator
        """
        if shuffle:
            dataset = dataset.shuffle()

        batched = self.stream_optimizer.batch_stream(iter(dataset), batch_size)

        if drop_last:
            batched = (batch for batch in batched if len(batch) == batch_size)

        if prefetch:
            batched = self.stream_optimizer.prefetch_stream(batched)

        return batched


# Alias for backward compatibility
DataManager = FastDataManager
