import typing

import termcolor
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass
from typing import Optional, Union, List
import copy
from transformers import PreTrainedTokenizer


@dataclass
class DataProcessorArguments:
    prompt_field: str
    max_position_embeddings: int
    truncation_mode: typing.Literal["keep_end", "keep_start"] = "keep_end"
    use_deepcopy: bool = True
    with_indices: bool = False
    with_rank: bool = False
    batched: bool = False
    batch_size: Optional[int] = 1000
    drop_last_batch: bool = False
    remove_columns: Optional[Union[str, List[str]]] = None
    load_from_cache_file: Optional[bool] = None
    writer_batch_size: Optional[int] = 1000
    disable_nullable: bool = False
    fn_kwargs: Optional[dict] = None
    num_proc: Optional[int] = None
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}"
    new_fingerprint: Optional[str] = None
    desc: Optional[str] = None


class DataProcessor:
    @staticmethod
    def process_data(
            data: Dataset,
            tokenizer: PreTrainedTokenizer,
            arguments: DataProcessorArguments,
            field: str = 'train'
    ):
        data = copy.deepcopy(data) if arguments.use_deepcopy else data

        map_kwargs = {}

        if arguments.remove_columns is None:
            arguments.remove_columns = data.column_names

        _reqs = {
            "with_indices": arguments.with_indices,
            "with_rank": arguments.with_rank,
            "batched": arguments.batched,
            "batch_size": arguments.batch_size,
            "drop_last_batch": arguments.drop_last_batch,
            "remove_columns": arguments.remove_columns,
            "load_from_cache_file": arguments.load_from_cache_file,
            "writer_batch_size": arguments.writer_batch_size,
            "disable_nullable": arguments.disable_nullable,
            "fn_kwargs": arguments.fn_kwargs,
            "num_proc": arguments.num_proc,
            "suffix_template": arguments.suffix_template,
            "new_fingerprint": arguments.new_fingerprint,
            "desc": arguments.desc
        }

        for k, v in _reqs.items():
            if v is not None:
                map_kwargs[k] = v

        if tokenizer.pad_token is None:
            termcolor.cprint(
                "Tokenizer Doesn't include the padding "
                "token so i set `(tokenizer.pad_token = tokenizer.eos_token)`",
                color="red", force_color=True
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' if arguments.truncation_mode == "keep_end" else 'right'
        data = data.map(
            lambda x: tokenizer(
                x[arguments.prompt_field],
                max_length=arguments.max_position_embeddings,
                padding='max_length',
                return_tensors='jax'
            ),
            **map_kwargs
        )

        return DatasetDict({field: data})
