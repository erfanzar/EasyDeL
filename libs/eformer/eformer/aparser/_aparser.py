# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


import dataclasses
import json
import os
import sys
import types
import typing as tp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path

import yaml

from eformer.paths import ePath

DataClass = tp.NewType("DataClass", tp.Any)
DataClassType = tp.NewType("DataClassType", tp.Any)


def string_to_bool(v: str | bool) -> bool:
    """Convert a string to a boolean.

    Accepts various string representations for truthy and falsy values.
    Case-insensitive matching is used.

    Args:
        v: Value to convert. Can be a string or already a boolean.

    Returns:
        Boolean value corresponding to the input.

    Raises:
        ArgumentTypeError: If the string cannot be interpreted as boolean.

    Example:
        >>> string_to_bool("yes")
        True
        >>> string_to_bool("false")
        False
        >>> string_to_bool("1")
        True

    Accepted truthy values: "yes", "true", "t", "y", "1"
    Accepted falsy values: "no", "false", "f", "n", "0"
    """
    if isinstance(v, bool):
        return v
    lower_v = v.lower()
    if lower_v in ("yes", "true", "t", "y", "1"):
        return True
    elif lower_v in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError(
        f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
    )


def make_choice_type_function(choices: list[tp.Any]) -> tp.Callable[[str], tp.Any]:
    """Create a type converter function for argparse choices.

    Creates a function that maps string representations back to their original
    values from a list of choices. This is useful for Literal and Enum types
    where argparse receives strings but the original values may be different types.

    Args:
        choices: List of valid choice values.

    Returns:
        A function that converts a string argument to its corresponding choice
        value, or returns the original string if no match is found.

    Example:
        >>> converter = make_choice_type_function([1, 2, 3])
        >>> converter("2")
        2
        >>> converter = make_choice_type_function(["small", "large"])
        >>> converter("small")
        'small'
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def Argu(
    *,
    aliases: str | list[str] | None = None,
    help: str | None = None,  # noqa
    default: tp.Any = dataclasses.MISSING,
    default_factory: tp.Callable[[], tp.Any] = dataclasses.MISSING,
    metadata: dict | None = None,
    **kwargs,
) -> dataclasses.Field:
    """Create a dataclass field with argument parsing metadata.

    A convenience wrapper around dataclasses.field() that adds metadata for
    command-line argument generation. Use this to specify help text, aliases,
    and other argparse options for dataclass fields.

    Args:
        aliases: Alternative command-line names for this argument.
                 Can be a single string or list of strings.
                 Example: aliases=["-lr", "--rate"]
        help: Help text displayed in --help output.
        default: Default value for the field.
        default_factory: Factory function for mutable default values.
        metadata: Additional metadata dictionary to extend.
        **kwargs: Additional arguments passed to dataclasses.field().

    Returns:
        A dataclass Field object with argument metadata.

    Example:
        >>> from dataclasses import dataclass
        >>> from eformer.aparser import Argu, DataClassArgumentParser
        >>>
        >>> @dataclass
        >>> class Config:
        ...     learning_rate: float = Argu(
        ...         default=1e-4,
        ...         aliases=["-lr"],
        ...         help="Learning rate for optimizer"
        ...     )
        ...     output_dir: str = Argu(
        ...         default="./output",
        ...         help="Directory for saving outputs"
        ...     )
        >>>
        >>> parser = DataClassArgumentParser(Config)
        >>> # Can use: --learning-rate 0.01 OR -lr 0.01
    """
    if metadata is None:
        metadata = {}
    if aliases is not None:
        metadata["aliases"] = aliases
    if help is not None:
        metadata["help"] = help

    return dataclasses.field(metadata=metadata, default=default, default_factory=default_factory, **kwargs)


class DataClassArgumentParser(ArgumentParser):
    """ArgumentParser that generates arguments from dataclass type hints.

    This class extends argparse.ArgumentParser to automatically create command-line
    arguments from dataclass field definitions. It supports multiple dataclasses,
    various field types, and configuration loading from files.

    Supported field types:
        - Basic types: str, int, float, bool
        - Optional types: Optional[T] / T | None
        - Literal types: Literal["a", "b", "c"]
        - Enum types: MyEnum with automatic choices
        - List types: list[T] with nargs="+"

    Special handling:
        - Boolean fields get --no-{field} variants when default is True
        - Fields with underscores also accept hyphenated names
        - Aliases can be specified via Argu() metadata

    Attributes:
        dataclass_types: List of dataclass types to generate arguments for.

    Example:
        >>> from dataclasses import dataclass
        >>> from typing import Literal
        >>>
        >>> @dataclass
        >>> class TrainConfig:
        ...     batch_size: int = 32
        ...     learning_rate: float = 1e-4
        ...     optimizer: Literal["adam", "sgd"] = "adam"
        ...     use_amp: bool = True
        >>>
        >>> parser = DataClassArgumentParser(TrainConfig)
        >>> config, = parser.parse_args_into_dataclasses(["--batch-size", "64"])
        >>> print(config.batch_size)
        64

    Multiple dataclasses:
        >>> parser = DataClassArgumentParser([TrainConfig, ModelConfig])
        >>> train_cfg, model_cfg = parser.parse_args_into_dataclasses()
    """

    dataclass_types: tp.Iterable[DataClassType]

    def __init__(
        self,
        dataclass_types: DataClassType | tp.Iterable[DataClassType],
        **kwargs: tp.Any,
    ) -> None:
        """Initialize the parser with one or more dataclass types.

        Args:
            dataclass_types: A single dataclass type or iterable of dataclass types.
                            Arguments are generated from all provided dataclasses.
            **kwargs: Additional arguments passed to ArgumentParser.
                     Defaults to ArgumentDefaultsHelpFormatter if not specified.
        """
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    @staticmethod
    def _parse_dataclass_field(parser: ArgumentParser, field: dataclasses.Field) -> None:
        """Convert a dataclass field into a corresponding argparse argument.

        Handles type conversion, default values, choices, and special cases
        like boolean flags and Optional types.

        Args:
            parser: ArgumentParser or argument group to add the argument to.
            field: Dataclass field to convert.

        Raises:
            RuntimeError: If the field has an unresolved string type annotation.
            ValueError: If the field has an unsupported Union type.
        """

        long_options = [f"--{field.name}"]
        if "_" in field.name:
            long_options.append(f"--{field.name.replace('_', '-')}")

        kwargs = field.metadata.copy()

        if isinstance(field.type, str):
            raise RuntimeError(
                f"Unresolved type detected for field '{field.name}'. Ensure that type annotations are fully resolved."
            )

        aliases = kwargs.pop("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        origin_type = getattr(field.type, "__origin__", None)
        if origin_type in (tp.Union, getattr(types, "UnionType", None)):
            union_args = field.type.__args__
            if len(union_args) == 2 and type(None) in union_args:
                field.type = next(arg for arg in union_args if arg is not type(None))
                origin_type = getattr(field.type, "__origin__", None)
            else:
                raise ValueError(
                    f"Only tp.Optional types (tp.Union[T, None]) are supported for "
                    f"field '{field.name}', got {field.type}."
                )

        bool_kwargs: dict[str, tp.Any] = {}
        if field.type is bool:
            bool_kwargs = copy(kwargs)
            kwargs["type"] = string_to_bool
            default_val = False if field.default is dataclasses.MISSING else field.default
            kwargs["default"] = default_val
            kwargs["nargs"] = "?"
            kwargs["const"] = True

        elif origin_type is tp.Literal or (isinstance(field.type, type) and issubclass(field.type, Enum)):
            if origin_type is tp.Literal:
                kwargs["choices"] = field.type.__args__
            else:
                kwargs["choices"] = [member.value for member in field.type]
            kwargs["type"] = make_choice_type_function(kwargs["choices"])
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True

        elif isclass(field.type) and issubclass(field.type, list):
            kwargs["type"] = field.type.__args__[0]
            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True

        current_type = kwargs["type"]
        type_args = tp.get_args(current_type)

        if type_args and type(None) in type_args:
            non_none_args = [arg for arg in type_args if arg is not type(None)]

            if len(non_none_args) == 1:
                kwargs["type"] = non_none_args[0]
            elif len(non_none_args) > 1:
                kwargs["type"] = tp.Union[tuple(non_none_args)]  # noqa:UP007

        parser.add_argument(*long_options, *aliases, **kwargs)

        if field.type is bool and field.default is True:
            bool_kwargs["default"] = False
            parser.add_argument(
                f"--no_{field.name}",
                f"--no-{field.name.replace('_', '-')}",
                action="store_false",
                dest=field.name,
                **bool_kwargs,
            )

    def _add_dataclass_arguments(self, dtype: DataClassType) -> None:
        """Add arguments for all init-enabled fields of a dataclass.

        Fields with init=False are skipped. If the dataclass has an
        _argument_group_name attribute, arguments are added to a named group.

        Args:
            dtype: Dataclass type to add arguments for.

        Raises:
            RuntimeError: If type hints cannot be resolved for the dataclass.
        """
        group_name = getattr(dtype, "_argument_group_name", None)
        parser = self.add_argument_group(group_name) if group_name else self

        try:
            type_hints: dict[str, type] = tp.get_type_hints(dtype)
        except NameError as e:
            raise RuntimeError(
                f"Type resolution failed for {dtype}. Consider declaring the class in global scope or disabling "
                "PEP 563 (postponed evaluation of annotations)."
            ) from e
        except TypeError as ex:
            if sys.version_info < (3, 10) and "unsupported operand type(s) for |" in str(ex):
                python_version = ".".join(map(str, sys.version_info[:3]))
                raise RuntimeError(
                    f"Type resolution failed for {dtype} on Python {python_version}. "
                    "Please use typing.tp.Union and typing.tp.Optional instead of the | syntax for union types."
                ) from ex
            raise

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field.type = type_hints[field.name]
            self._parse_dataclass_field(parser, field)

    def parse_args_into_dataclasses(
        self,
        args: list[str] | None = None,
        return_remaining_strings: bool = False,
        look_for_args_file: bool = True,
        args_filename: str | None = None,
        args_file_flag: str | None = None,
    ) -> tuple[tp.Any, ...]:
        """Parse command-line arguments into dataclass instances.

        Parses arguments and constructs instances of all registered dataclass
        types. Supports loading additional arguments from files.

        Args:
            args: List of argument strings. If None, uses sys.argv[1:].
            return_remaining_strings: If True, include unparsed arguments in output.
            look_for_args_file: If True, look for a .args file matching the script name.
            args_filename: Explicit path to an args file to load.
            args_file_flag: Command-line flag for specifying args file(s).

        Returns:
            Tuple of dataclass instances, one per registered dataclass type.
            If return_remaining_strings is True, the last element is a list
            of unparsed argument strings.

        Raises:
            ValueError: If there are unknown arguments and return_remaining_strings
                       is False.

        Example:
            >>> parser = DataClassArgumentParser([TrainConfig, ModelConfig])
            >>> train, model = parser.parse_args_into_dataclasses()
            >>>
            >>> # With remaining args
            >>> train, model, remaining = parser.parse_args_into_dataclasses(
            ...     return_remaining_strings=True
            ... )
        """
        if args_file_flag or args_filename or (look_for_args_file and sys.argv):
            args_files: list[Path] = []
            if args_filename:
                args_files.append(Path(args_filename))
            elif look_for_args_file and sys.argv:
                args_files.append(Path(sys.argv[0]).with_suffix(".args"))

            if args_file_flag:
                args_file_parser = ArgumentParser(add_help=False)
                args_file_parser.add_argument(args_file_flag, type=str, action="append")
                cfg, args = args_file_parser.parse_known_args(args=args)
                cmd_args_file_paths = getattr(cfg, args_file_flag.lstrip("-"), None)
                if cmd_args_file_paths:
                    args_files.extend(Path(p) for p in cmd_args_file_paths)

            file_args: list[str] = []
            for args_file in args_files:
                if args_file.exists():
                    file_args.extend(args_file.read_text(encoding="utf-8").split())

            if args is None:
                args = sys.argv[1:]

            args = file_args + args

        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            field_names = {f.name for f in dataclasses.fields(dtype) if f.init}
            init_args = {k: v for k, v in vars(namespace).items() if k in field_names}

            for key in init_args:
                delattr(namespace, key)
            outputs.append(dtype(**init_args))
        if namespace.__dict__:
            outputs.append(namespace)

        if return_remaining_strings:
            return (*outputs, remaining_args)
        elif remaining_args:
            raise ValueError(f"Some arguments were not used by DataClassArgumentParser: {remaining_args}")
        return tuple(outputs)

    def parse_dict(self, args: dict[str, tp.Any], allow_extra_keys: bool = False) -> tuple[tp.Any, ...]:
        """Parse a dictionary of configuration values into dataclass instances.

        Useful for programmatic configuration or loading from config files.

        Args:
            args: Dictionary with keys matching dataclass field names.
            allow_extra_keys: If True, ignore keys that don't match any field.
                            If False, raise ValueError for unknown keys.

        Returns:
            Tuple of dataclass instances, one per registered dataclass type.

        Raises:
            ValueError: If allow_extra_keys is False and unknown keys are present.

        Example:
            >>> parser = DataClassArgumentParser(TrainConfig)
            >>> config, = parser.parse_dict({
            ...     "learning_rate": 0.001,
            ...     "batch_size": 64
            ... })
        """
        unused_keys = set(args.keys())
        outputs = []
        for dtype in self.dataclass_types:
            field_names = {f.name for f in dataclasses.fields(dtype) if f.init}
            init_args = {k: v for k, v in args.items() if k in field_names}
            unused_keys -= init_args.keys()
            outputs.append(dtype(**init_args))
        if not allow_extra_keys and unused_keys:
            raise ValueError(f"Unused keys in configuration: {sorted(unused_keys)}")
        return tuple(outputs)

    def parse_json_file(
        self,
        json_file: str | os.PathLike,
        allow_extra_keys: bool = False,
    ) -> tuple[tp.Any, ...]:
        """Load a JSON file and parse it into dataclass instances.

        Args:
            json_file: Path to the JSON configuration file.
                      Supports both local paths and GCS paths (gs://).
            allow_extra_keys: If True, ignore keys that don't match any field.

        Returns:
            Tuple of dataclass instances, one per registered dataclass type.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If allow_extra_keys is False and unknown keys are present.

        Example:
            >>> parser = DataClassArgumentParser(TrainConfig)
            >>> config, = parser.parse_json_file("config.json")
        """
        data = json.loads(ePath(json_file).read_text())
        return self.parse_dict(data, allow_extra_keys=allow_extra_keys)

    def parse_yaml_file(
        self,
        yaml_file: str | os.PathLike,
        allow_extra_keys: bool = False,
    ) -> tuple[tp.Any, ...]:
        """Load a YAML file and parse it into dataclass instances.

        Args:
            yaml_file: Path to the YAML configuration file.
            allow_extra_keys: If True, ignore keys that don't match any field.

        Returns:
            Tuple of dataclass instances, one per registered dataclass type.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the file contains invalid YAML.
            ValueError: If allow_extra_keys is False and unknown keys are present.

        Example:
            >>> parser = DataClassArgumentParser(TrainConfig)
            >>> config, = parser.parse_yaml_file("config.yaml")
        """
        yaml_text = Path(yaml_file).read_text(encoding="utf-8")
        data = yaml.safe_load(yaml_text)
        return self.parse_dict(data, allow_extra_keys=allow_extra_keys)
