import argparse
import inspect
from dataclasses import MISSING
from typing import get_type_hints


def create_parser_from_class(cls):
    parser = argparse.ArgumentParser()

    signature = inspect.signature(cls.__init__)
    type_hints = get_type_hints(cls.__init__)

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue

        param_type = type_hints.get(param_name, str)
        default = param.default if param.default != inspect.Parameter.empty else None
        required = default is inspect.Parameter.empty

        if isinstance(param_type, bool):
            parser.add_argument(
                f"--{param_name}",
                action="store_true",
                default=default,
            )
        else:
            parser.add_argument(
                f"--{param_name}",
                type=param_type,
                default=default,
                required=required,
            )

    return parser


def create_parser_from_dataclass(dataclass_type):
    parser = argparse.ArgumentParser()

    hints = get_type_hints(dataclass_type)
    for field_name, field_type in hints.items():
        field_info = dataclass_type.__dataclass_fields__[field_name]

        # Check if the field has a default value
        if field_info.default is not MISSING:
            default = field_info.default
        elif field_info.default_factory is not MISSING:
            default = field_info.default_factory()
        else:
            default = None

        required = default is MISSING

        if isinstance(field_type, bool):
            parser.add_argument(
                f"--{field_name}",
                action="store_true" if default is False else "store_false",
                default=default,
            )
        else:
            parser.add_argument(
                f"--{field_name}",
                type=field_type,
                default=default,
                required=required,
            )

    return parser
