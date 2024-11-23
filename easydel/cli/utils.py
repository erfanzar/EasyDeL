# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import argparse
import inspect
from dataclasses import MISSING, is_dataclass
from typing import get_type_hints


def gather_class_param_names(cls):
	"""
	Gather parameter names from a class's __init__ method.

	Args:
	    cls: The class to gather parameter names from.

	Returns:
	    A list of parameter names.
	"""
	return [
		param_name
		for param_name, param in inspect.signature(cls.__init__).parameters.items()
		if param_name != "self"
	]


def gather_dataclass_param_names(dataclass_type):
	"""
	Gather field names from a dataclass.

	Args:
	    dataclass_type: The dataclass type to gather field names from.

	Returns:
	    A list of field names.
	"""
	return [
		field_name for field_name, field_type in get_type_hints(dataclass_type).items()
	]


def get_param_names(cls):
	"""
	Get parameter or field names from a class or dataclass.

	Args:
	    cls: The class or dataclass type to get parameter/field names from.

	Returns:
	    A list of parameter/field names.
	"""
	return (
		gather_dataclass_param_names(cls)
		if is_dataclass(cls)
		else gather_class_param_names(cls)
	)


def get_parser_from_class(cls):
	"""
	Create an argument parser from a class or dataclass.

	This function automatically creates command-line arguments based on the parameters
	of the class's __init__ method or the fields of the dataclass.

	Args:
	    cls: The class or dataclass type to create the parser from.

	Returns:
	    An argparse.ArgumentParser instance.
	"""
	return (
		create_parser_from_dataclass(cls)
		if is_dataclass(cls)
		else create_parser_from_class(cls)
	)


def create_parser_from_class(cls):
	"""
	Create an argument parser from a regular class.

	Args:
	    cls: The class to create the parser from.

	Returns:
	    An argparse.ArgumentParser instance.
	"""
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
	"""
	Create an argument parser from a dataclass.

	Args:
	    dataclass_type: The dataclass type to create the parser from.

	Returns:
	    An argparse.ArgumentParser instance.
	"""
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
		elif field_type.__class__.__name__ == "_LiteralGenericAlias":
			choices = eval(field_type.__str__().replace("typing.Literal", ""))
			parser.add_argument(
				f"--{field_name}",
				type=str,
				choices=choices,
				default=default,
				required=required,
			)
		else:
			parser.add_argument(
				f"--{field_name}",
				type=field_type,
				default=default,
				required=required,
			)

	return parser
