# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract base classes for reasoning content extraction in EasyDeL inference.

This module provides the foundational classes for extracting reasoning/thinking
content from Large Language Model outputs. It includes:

- ReasoningParser: Abstract base class that defines the interface for all reasoning parsers
- ReasoningParserManager: Registry for managing and retrieving parser implementations

The module supports both batch and streaming extraction of reasoning content, with
model-specific parsers handling different formats (<think>, [THINK], text delimiters, etc.).

Note:
    Ideas and design patterns are inspired by vLLM's reasoning parser implementation.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Literal, assert_never

from transformers import AutoTokenizer as AnyTokenizer

from ..openai_api_modules import DeltaMessage


class ReasoningParser:
    """Abstract base class for reasoning content extraction from LLM outputs.

    This class provides the foundation for extracting thinking/reasoning content
    from language model outputs. Subclasses implement model-specific parsing logic
    for different reasoning formats (<think>...</think>, [THINK]...[/THINK], etc.).

    The parser maintains state for streaming responses and provides methods
    for both batch and streaming extraction of reasoning content.

    Attributes:
        model_tokenizer (AnyTokenizer): Tokenizer instance for the model.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        self.model_tokenizer = tokenizer
        self.assume_reasoning: bool = False

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the tokenizer vocabulary mapping tokens to IDs."""
        return self.model_tokenizer.get_vocab()

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check if reasoning section has ended based on token IDs.

        Args:
            input_ids: Sequence of token IDs to check.

        Returns:
            True if the reasoning section is complete.
        """
        raise NotImplementedError

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract content token IDs (excluding reasoning tokens).

        Args:
            input_ids: Complete sequence of token IDs.

        Returns:
            Token IDs for content only (reasoning tokens removed).
        """
        raise NotImplementedError

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning and content from complete output (batch mode).

        Args:
            model_output: Complete text output from model.
            request: Optional request context.

        Returns:
            Tuple of (reasoning_content, content). Either may be None.
        """
        raise NotImplementedError

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request=None,
    ) -> DeltaMessage | None:
        """Extract reasoning content from streaming model output.

        Args:
            previous_text: Text accumulated before this chunk.
            current_text: Text including current chunk.
            delta_text: New text in current chunk.
            previous_token_ids: Token IDs before current chunk.
            current_token_ids: Token IDs including current chunk.
            delta_token_ids: New token IDs in current chunk.
            request: Optional request context.

        Returns:
            DeltaMessage with reasoning_content and/or content fields set,
            or None if no update available.
        """
        raise NotImplementedError


class ReasoningParserManager:
    """Registry and manager for reasoning parser implementations.

    Provides a centralized registry for reasoning parsers with decorator-based
    and direct registration support. Follows the same pattern as ToolParserManager.
    """

    reasoning_parsers: dict[str, type] = {}  # noqa: RUF012

    @classmethod
    def get_reasoning_parser(cls, name: str) -> type:
        """Retrieve a registered reasoning parser by name."""
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]
        raise KeyError(f"reasoning parser: '{name}' not found in reasoning_parsers")

    @classmethod
    def _register_module(cls, module: type, module_name: str | list[str] | None = None, force: bool = True) -> None:
        if not issubclass(module, ReasoningParser):
            raise TypeError(f"module must be subclass of ReasoningParser, but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.reasoning_parsers:
                existed_module = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered at {existed_module.__module__}")
            cls.reasoning_parsers[name] = module

    @classmethod
    def register_module(
        cls, name: str | list[str] | None = None, force: bool = True, module: type | None = None
    ) -> type | Callable:
        """Register a reasoning parser module. Can be used as a decorator or called directly."""
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        if not (name is None or isinstance(name, str) or _is_list_of(name, str)):
            raise TypeError(f"name must be None, an instance of str, or a sequence of str, but got {type(name)}")

        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """Import and register a reasoning parser from an external Python file."""
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        try:
            _import_from_path(module_name, plugin_path)
        except Exception:
            import logging

            logging.getLogger(__name__).warning("Failed to load module '%s' from %s.", module_name, plugin_path)


def _import_from_path(module_name: str, file_path: str | os.PathLike):
    """Import a Python module from a file path dynamically."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _is_list_of(value: object, typ, *, check: Literal["first", "all"] = "first") -> bool:
    if not isinstance(value, list):
        return False
    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)
    assert_never(check)
