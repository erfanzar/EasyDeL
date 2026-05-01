# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from typing import Literal, TypeAlias, assert_never

from transformers import AutoTokenizer as AnyTokenizer

from ..openai_api_modules import DeltaMessage

ReasoningParserName: TypeAlias = Literal[
    "deepseek_r1",
    "deepseek-r1",
    "deepseek_v3",
    "glm45",
    "holo2",
    "kimi_k2",
    "qwen3",
    "qwen3_reasoning",
    "granite",
    "olmo3",
    "mistral",
    "seed_oss",
    "ernie45",
    "step3p5",
    "step3.5",
    "openai_gptoss",
    "gptoss",
    "gemma4",
    "identity",
    "none",
    "passthrough",
    "minimax_m2",
    "minimax_m2_append_think",
    "step3",
    "hunyuan_a13b",
]


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
        """Initialize the reasoning parser with a tokenizer.

        Args:
            tokenizer: Tokenizer instance for encoding/decoding tokens.
                Stored as model_tokenizer attribute.
        """
        self.model_tokenizer = tokenizer
        self.assume_reasoning: bool = False

    def configure_prompt_context(self, prompt_text: str, prompt_token_ids: Sequence[int]) -> None:
        """Configure parser state from prompt context for this request.

        Parsers that need prompt-aware behavior (for example, when chat templates
        inject a reasoning start token via ``add_generation_prompt``) can override
        this hook. The default implementation is a no-op.

        Args:
            prompt_text: Prompt text that is sent to the model.
            prompt_token_ids: Token IDs for ``prompt_text``.
        """
        del prompt_text, prompt_token_ids

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the tokenizer vocabulary mapping tokens to IDs.

        Returns:
            Dictionary mapping token strings to their integer IDs.
        """
        return self.model_tokenizer.get_vocab()

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Decide whether the reasoning block is closed in the given token sequence.

        Implementations typically scan ``input_ids`` for the parser-
        specific end marker (the ``</think>`` token id, the
        ``<channel|>`` close marker, or whichever boundary the model
        uses). This is consulted by upstream code (for example the
        delegating parser) to know whether the engine has crossed the
        reasoning boundary so visible content emission can resume.

        Args:
            input_ids: Token IDs decoded so far.

        Returns:
            ``True`` when the reasoning section has ended in
            ``input_ids``, ``False`` while it is still open.
        """
        raise NotImplementedError

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Strip reasoning-only token IDs from a decoded sequence.

        Used by callers that need the post-reasoning portion of the
        generation as raw token IDs (for example to feed it into a
        downstream tokenizer-aware tool parser). Implementations should
        return the suffix following the reasoning end marker and may
        return the input unchanged when no reasoning section was
        detected.

        Args:
            input_ids: Full sequence of decoded token IDs.

        Returns:
            Token IDs corresponding to visible content, with reasoning
            tokens removed.
        """
        raise NotImplementedError

    def extract_reasoning(
        self,
        model_output: str,
        request=None,
    ) -> tuple[str | None, str | None]:
        """Split a complete generation into reasoning and visible content.

        Called once per request after generation finishes, this method
        is responsible for locating the model-specific reasoning markers
        in ``model_output`` and returning the two halves separately.
        Returning ``None`` for a half signals that nothing was detected
        (for example when the model never opened the reasoning block, or
        when reasoning mode is asymmetric and no visible content was
        produced).

        Args:
            model_output: Complete decoded text from the model.
            request: Optional request context (for parsers whose grammar
                depends on request flags such as ``enable_thinking``).

        Returns:
            Tuple ``(reasoning_content, visible_content)`` where each
            element is the corresponding portion of the text, or
            ``None`` when the parser found no such portion.
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
        """Stream-aware variant of :meth:`extract_reasoning`.

        Invoked once per engine snapshot during a streaming request,
        with cumulative and incremental views of both the decoded text
        and the token IDs. Implementations route the new chunk to either
        ``DeltaMessage.reasoning_content`` (still inside reasoning),
        ``DeltaMessage.content`` (already past the boundary), or both
        when the reasoning end marker straddles the chunk. Returning
        ``None`` is the legitimate way to suppress an event when only
        a boundary token arrived and there is nothing meaningful to
        forward yet.

        Args:
            previous_text: Cumulative text seen before this chunk.
            current_text: Cumulative text including the new chunk.
            delta_text: Newly produced text in the current chunk.
            previous_token_ids: Token IDs prior to the chunk.
            current_token_ids: Token IDs after the chunk has been added.
            delta_token_ids: Token IDs corresponding to ``delta_text``.
            request: Optional request context (for parsers whose grammar
                varies based on request flags).

        Returns:
            A :class:`DeltaMessage` carrying ``reasoning_content`` and/or
            ``content`` for downstream emission, or ``None`` when there
            is nothing usable to emit this step.
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
        """Retrieve a registered reasoning parser by name.

        Args:
            name: Name under which a parser was registered.

        Returns:
            The registered :class:`ReasoningParser` subclass.

        Raises:
            KeyError: If ``name`` is not present in the registry.
        """
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]
        raise KeyError(f"reasoning parser: '{name}' not found in reasoning_parsers")

    @classmethod
    def _register_module(cls, module: type, module_name: str | list[str] | None = None, force: bool = True) -> None:
        """Register one or more aliases for a parser class in the registry.

        Args:
            module: The :class:`ReasoningParser` subclass to register.
            module_name: Single name or list of aliases. ``None`` uses the
                class ``__name__``.
            force: When ``False``, raise :class:`KeyError` if any alias is
                already registered.

        Raises:
            TypeError: If ``module`` is not a :class:`ReasoningParser` subclass.
            KeyError: If ``force`` is False and an alias is already taken.
        """
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
        """Register a reasoning parser module. Can be used as a decorator or called directly.

        Args:
            name: Optional alias or list of aliases for the registered parser.
                Defaults to the class ``__name__``.
            force: Overwrite existing registrations when ``True``.
            module: When provided, register ``module`` immediately and return
                it. When ``None``, return a decorator.

        Returns:
            The registered parser class when ``module`` is supplied, otherwise
            a decorator that performs the registration.

        Raises:
            TypeError: If ``force`` is not a bool or ``name`` has an unsupported type.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        if not (name is None or isinstance(name, str) or _is_list_of(name, str)):
            raise TypeError(f"name must be None, an instance of str, or a sequence of str, but got {type(name)}")

        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            """Register ``module`` and return it so the decorator preserves the class."""
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """Import and register a reasoning parser from an external Python file.

        The file should call :meth:`register_module` (typically via decorator)
        at import time. Failures are logged at warning level rather than
        propagated.

        Args:
            plugin_path: Filesystem path to the Python module to load.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        try:
            _import_from_path(module_name, plugin_path)
        except Exception:
            import logging

            logging.getLogger(__name__).warning("Failed to load module '%s' from %s.", module_name, plugin_path)


def _import_from_path(module_name: str, file_path: str | os.PathLike):
    """Import a Python module from a file path dynamically.

    Args:
        module_name: Name to associate with the loaded module in ``sys.modules``.
        file_path: Filesystem path to the ``.py`` source file.

    Returns:
        The newly imported module object.

    Raises:
        ModuleNotFoundError: If the file cannot be located or no spec is created.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _is_list_of(value: object, typ, *, check: Literal["first", "all"] = "first") -> bool:
    """Return ``True`` when ``value`` is a list whose elements match ``typ``.

    Args:
        value: Object to inspect.
        typ: Type expected for the list elements.
        check: ``"first"`` checks only the first element (cheap heuristic),
            ``"all"`` checks every element.

    Returns:
        ``True`` when ``value`` is a list and the requested elements all
        match ``typ``; ``False`` otherwise.
    """
    if not isinstance(value, list):
        return False
    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)
    assert_never(check)
