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

"""Abstract base classes for tool call parsing in EasyDeL inference.

This module provides the foundational classes for extracting and parsing tool/function
calls from Large Language Model outputs. It includes:

- ToolParser: Abstract base class that defines the interface for all tool parsers
- ToolParserManager: Registry for managing and retrieving parser implementations
- Utility functions for dynamic module loading

The module supports both batch and streaming extraction of tool calls, with
model-specific parsers handling different formats (JSON, XML, pythonic, etc.).

Example:
    Creating a custom tool parser:

    >>> from easydel.inference.tools.abstract_tool import ToolParser, ToolParserManager
    >>>
    >>> @ToolParserManager.register_module("my_custom_parser")
    ... class MyCustomParser(ToolParser):
    ...     def extract_tool_calls(self, model_output, request):
    ...         # Custom parsing logic
    ...         ...
    ...
    ...     def extract_tool_calls_streaming(self, previous_text, current_text, ...):
    ...         # Streaming parsing logic
    ...         ...

    Using the parser manager:

    >>> parser_class = ToolParserManager.get_tool_parser("my_custom_parser")
    >>> parser = parser_class(tokenizer)
    >>> result = parser.extract_tool_calls(model_output, request)

See Also:
    - easydel.inference.tools.parsers: Model-specific parser implementations
    - easydel.inference.tools.utils: Utility functions for parsing
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Literal, TypeAlias, assert_never

from transformers import AutoTokenizer as AnyTokenizer

from ..openai_api_modules import ChatCompletionRequest, DeltaMessage, ExtractedToolCallInformation

ToolParserName: TypeAlias = Literal[
    "glm47",
    "glm-4.7",
    "openai",
    "phi4_mini_json",
    "qwen3_xml",
    "pythonic",
    "deepseek_v3",
    "step3",
    "internlm",
    "xlam",
    "glm45",
    "glm-4.5",
    "gigachat3",
    "minimax_m2",
    "deepseek_v31",
    "jamba",
    "ernie45",
    "llama3_json",
    "llama4_json",
    "granite-20b-fc",
    "deepseek_v32",
    "kimi_k2",
    "minimax",
    "longcat",
    "granite",
    "olmo3",
    "hunyuan_a13b",
    "step3p5",
    "step3.5",
    "seed_oss",
    "hermes",
    "qwen3_coder",
    "functiongemma",
    "llama4_pythonic",
    "mistral",
]


class ToolParser:
    """Abstract base class for tool call parsing from LLM outputs.

    This class provides the foundation for extracting function/tool calls from
    language model outputs. Subclasses implement model-specific parsing logic
    for different tool call formats (JSON, XML, pythonic, etc.).

    The parser maintains state for streaming responses and provides methods
    for both batch and streaming extraction of tool calls. Each parser instance
    is associated with a specific tokenizer, which may be used for token-level
    parsing decisions.

    Attributes:
        prev_tool_call_arr (list[dict]): History of previously parsed tool calls.
            Used to track state across streaming chunks.
        current_tool_id (int): ID counter for tool calls in current session.
            Starts at -1 and increments for each new tool call detected.
        current_tool_name_sent (bool): Flag indicating if the current tool's
            name has been sent in streaming mode.
        streamed_args_for_tool (list[str]): Buffer for accumulating streaming
            tool arguments before they are complete.
        model_tokenizer (AnyTokenizer): Tokenizer instance for the model.
            Used for token-level operations and vocabulary access.

    Example:
        >>> # Creating a parser instance (using a concrete subclass)
        >>> from easydel.inference.tools import HermesToolParser
        >>> parser = HermesToolParser(tokenizer)
        >>>
        >>> # Batch extraction
        >>> result = parser.extract_tool_calls(model_output, request)
        >>> if result.tools_called:
        ...     for tool_call in result.tool_calls:
        ...         print(f"Called: {tool_call.function.name}")
        >>>
        >>> # Streaming extraction
        >>> delta = parser.extract_tool_calls_streaming(
        ...     prev_text, curr_text, delta_text,
        ...     prev_ids, curr_ids, delta_ids, request
        ... )

    Note:
        This is an abstract class. Use model-specific subclasses like
        HermesToolParser, MistralToolParser, Qwen3XMLToolParser, etc.
        for actual parsing.

    See Also:
        - ToolParserManager: For registering and retrieving parser implementations
        - ExtractedToolCallInformation: Return type for batch extraction
        - DeltaMessage: Return type for streaming extraction
    """

    def __init__(self, tokenizer: AnyTokenizer):
        """Initialize the tool parser with a tokenizer.

        Sets up the parser's initial state for tracking tool calls across
        streaming chunks and stores the tokenizer for vocabulary access.

        Args:
            tokenizer (AnyTokenizer): A tokenizer instance (typically from
                HuggingFace transformers) that will be used for token-level
                operations. Should be compatible with the model being used.

        Example:
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
            >>> parser = MyToolParser(tokenizer)
        """
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the tokenizer vocabulary mapping tokens to IDs.

        Returns the vocabulary dictionary from the underlying tokenizer.
        This property is cached for performance since vocabulary doesn't
        change after tokenizer initialization.

        Returns:
            dict[str, int]: A dictionary mapping token strings to their
                integer IDs in the tokenizer's vocabulary.

        Example:
            >>> parser = MyToolParser(tokenizer)
            >>> vocab = parser.vocab
            >>> print(vocab.get("<tool_call>"))  # Get ID for tool call token
            32001

        Note:
            Only PreTrainedTokenizerFast is guaranteed to have a .vocab
            attribute or get_vocab() method. Other tokenizer types may
            have different interfaces.
        """
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request parameters for model-specific requirements.

        Override this method in subclasses to modify request parameters like
        system prompts, tool definitions, or formatting before processing.
        This is useful when models require specific formatting or additional
        instructions for proper tool calling behavior.

        Args:
            request (ChatCompletionRequest): The original chat completion
                request containing messages, tools, and other parameters.

        Returns:
            ChatCompletionRequest: The potentially modified request. Default
                implementation returns the request unchanged.

        Example:
            >>> class CustomParser(ToolParser):
            ...     def adjust_request(self, request):
            ...         # Add system instruction for tool usage
            ...         if request.tools and not any(
            ...             m.role == "system" for m in request.messages
            ...         ):
            ...             request.messages.insert(0, ChatMessage(
            ...                 role="system",
            ...                 content="Use tools when appropriate."
            ...             ))
            ...         return request

        Note:
            Some models may need to reformat tool definitions, add specific
            system instructions, or modify stop sequences for proper tool
            calling. Override this method to implement such adjustments.
        """
        return request

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (batch mode).

        Parses the entire model response to identify and extract tool/function
        calls. This method is used for non-streaming responses where the complete
        output is available at once.

        Args:
            model_output (str): Complete text generated by the model. This may
                contain tool calls in various formats (JSON, XML, etc.) depending
                on the specific parser implementation.
            request (ChatCompletionRequest): Original request containing tool
                definitions, messages, and other parameters. Used to validate
                tool calls against available tools.

        Returns:
            ExtractedToolCallInformation: An object containing:
                - tools_called (bool): Whether any tool calls were detected
                - tool_calls (list[ToolCall]): List of extracted tool calls
                - content (str | None): Non-tool-call content from the response

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
                The base class does not provide parsing logic.

        Example:
            >>> output = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
            >>> result = parser.extract_tool_calls(output, request)
            >>> if result.tools_called:
            ...     print(result.tool_calls[0].function.name)
            'get_weather'

        Note:
            This method is stateless - it doesn't use or modify instance state.
            Each call parses the output independently. Parser implementations
            should handle malformed tool calls gracefully, returning the
            original content if parsing fails.
        """
        raise NotImplementedError("AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output.

        Processes incremental model output to identify partial tool calls
        and emit appropriate streaming updates. Maintains state across
        calls (via instance attributes) to handle incomplete JSON/XML
        structures as they are generated token-by-token.

        Args:
            previous_text (str): Text accumulated up to the previous call.
                Used to determine what content has already been processed.
            current_text (str): Text accumulated including the current chunk.
                The full text generated so far.
            delta_text (str): New text in the current chunk. The difference
                between current_text and previous_text.
            previous_token_ids (Sequence[int]): Token IDs accumulated up to
                the previous call. May be used for token-level parsing.
            current_token_ids (Sequence[int]): Token IDs including current
                chunk. The full sequence of tokens generated so far.
            delta_token_ids (Sequence[int]): New token IDs in the current
                chunk. Useful for detecting special tokens.
            request (ChatCompletionRequest): Original request with tool
                definitions. Used to validate tool names and parameters.

        Returns:
            DeltaMessage | None: An incremental update containing partial
                tool call information (tool_calls field with index, function
                name delta, and/or arguments delta). Returns None if no
                tool call update is available for this chunk.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
                The base class does not provide streaming parsing logic.

        Example:
            >>> # Process streaming chunks
            >>> for chunk in stream:
            ...     delta = parser.extract_tool_calls_streaming(
            ...         prev_text, curr_text, delta_text,
            ...         prev_ids, curr_ids, delta_ids, request
            ...     )
            ...     if delta and delta.tool_calls:
            ...         # Send incremental tool call update
            ...         yield delta

        Note:
            This method is stateful - it uses instance variables (prev_tool_call_arr,
            current_tool_id, etc.) to track parsing progress across streaming
            chunks. Make sure to use the same parser instance for an entire
            streaming session.
        """
        raise NotImplementedError("AbstractToolParser.extract_tool_calls_streaming has not been implemented!")


class ToolParserManager:
    """Registry and manager for tool parser implementations.

    This class provides a centralized registry for tool parsers, allowing
    dynamic registration and retrieval of parser implementations. It supports
    both decorator-based and direct registration patterns, making it easy to
    add custom parsers for new model formats.

    The manager enables:
        - Registration of custom parser implementations by name
        - Retrieval of parsers by registered name
        - Dynamic loading of parser plugins from external Python files
        - Validation that parsers inherit from ToolParser base class
        - Registration under multiple names (aliases)

    Attributes:
        tool_parsers (dict[str, type]): Class-level registry mapping parser
            names (strings) to parser classes. Shared across all instances.

    Example:
        Registering a parser using the decorator pattern:

        >>> @ToolParserManager.register_module("my_parser")
        ... class MyCustomParser(ToolParser):
        ...     def extract_tool_calls(self, model_output, request):
        ...         ...
        ...     def extract_tool_calls_streaming(self, ...):
        ...         ...

        Registering with multiple names:

        >>> @ToolParserManager.register_module(["parser_v1", "parser_latest"])
        ... class ParserV1(ToolParser):
        ...     ...

        Direct registration:

        >>> ToolParserManager.register_module(
        ...     name="external_parser",
        ...     module=ExternalParserClass
        ... )

        Retrieving a parser:

        >>> parser_class = ToolParserManager.get_tool_parser("my_parser")
        >>> parser = parser_class(tokenizer)
        >>> result = parser.extract_tool_calls(output, request)

        Loading from external file:

        >>> ToolParserManager.import_tool_parser("/path/to/custom_parser.py")
        >>> parser_class = ToolParserManager.get_tool_parser("custom")

    See Also:
        - ToolParser: Base class that registered parsers must inherit from
        - import_from_path: Utility function for loading external modules
    """

    tool_parsers: dict[str, type] = {}  # noqa: RUF012

    @classmethod
    def get_tool_parser(cls, name: str) -> type:
        """Retrieve a registered tool parser by name.

        Looks up a parser class in the registry by its registered name.
        This is the primary method for obtaining parser classes for
        instantiation.

        Args:
            name (str): The name of the parser to retrieve. Must match
                a name that was used during registration.

        Returns:
            type: The parser class registered with the given name. This
                class can be instantiated with a tokenizer to create a
                parser instance.

        Raises:
            KeyError: If no parser is registered with the given name.
                The error message includes the requested name.

        Example:
            >>> # Get parser class by name
            >>> HermesParser = ToolParserManager.get_tool_parser("hermes")
            >>> parser = HermesParser(tokenizer)
            >>>
            >>> # Handle missing parser
            >>> try:
            ...     parser_class = ToolParserManager.get_tool_parser("nonexistent")
            ... except KeyError as e:
            ...     print(f"Parser not found: {e}")
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        raise KeyError(f"tool helper: '{name}' not found in tool_parsers")

    @classmethod
    def _register_module(cls, module: type, module_name: str | list[str] | None = None, force: bool = True) -> None:
        """Internal method to register a parser module in the registry.

        This is the core registration logic used by register_module().
        It validates the module type, handles name defaults, and manages
        registry updates.

        Args:
            module (type): The parser class to register. Must be a subclass
                of ToolParser.
            module_name (str | list[str] | None, optional): Name(s) to register
                the parser under. If None, uses the class's __name__ attribute.
                Can be a single string or list of strings for multiple aliases.
                Defaults to None.
            force (bool, optional): If True, overwrites any existing registration
                with the same name. If False, raises KeyError on name conflict.
                Defaults to True.

        Raises:
            TypeError: If the module is not a subclass of ToolParser.
            KeyError: If force=False and a name is already registered.

        Note:
            This is an internal method. Use register_module() for the public
            API with full validation and decorator support.
        """
        if not issubclass(module, ToolParser):
            raise TypeError(f"module must be subclass of ToolParser, but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.tool_parsers:
                existed_module = cls.tool_parsers[name]
                raise KeyError(f"{name} is already registered at {existed_module.__module__}")
            cls.tool_parsers[name] = module

    @classmethod
    def register_module(
        cls, name: str | list[str] | None = None, force: bool = True, module: type | None = None
    ) -> type | Callable:
        """Register a tool parser module in the registry.

        Can be used as a decorator or called directly. Supports registering
        a parser under one or multiple names (aliases). This is the primary
        API for adding new parsers to the registry.

        Args:
            name (str | list[str] | None, optional): Name(s) to register the
                parser under. If None, uses the class's __name__. Can be a
                single string or a list of strings for multiple aliases.
                Defaults to None.
            force (bool, optional): If True, overwrites any existing registration
                with the same name(s). If False, raises KeyError on conflict.
                Defaults to True.
            module (type | None, optional): Parser class to register directly.
                If None, returns a decorator that can be applied to a class.
                Defaults to None.

        Returns:
            type | Callable: If module is provided, returns the module unchanged
                (for direct registration). If module is None, returns a decorator
                function that will register the decorated class.

        Raises:
            TypeError: If force is not a boolean, or if name has an invalid type.
            KeyError: If force=False and name conflicts with existing registration.

        Example:
            As a decorator with a name:

            >>> @ToolParserManager.register_module("custom")
            ... class CustomParser(ToolParser):
            ...     ...

            As a decorator with multiple names:

            >>> @ToolParserManager.register_module(["v1", "version1", "latest"])
            ... class V1Parser(ToolParser):
            ...     ...

            As a decorator without a name (uses class name):

            >>> @ToolParserManager.register_module()
            ... class MyParser(ToolParser):
            ...     ...  # Registered as "MyParser"

            Direct registration:

            >>> ToolParserManager.register_module(
            ...     name="alternate",
            ...     module=ExistingParser
            ... )
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        if not (name is None or isinstance(name, str) or is_list_of(name, str)):
            raise TypeError(f"name must be None, an instance of str, or a sequence of str, but got {type(name)}")

        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """Import and register a tool parser from an external Python file.

        Dynamically loads a Python module from the specified file path.
        The module should contain parser classes that are either decorated
        with @register_module or that manually call register_module during
        module initialization.

        Args:
            plugin_path (str): Absolute or relative file path to the Python
                module containing tool parser definition(s). The file should
                be a valid Python module (.py file).

        Example:
            External parser file (custom_parser.py):

            >>> # custom_parser.py
            >>> from easydel.inference.tools import ToolParser, ToolParserManager
            >>>
            >>> @ToolParserManager.register_module("external_custom")
            ... class ExternalCustomParser(ToolParser):
            ...     def extract_tool_calls(self, model_output, request):
            ...         ...
            ...     def extract_tool_calls_streaming(self, ...):
            ...         ...

            Loading the external parser:

            >>> ToolParserManager.import_tool_parser("/path/to/custom_parser.py")
            >>> parser_class = ToolParserManager.get_tool_parser("external_custom")
            >>> parser = parser_class(tokenizer)

        Note:
            - The parser class(es) in the file should use @register_module
              decorator or call register_module during module initialization.
            - If loading fails, a message is printed but no exception is raised.
            - The module is registered in sys.modules under its filename
              (without extension).
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            print("Failed to load module '%s' from %s.", module_name, plugin_path)
            return


def import_from_path(module_name: str, file_path: str | os.PathLike):
    """Import a Python module from a file path dynamically.

    Loads a Python file as a module and registers it in sys.modules.
    This is the underlying function used by ToolParserManager.import_tool_parser()
    for loading external parser plugins.

    Args:
        module_name (str): Name to register the module under in sys.modules.
            This is the name that can be used for subsequent imports.
        file_path (str | os.PathLike): Path to the Python file to import.
            Can be absolute or relative.

    Returns:
        module: The imported module object. Can be used to access module
            attributes and classes directly.

    Raises:
        ModuleNotFoundError: If the file cannot be loaded as a module
            (e.g., file doesn't exist or is not a valid Python file).

    Example:
        >>> # Import a custom module
        >>> custom_module = import_from_path("my_parsers", "/path/to/parsers.py")
        >>> print(custom_module.MyParserClass)
        <class 'my_parsers.MyParserClass'>
        >>>
        >>> # Module is now available for regular import
        >>> import my_parsers
        >>> parser = my_parsers.MyParserClass(tokenizer)

    Note:
        This implementation is based on the official Python importlib recipe:
        https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def is_list_of(value: object, typ, *, check: Literal["first", "all"] = "first") -> bool:
    """Check if a value is a list containing elements of a specific type.

    Validates that a value is a list and that its elements are of the
    expected type. Supports checking only the first element (for performance)
    or all elements (for strictness).

    Args:
        value (object): The value to check. Will return False if not a list.
        typ: The expected type of list elements. Can be any type that works
            with isinstance().
        check (Literal["first", "all"], optional): Checking mode. "first"
            checks only the first element (faster for homogeneous lists).
            "all" checks every element (stricter). Defaults to "first".

    Returns:
        bool: True if value is a list of the specified type. Returns True
            for empty lists. Returns False if value is not a list or if
            type check fails.

    Example:
        >>> is_list_of(["a", "b", "c"], str)
        True
        >>> is_list_of(["a", 1, "c"], str)  # Only checks first element
        True
        >>> is_list_of(["a", 1, "c"], str, check="all")  # Checks all elements
        False
        >>> is_list_of([], str)  # Empty list passes
        True
        >>> is_list_of("not a list", str)
        False
        >>> is_list_of([1, 2, 3], int, check="all")
        True

    Note:
        The "first" mode is suitable when you expect homogeneous lists and
        want better performance. Use "all" when strict validation is required.
    """
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)
