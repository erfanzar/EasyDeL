# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Custom exception types for the ejKernel library.

This module defines the exception hierarchy used throughout ejKernel to signal
runtime errors related to unsupported platforms, missing backends, or invalid
kernel configurations. All custom exceptions inherit from standard Python
exception classes to allow both specific and generic exception handling.

Example:
    >>> from ejkernel.errors import EjkernelRuntimeError
    >>> try:
    ...     raise EjkernelRuntimeError("Flash attention not supported on CPU")
    ... except RuntimeError:
    ...     print("Caught as generic RuntimeError")
"""


class EjkernelRuntimeError(RuntimeError):
    """Runtime error raised when an ejKernel operation cannot proceed.

    This exception is raised when a kernel or operation is invoked on an
    unsupported platform, with an incompatible backend, or with invalid
    configuration that prevents execution.

    Inherits from ``RuntimeError`` so callers can catch it either specifically
    or via the standard ``RuntimeError`` handler.

    Example:
        >>> raise EjkernelRuntimeError(
        ...     "Triton flash attention requires GPU platform"
        ... )
    """
