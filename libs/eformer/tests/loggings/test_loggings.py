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

"""Tests for logging utilities."""

import logging

from eformer.loggings import LazyLogger


class DummyLogger:
    def __init__(self):
        self.calls = []

    def log(self, level, message, *args, **kwargs):
        self.calls.append((level, message, args, kwargs))

    def info(self, message, *args, **kwargs):
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.log(logging.WARNING, message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self.log(logging.DEBUG, message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.log(logging.ERROR, message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self.log(logging.ERROR, message, *args, **kwargs)


def test_log_once_handles_unhashable_args():
    logger = LazyLogger("test-log-once")
    dummy = DummyLogger()
    logger._logger = dummy
    logger._ensure_initialized = lambda: None
    logger.clear_once_cache()

    logger.info_once("hello", {"a": 1})
    logger.info_once("hello", {"a": 1})

    assert len(dummy.calls) == 1


def test_uppercase_levels_dispatch_to_methods():
    logger = LazyLogger("test-uppercase")
    dummy = DummyLogger()
    logger._logger = dummy
    logger._ensure_initialized = lambda: None

    logger.INFO("uppercase info")

    assert dummy.calls
    assert dummy.calls[0][0] == logging.INFO
