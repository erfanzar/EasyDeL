# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Trainer-local model/state helpers."""

from __future__ import annotations

import typing as tp

import spectrax as spx
from eformer.paths import ePath
from transformers import AutoTokenizer

from easydel.infra.base_state import EasyDeLState


def apply_tokenizer_overrides(tokenizer: object, *, eos_token: str | None = None, pad_token: str | None = None) -> None:
    """Apply configured token overrides to a tokenizer in place.

    Args:
        tokenizer: Tokenizer whose special tokens will be updated.
        eos_token: Replacement EOS token, or ``None`` to preserve the current one.
        pad_token: Replacement padding token, or ``None`` to preserve the current one.
    """
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    if pad_token is not None:
        tokenizer.pad_token = pad_token


def apply_chat_template_path(tokenizer: object, chat_template_path: str) -> None:
    """Assign a chat template from a file or a pretrained tokenizer source.

    Args:
        tokenizer: Tokenizer whose chat template will be updated in place.
        chat_template_path: Local or cloud template file, or tokenizer source
            accepted by ``AutoTokenizer.from_pretrained``.

    Raises:
        ValueError: If the tokenizer source does not provide a chat template.
    """
    path = ePath(chat_template_path)
    if path.is_file():
        tokenizer.chat_template = path.read_text(encoding="utf-8")
        return

    template_tokenizer = AutoTokenizer.from_pretrained(chat_template_path)
    chat_template = template_tokenizer.chat_template
    if chat_template is None:
        raise ValueError(f"No chat_template found at {chat_template_path!r}.")
    tokenizer.chat_template = chat_template


def zero_reward_func(completions: tp.Sequence[object], **_: object) -> list[float]:
    """Return one zero reward per completion for teacher-only distillation."""
    return [0.0] * len(completions)


def disable_state_dropout(state: EasyDeLState | None) -> EasyDeLState | None:
    """Return a state whose module graph is in eval mode.

    Optional state slots pass through as ``None``. Concrete states expose their
    module, call ``eval()`` to disable dropout-like stochastic layers, and then
    export the updated graph definition back into the state when supported.
    """
    if state is None:
        return None
    module = state.model
    module.eval()
    try:
        graphdef, _ = spx.export(module)
    except TypeError:
        return state
    return state.replace(graphdef=graphdef)


def reject_string_model_id(model_id: str, *, role: str = "model") -> None:
    """Raise for TRL-style string model identifiers passed to trainers.

    These EasyDeL trainer paths require already-initialized modules or states.
    Rejecting strings prevents hidden model loading, network access, and
    runtime behavior that differs from the rest of the training suite.
    """
    raise ValueError(
        f"EasyDeL trainers do not load {role} from string ids. "
        f"Load {model_id!r} with the appropriate AutoEasyDeLModel class first, "
        "then pass the initialized module or EasyDeLState to the trainer."
    )
