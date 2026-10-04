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

"""DeepSeek-V4.1 implementations of the optional eSurge model hooks.

The per-token side inputs follow the generic contract in :mod:`easydel.inference.esurge.multimodal.side_inputs`:

* engram layers with host-resident fp8 tables get ``engram_rows`` / ``engram_scales`` gathered on the host while the
  step is prepared, so the compiled program does no table lookup (no ``jax.pure_callback`` on the device path);
* engram layers with device tables get ``engram_hash_ids`` (looked up in-program);
* the vision-language model also gets ``token_types`` (image-span routing and engram dead mask).
"""

from __future__ import annotations

import typing as tp

import numpy as np


def prepare_tokenizer(tokenizer: tp.Any, config: tp.Any) -> np.ndarray | None:
    """Build and validate the compressed-token map used by the engram hasher.

    Args:
        tokenizer: Fast tokenizer, or ``None`` when the identity mapping may be used.
        config: Model configuration.

    Returns:
        The compressed token map, or ``None`` when the identity map is valid.

    Raises:
        ValueError: If a fast tokenizer is required or its vocabulary disagrees with the config.
    """
    if not getattr(config, "engram_layer_ids", None):
        return None
    expected = int(config.engram_compressed_vocab_size)
    if tokenizer is None or getattr(tokenizer, "backend_tokenizer", None) is None:
        if expected != int(config.vocab_size):
            raise ValueError(
                "This model hashes engram n-grams over the tokenizer's compressed vocabulary "
                f"(engram_compressed_vocab_size={expected}), but no fast tokenizer is available to build it."
            )
        return None

    from easydel.modules.deepseek_v41._engram_hash import build_engram_token_map

    token_map = build_engram_token_map(tokenizer)
    got = int(np.max(token_map)) + 1 if token_map.size else 0
    if got != expected:
        raise ValueError(
            f"tokenizer compressed vocab size {got} != config.engram_compressed_vocab_size {expected}; "
            "the tokenizer does not match this checkpoint."
        )
    return token_map


def _engram_modules(model: tp.Any) -> list[tp.Any]:
    """The model's engram modules ordered by ``layer_hash_index`` (empty without engram layers)."""
    backbone = getattr(model, "model", model)
    found = [layer.engram for layer in getattr(backbone, "layers", ()) if getattr(layer, "engram", None) is not None]
    return sorted(found, key=lambda module: int(module.layer_hash_index))


def _engram_shape(config: tp.Any) -> tuple[int, int]:
    """``(n_engram_layers, n_hash_columns)``."""
    return len(config.engram_layer_ids), (int(config.engram_max_ngram_size) - 1) * int(config.engram_n_heads)


def _host_gathered(model: tp.Any) -> bool:
    """Whether every engram layer keeps its fp8 table in host RAM (rows are then gathered while preparing a step)."""
    modules = _engram_modules(model)
    return bool(modules) and all(bool(getattr(module, "uses_host_table", False)) for module in modules)


def side_input_specs(model: tp.Any) -> dict[str, tp.Any]:
    """Per-token side inputs the V4.1 forward takes in eSurge (see the module docstring)."""
    from easydel.inference.esurge.multimodal.side_inputs import SideInputSpec

    config = model.config
    specs: dict[str, SideInputSpec] = {}
    if getattr(config, "engram_layer_ids", None):
        n_layers, n_cols = _engram_shape(config)
        if _host_gathered(model):
            head_dim = int(config.engram_head_dim)
            from easydel.layers.embeddings._engram import FP8_SCALE_BLOCK

            specs["engram_rows"] = SideInputSpec((n_layers, n_cols, head_dim), np.uint8)
            specs["engram_scales"] = SideInputSpec((n_layers, n_cols, head_dim // FP8_SCALE_BLOCK), np.uint8)
        else:
            specs["engram_hash_ids"] = SideInputSpec((n_layers, n_cols), np.int32)
    if bool(getattr(model, "esurge_requires_token_types", False)):
        specs["token_types"] = SideInputSpec((), np.int32, -1)
    return specs


class DeepseekV41SideInputProvider:
    """Builds V4.1's per-token side inputs for one eSurge step from each request's token history."""

    def __init__(self, model: tp.Any, token_map: np.ndarray | None) -> None:
        """Bind the provider to a model.

        Args:
            model: The served V4.1 model.
            token_map: Compressed token map from :func:`prepare_tokenizer` (``None`` uses the identity map).
        """
        self.specs = side_input_specs(model)
        self.hasher = None
        self.engram = []
        if getattr(model.config, "engram_layer_ids", None):
            from ._serving import EngramWindowHasher, build_engram_hasher

            self.hasher = EngramWindowHasher(build_engram_hasher(model.config, token_map))
            if "engram_rows" in self.specs:
                self.engram = _engram_modules(model)

    def build(self, windows: tp.Sequence[tp.Any], num_tokens: int) -> dict[str, np.ndarray]:
        """Arrays ``[num_tokens, ...]`` per declared spec; window rows packed back to back, padding at each fill."""
        out = {
            name: np.full((int(num_tokens), *spec.per_token_shape), spec.fill, dtype=spec.dtype)
            for name, spec in self.specs.items()
        }
        hash_ids = None
        if self.hasher is not None:
            hash_ids = np.zeros((int(num_tokens), self.hasher.n_layers, self.hasher.n_columns), dtype=np.int32)
        types_out = out.get("token_types")
        off = 0
        for window in windows:
            n = int(window.end) - int(window.start)
            if n <= 0:
                continue
            types = window.token_types
            if types_out is not None and types is not None and window.start < types.shape[0]:
                hi = min(int(window.end), int(types.shape[0]))
                types_out[off : off + hi - int(window.start)] = types[int(window.start) : hi]
            if hash_ids is not None:
                hash_ids[off : off + n] = self.hasher.hash_window(window.token_ids, types, window.start, window.end)
            off += n
        if hash_ids is not None:
            if self.engram:
                # Host gather of only the live rows; padded rows keep zero bytes (never read downstream).
                for layer, module in enumerate(self.engram):
                    rows, scales = module.gather_host_rows(hash_ids[:off, layer])
                    out["engram_rows"][:off, layer] = rows
                    out["engram_scales"][:off, layer] = scales
            else:
                out["engram_hash_ids"] = hash_ids
        return out


def build_side_input_provider(model: tp.Any, tokenizer_state: tp.Any) -> DeepseekV41SideInputProvider:
    """eSurge hook: the V4.1 side-input provider."""
    return DeepseekV41SideInputProvider(model, tokenizer_state)


def build_image_expander(config: tp.Any) -> tp.Any:
    """Build the model's scheduler-side image prompt expander."""
    from ._serving import DeepseekV41ImageExpander

    return DeepseekV41ImageExpander(config)


def compute_prefill_embeddings(
    model: tp.Any, prompt_token_ids: tp.Sequence[int], features: tp.Sequence[tp.Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Encode and merge this model's image features for eSurge prefill."""
    from ._serving import compute_prefill_embeddings as compute

    return compute(model, prompt_token_ids, features)


def register_esurge_hooks() -> None:
    """Attach the optional serving hooks to the registered V4.1 model classes."""
    from .modeling_deepseek_v41 import DeepseekV41ForCausalLM, DeepseekV41ForConditionalGeneration

    for cls in (DeepseekV41ForCausalLM, DeepseekV41ForConditionalGeneration):
        cls.esurge_prepare_tokenizer = staticmethod(prepare_tokenizer)
        cls.esurge_side_input_specs = staticmethod(side_input_specs)
        cls.esurge_side_input_provider = staticmethod(build_side_input_provider)
    DeepseekV41ForConditionalGeneration.esurge_image_expander_builder = staticmethod(build_image_expander)
    DeepseekV41ForConditionalGeneration.esurge_compute_prefill_embeddings = staticmethod(compute_prefill_embeddings)
