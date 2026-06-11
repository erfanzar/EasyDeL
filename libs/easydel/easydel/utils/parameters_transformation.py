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


"""Parameter format conversion between PyTorch/HuggingFace and EasyDeL.

This module is the bridge that lets EasyDeL load HuggingFace checkpoints
into JAX/Spectrax modules and vice versa. It groups three responsibilities:

* :class:`DtypeHandler` -- string-to-``jnp.dtype`` mapping and float tensor
  re-cast helpers used throughout the converters.
* :class:`TensorConverter` -- low-level torch-tensor / JAX-array conversion
  (with optional zero-copy DLPack transfer).
* :class:`StateDictConverter` and :class:`ModelConverter` -- the high-level
  HuggingFace -> EasyDeL and EasyDeL -> HuggingFace converters, including
  MoE expert consolidation and key renaming logic.

Most public entry points are exposed via the top-level ``easydel.utils``
package as :class:`ModelConverter`, :class:`StateDictConverter`, and
:class:`TensorConverter`.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import inspect
import os
import re
import typing as tp
import warnings
from collections.abc import Mapping

import jax
import jax.extend
import ml_dtypes
import numpy as np
from eformer.loggings import get_logger
from jax import dlpack
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.utils.helpers import check_bool_flag

from .analyze_memory import SMPMemoryMonitor
from .traversals import unflatten_dict

if tp.TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.infra.base_module import EasyDeLBaseModule


mem_ops = SMPMemoryMonitor(5)
logger = get_logger(__name__)
EASYDEL_PREFERRED_HOST_COPY_INDEX = int(
    os.getenv("EASYDEL_PREFERRED_HOST_COPY_INDEX", os.getenv("EASYDEL_PERFRED_HOST_COPY_INDEX", "0"))
)
_preferred_host_copy_raw = str(
    os.getenv("EASYDEL_PREFERRED_HOST_COPY", os.getenv("EASYDEL_PERFRED_HOST_COPY", "cpu"))
).lower()
EASYDEL_PREFERRED_HOST_COPY: str | None = None if _preferred_host_copy_raw == "none" else _preferred_host_copy_raw


class DtypeHandler:
    """Static helpers for parsing dtype aliases and re-casting float tensors.

    EasyDeL's checkpoint and conversion code receives dtypes from CLI flags
    and YAML configs as short strings (``"bf16"``, ``"fp16"``, ``"fp8_e4m3fn"``,
    …); :meth:`get_dtype` is the single canonical place that maps those
    aliases to ``jnp.dtype`` objects, including the various 8-bit FP variants.
    :meth:`float_tensor_to_dtype` is the matching tensor-side helper used by
    :class:`StateDictConverter`/:class:`ModelConverter` to coerce *only*
    floating-point arrays while leaving integer/bool tensors (e.g. masks,
    indices) at their native precision.

    The class deliberately exposes only ``@staticmethod`` callables so it
    can be used as a namespace without instantiation.
    """

    @staticmethod
    def get_dtype(dtype: str | jnp.dtype) -> jnp.dtype:
        """Convert string dtype representation to JAX dtype.

        Args:
            dtype: Either a ``jnp.dtype`` (returned unchanged) or a short
                alias string such as ``"bf16"``, ``"fp16"``, ``"fp8_e4m3fn"``.

        Returns:
            The corresponding ``jnp.dtype``.

        Raises:
            KeyError: If ``dtype`` is a string not in the supported alias map.
        """
        if isinstance(dtype, str):
            dtype_map: dict[str, jnp.dtype] = {
                "bf16": jnp.bfloat16,
                "bfloat16": jnp.bfloat16,
                "fp16": jnp.float16,
                "float16": jnp.float16,
                "fp32": jnp.float32,
                "float32": jnp.float32,
                "fp64": jnp.float64,
                "float64": jnp.float64,
                "fp8": jnp.float8_e5m2,
                "nvfp8": jnp.float8_e4m3,
                "mxfp8": jnp.float8_e5m2,
                "mxfp4": jnp.float4_e2m1fn,
                "fp8_e4m3fn": jnp.float8_e4m3fn,
                "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
                "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
                "fp8_e5m2": jnp.float8_e5m2,
                "fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
                "float8_e4m3fn": jnp.float8_e4m3fn,
                "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
                "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
                "float8_e5m2": jnp.float8_e5m2,
                "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
            }
            return dtype_map[dtype]
        return dtype

    @staticmethod
    def float_tensor_to_dtype(tensor: tp.Any, dtype: str | jnp.dtype | None) -> tp.Any:
        """Convert a float-valued tensor to the specified dtype, if applicable.

        Integer/boolean tensors and ``None``/``""`` dtypes pass through.

        Args:
            tensor: Any object exposing a ``dtype`` attribute and an
                ``astype`` method (JAX/NumPy/PyTorch arrays all qualify).
            dtype: Target dtype as a string or ``jnp.dtype``; ``None`` or
                ``""`` skip the conversion.

        Returns:
            ``tensor`` cast to ``dtype`` when its current dtype is a known
            float type; otherwise ``tensor`` unchanged.
        """
        if dtype is None or dtype == "":
            return tensor

        dtype = DtypeHandler.get_dtype(dtype)
        float_dtypes = (
            jnp.bfloat16,
            jnp.float16,
            jnp.float32,
            jnp.float64,
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fnuz,
            jnp.float8_e4m3b11fnuz,
            jnp.float8_e5m2,
            jnp.float8_e5m2fnuz,
        )

        if getattr(tensor, "dtype", None) in float_dtypes:
            tensor = tensor.astype(dtype)
        return tensor


class TensorConverter:
    """Low-level tensor adapters between PyTorch tensors and JAX arrays.

    Used by the higher-level :class:`StateDictConverter` to move parameter
    leaves between frameworks while preserving dtype semantics. Two
    transfer paths are supported:

    * **NumPy bridge** — slower but format-stable. ``bfloat16`` PyTorch
      tensors are upcast to ``float`` before going through NumPy because
      NumPy lacks native ``bfloat16`` storage.
    * **DLPack** — zero-copy capsule transfer used when both runtimes are
      on a compatible backend (CPU or CUDA). Gated by ``EASY_SAFE_TRANSFER``;
      callers can force the safe NumPy path on platforms with broken DLPack
      capsules.

    All callables are ``@staticmethod`` so the class is used as a
    namespace; ``get_torch`` is ``lru_cache``-d to avoid paying repeated
    import-time cost on hot conversion loops.
    """

    @staticmethod
    def convert_pytorch_to_jnp(tensor: tp.Any, dtype: jnp.dtype) -> jnp.ndarray:
        """Convert a PyTorch tensor to a JAX array of the requested dtype.

        ``bfloat16`` PyTorch tensors are upcast to ``float`` before going
        through NumPy because NumPy doesn't natively support ``bfloat16``.

        Args:
            tensor: A PyTorch tensor to convert.
            dtype: Target ``jnp.dtype`` for the resulting array.

        Returns:
            A JAX array with the same data and the requested dtype.
        """
        if "bfloat16" in str(tensor.dtype):
            tensor = tensor.float()
        npv = tensor.cpu().detach().numpy()
        return jnp.array(npv, dtype=dtype)

    @staticmethod
    @functools.lru_cache
    def get_torch():
        """Import and return the ``torch`` module, cached across calls.

        Returns:
            The imported ``torch`` module.
        """
        import torch

        return torch

    @staticmethod
    def jax_to_pytorch(x: jax.Array) -> tp.Any:
        """Convert a JAX array to a PyTorch tensor.

        When ``EASY_SAFE_TRANSFER`` is enabled (the default) the data is
        moved through NumPy, which is slower but avoids DLPack edge cases.
        Otherwise a zero-copy DLPack transfer is used when the JAX backend
        is CPU/GPU and CUDA is available.

        Args:
            x: JAX array to transfer.

        Returns:
            A PyTorch tensor view of the array.
        """
        if check_bool_flag("EASY_SAFE_TRANSFER", True):
            torch = TensorConverter.get_torch()
            x = jax.device_get(x)
            x = np.asarray(x)
            if not x.flags.c_contiguous:
                x = np.ascontiguousarray(x)
            if x.dtype == ml_dtypes.bfloat16:
                # ``torch.from_numpy`` cannot ingest ``ml_dtypes.bfloat16`` arrays (NumPy has
                # no native bfloat16), so the NumPy bridge raised TypeError for bf16 models.
                # Reinterpret the bits as uint16 and view back as torch bfloat16 --
                # bit-exact, no upcast, no extra copy.
                return torch.from_numpy(x.view(np.uint16)).view(torch.bfloat16)
            return torch.from_numpy(x)
        else:
            from torch import cuda
            from torch.utils import dlpack as dlpack_pt

            platform = jax.extend.backend.get_backend()
            cpu_force = not cuda.is_available()

            if (
                platform in ["cpu", "gpu"]
                and not cpu_force
                and not check_bool_flag("EASYDEL_FORCE_TORCH_USE_CPU", False)
            ):
                dl_pack_jax = dlpack.to_dlpack(
                    x,
                    stream=True if (platform == "gpu" and not cpu_force) else None,
                    src_device=next(iter(x.devices())),
                )
            else:
                dl_pack_jax = dlpack.to_dlpack(
                    jax.device_put(
                        jax.device_get(x),
                        jax.devices(EASYDEL_PREFERRED_HOST_COPY)[EASYDEL_PREFERRED_HOST_COPY_INDEX],
                    ),
                    stream=None,
                )
            return dlpack_pt.from_dlpack(dl_pack_jax)

    @staticmethod
    def pytorch_to_jax(x: tp.Any) -> jnp.ndarray:
        """Convert a PyTorch tensor to a JAX array via NumPy.

        Args:
            x: A PyTorch tensor.

        Returns:
            A JAX array with the same dtype and shape.
        """
        return jnp.asarray(x.detach().cpu().numpy())


class StateDictConverter:
    """Convert flat PyTorch ``state_dict``s to nested EasyDeL parameter trees.

    PyTorch ships parameters as a flat ``{'layer.0.attn.q_proj.weight': Tensor}``
    mapping where each entry follows PyTorch's ``[out, in]`` weight layout.
    EasyDeL's Spectrax modules expect a nested ``{tuple_path: jnp.ndarray}``
    pytree with JAX-friendly axis order (``[in, out]`` for dense layers, with
    higher-rank tensors transposed analogously). This class is the bridge.

    The high-traffic entry point is the private helper
    :meth:`_base_huggingface_to_easydel`; trainer/loader code reaches it
    through :class:`ModelConverter`, which adds MoE expert consolidation and
    config-driven hooks. Static methods on this class implement the
    individual stages: keyword filtering (:meth:`match_keywords`), per-tensor
    rewriting (:meth:`process_tensor` — applies the axis transposition,
    detects embeddings/layernorms, and applies optional ``reform_param``
    splits), and the orchestration loop that progress-bars over a state dict
    and runs each tensor through user-provided shard/callback hooks.

    Notes:
        * The class is deliberately stateless; configuration flows through
          the per-call ``config`` dict so concurrent conversions don't
          interact.
        * Tied LM-head weights are intentionally not dropped during
          conversion to keep tree-key parity with EasyDeL graphs that still
          materialise ``lm_head.kernel``.
    """

    @staticmethod
    def validate_reform_param_schema(reform_param: Mapping[str, Mapping[str, tp.Any]] | None) -> None:
        """Validate the structural contract used by ``reform_param`` rules.

        Checks rule shape and callable presence, not tensor shapes. Rules
        that merge multiple checkpoint tensors into one EasyDeL tensor must be
        bidirectional so export cannot accidentally write runtime-only layouts.

        Args:
            reform_param: Mapping of target-key pattern to rule mapping. May
                be ``None`` or empty (treated as no-op).

        Raises:
            TypeError: If a rule entry, its ``sources`` list, or one of its
                callables has the wrong type.
            ValueError: If a fusion rule is missing one of ``sources``,
                ``fuser`` or ``inverse_fuser``; or if a split rule omits
                ``inverse_spliter``.
        """
        if not reform_param:
            return
        for key, rule in reform_param.items():
            if not isinstance(rule, Mapping):
                raise TypeError(f"reform_param[{key!r}] must be a mapping, got {type(rule).__name__}")

            has_sources = "sources" in rule
            has_fuser = "fuser" in rule
            has_inverse_fuser = "inverse_fuser" in rule
            if has_sources or has_fuser or has_inverse_fuser:
                if not has_sources or not has_fuser or not has_inverse_fuser:
                    raise ValueError(
                        f"reform_param[{key!r}] fusion rules must define 'sources', 'fuser', and 'inverse_fuser'"
                    )
                sources = rule["sources"]
                if not isinstance(sources, tuple | list) or not all(isinstance(source, str) for source in sources):
                    raise TypeError(f"reform_param[{key!r}]['sources'] must be a sequence of strings")
                if not callable(rule["fuser"]):
                    raise TypeError(f"reform_param[{key!r}]['fuser'] must be callable")
                if not callable(rule["inverse_fuser"]):
                    raise TypeError(f"reform_param[{key!r}]['inverse_fuser'] must be callable")

            if "splits" in rule:
                splits = rule["splits"]
                if not isinstance(splits, tuple | list):
                    raise TypeError(f"reform_param[{key!r}]['splits'] must be a sequence")
                if "inverse_spliter" not in rule:
                    raise ValueError(f"reform_param[{key!r}] split rules must define 'inverse_spliter'")
                if not callable(rule["inverse_spliter"]):
                    raise TypeError(f"reform_param[{key!r}]['inverse_spliter'] must be callable")
                for split in splits:
                    if not isinstance(split, Mapping):
                        raise TypeError(f"reform_param[{key!r}] split entries must be mappings")
                    if not isinstance(split.get("name"), str):
                        raise TypeError(f"reform_param[{key!r}] split entries must define string 'name'")
                    if not callable(split.get("spliter")):
                        raise TypeError(f"reform_param[{key!r}] split entries must define callable 'spliter'")

    @staticmethod
    def match_keywords(string: str, required: list[str], forbidden: list[str]) -> bool:
        """Check if a string contains all required keywords and none of the forbidden ones.

        Args:
            string: The text to inspect.
            required: Substrings that must all be present in ``string``.
            forbidden: Substrings that must not appear in ``string``.

        Returns:
            ``True`` when every required keyword is present and no forbidden
            keyword appears.
        """
        return all(t in string for t in required) and not any(n in string for n in forbidden)

    @staticmethod
    def collect_reform_param_fusion_groups(
        keys: tp.Iterable[str],
        reform_param: dict | None,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, tp.Any]]]:
        """Find multi-source ``reform_param`` fusion groups present in *keys*.

        Existing ``reform_param`` entries split one incoming HF tensor into one
        or more EasyDeL leaves via ``splits``. Fusion entries are the inverse
        load-time shape: they provide ``sources`` and a ``fuser`` callable to
        materialize a target tensor before the normal per-tensor conversion
        path runs.

        Args:
            keys: Iterable of state-dict keys present in the incoming
                checkpoint.
            reform_param: Mapping of target-key pattern to rule mapping;
                ``None`` returns empty results.

        Returns:
            A pair ``(groups, rules_by_fused_key)`` where ``groups`` maps each
            fused target key to the tuple of source keys it consumes, and
            ``rules_by_fused_key`` maps the same target keys to their full
            rule mapping (for ``fuser`` lookup downstream).
        """
        if not reform_param:
            return {}, {}
        StateDictConverter.validate_reform_param_schema(reform_param)

        key_set = set(keys)
        groups: dict[str, tuple[str, ...]] = {}
        rules_by_fused_key: dict[str, dict[str, tp.Any]] = {}
        sorted_items = sorted(reform_param.items(), key=lambda x: len(x[0]), reverse=True)

        for key_check, value in sorted_items:
            if "sources" not in value or "fuser" not in value:
                continue
            target_key = key_check[:-1] if key_check.endswith("$") else key_check
            source_keys = tuple(value["sources"])
            if not source_keys:
                continue
            skip_substrings = tuple(value.get("skip_substrings", ()))
            if any(any(skip in source_key for skip in skip_substrings) for source_key in source_keys):
                continue
            if all(source_key in key_set for source_key in source_keys):
                groups[target_key] = source_keys
                rules_by_fused_key[target_key] = value

        return groups, rules_by_fused_key

    @staticmethod
    def fuse_reform_param_tensors(rule: dict[str, tp.Any], tensors: list[tp.Any]) -> tp.Any:
        """Apply a ``reform_param`` multi-source fusion rule.

        Detects whether the rule's ``fuser`` callable wants the ``torch``
        module as its first positional argument and dispatches accordingly.

        Args:
            rule: Rule mapping containing a ``"fuser"`` callable.
            tensors: Per-source tensors in the order declared by
                ``rule["sources"]``.

        Returns:
            The fused tensor returned by the ``fuser`` callable.
        """
        fuser = rule["fuser"]
        torch = TensorConverter.get_torch()
        accepted = [
            p
            for p in inspect.signature(fuser).parameters.values()
            if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL}
        ]
        if accepted and accepted[0].name in {"torch", "pt"}:
            return fuser(torch, *tensors)
        return fuser(*tensors)

    @staticmethod
    def inverse_fuse_reform_param_tensor(rule: dict[str, tp.Any], tensor: tp.Any) -> tp.Any:
        """Apply a ``reform_param`` inverse fusion rule.

        Used during EasyDeL -> HuggingFace export to recover the original
        per-source tensors from a fused EasyDeL leaf. Dispatches with or
        without a leading ``torch`` argument based on the callable signature.

        Args:
            rule: Rule mapping containing an ``"inverse_fuser"`` callable.
            tensor: The fused EasyDeL tensor to split back into its sources.

        Returns:
            Either a tuple/list of per-source tensors or a mapping of source
            name to tensor, as produced by the configured ``inverse_fuser``.
        """
        inverse_fuser = rule["inverse_fuser"]
        torch = TensorConverter.get_torch()
        accepted = [
            p
            for p in inspect.signature(inverse_fuser).parameters.values()
            if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL}
        ]
        if accepted and accepted[0].name in {"torch", "pt"}:
            return inverse_fuser(torch, tensor)
        return inverse_fuser(tensor)

    @staticmethod
    def apply_reform_param_fusions(
        state_dict: dict[str, tp.Any] | None,
        reform_param: dict | None,
    ) -> dict[str, int]:
        """Materialize all present ``reform_param`` fusion groups in-place.

        For each fusion group whose source keys are all present in
        ``state_dict``, computes the fused tensor via
        :meth:`fuse_reform_param_tensors`, writes it under the target key, and
        drops the original source entries.

        Args:
            state_dict: HuggingFace-style state dict to mutate. ``None`` is a
                no-op.
            reform_param: Mapping of rules (see :meth:`validate_reform_param_schema`).

        Returns:
            Mapping of human-readable fusion label to the number of fused
            tensors materialized for that label.
        """
        if state_dict is None:
            return {}

        groups, rules_by_fused_key = StateDictConverter.collect_reform_param_fusion_groups(
            state_dict.keys(),
            reform_param,
        )
        if not groups:
            return {}

        fused_counts: dict[str, int] = {}
        source_keys_to_drop: set[str] = set()
        for fused_key, source_keys in groups.items():
            if fused_key in state_dict:
                continue
            rule = rules_by_fused_key[fused_key]
            tensors = [state_dict[source_key] for source_key in source_keys]
            state_dict[fused_key] = StateDictConverter.fuse_reform_param_tensors(rule, tensors)
            source_keys_to_drop.update(source_keys)
            label = str(rule.get("log_label", fused_key))
            fused_counts[label] = fused_counts.get(label, 0) + 1

        for source_key in source_keys_to_drop:
            state_dict.pop(source_key, None)

        return fused_counts

    @staticmethod
    def process_tensor(key: str, tensor: tp.Any, config: dict[str, tp.Any]) -> list[tuple[tuple, jnp.ndarray]] | None:
        """Process a single PyTorch tensor into EasyDeL format.

        Applies key renaming (e.g., ``.weight`` -> ``.kernel``), axis
        transposition for dense layers, embedding/layernorm detection,
        and optional ``reform_param`` splitting rules.

        Args:
            key: Dot-separated PyTorch parameter name.
            tensor: PyTorch tensor to convert.
            config: Conversion configuration containing ``embedding_layer_names``,
                ``layernorm_names``, ``dtype``, ``reform_param``, etc.

        Returns:
            List of ``(key_tuple, jax_array)`` pairs, or ``None`` if the
            parameter should be skipped.
        """
        new_key = key

        reform_param = config.get("reform_param", None)
        if reform_param:
            sorted_items = sorted(reform_param.items(), key=lambda x: len(x[0]), reverse=True)
            for key_check, value in sorted_items:
                if value.get("already_converted", False):
                    anchor_to_end = key_check.endswith("$")
                    match_target = key_check[:-1] if anchor_to_end else key_check
                    match_index = key.find(match_target)
                    if match_index != -1:
                        after_match = key[match_index + len(match_target) :]
                        if anchor_to_end and after_match:
                            continue
                        if not after_match or after_match.startswith("."):
                            before_match = key[:match_index]
                            if not before_match or before_match.endswith("."):
                                config = config.copy()
                                config["_reform_processed"] = True
                                break
                if "splits" not in value:
                    continue
                anchor_to_end = key_check.endswith("$")
                match_target = key_check[:-1] if anchor_to_end else key_check

                match_index = key.find(match_target)
                if match_index != -1:
                    after_match = key[match_index + len(match_target) :]
                    if anchor_to_end and after_match:
                        continue
                    if not after_match or after_match.startswith("."):
                        before_match = key[:match_index]
                        if not before_match or before_match.endswith("."):
                            splits = value["splits"]
                            results = []

                            new_config = config.copy()
                            new_config["reform_param"] = {}
                            new_config["_reform_processed"] = True

                            for split in splits:
                                split_name = split["name"]
                                spliter = split["spliter"]
                                new_key_split = f"{before_match}{split_name}{after_match}"
                                tensor_split = spliter(tensor)
                                sub_results = StateDictConverter.process_tensor(
                                    new_key_split,
                                    tensor_split,
                                    new_config,
                                )
                                if sub_results:
                                    results.extend(sub_results)
                            return results

        if "weight" in key and not config.get("_reform_processed", False):
            is_embedding = any(layer_name in key for layer_name in config.get("embedding_layer_names", []))
            is_moe_expert = key in config.get("consolidated_moe_keys", set())
            ndim = len(tensor.shape)
            if not is_embedding and not is_moe_expert:
                if ndim == 2:
                    tensor = tensor.permute(1, 0)
                elif ndim == 3:
                    tensor = tensor.permute(2, 1, 0)
                elif ndim == 4:
                    tensor = tensor.permute(2, 3, 1, 0)
                elif ndim == 5:
                    tensor = tensor.permute(2, 3, 4, 1, 0)
                elif ndim == 6:
                    tensor = tensor.permute(4, 5, 3, 2, 1, 0)
            elif is_moe_expert:
                if ndim == 3:
                    tensor = tensor.permute(0, 2, 1)
            # Everything now uses .weight; no key renaming needed

        key_tuple = tuple(int(n) if n.isdigit() else n for n in new_key.split("."))

        # Do not drop tied LM-head weights during conversion.
        # Some EasyDeL module graphs still materialize `lm_head.kernel` even when
        # logits use tied embeddings at runtime; dropping this leaf causes noisy
        # "missing parameter" warnings during merge.
        #
        # Keeping it here preserves graph/tree key parity without changing tied
        # runtime behavior (`apply_lm_head` still uses embedding weights when tied).
        # if config["uses_tie_word_embedding"] and config["lm_head_name"] and key_tuple[0] == config["lm_head_name"]:
        #     return None

        array = TensorConverter.convert_pytorch_to_jnp(tensor, config["dtype"])
        return [(key_tuple, array)]

    @staticmethod
    def _base_huggingface_to_easydel(
        state_dict: dict[str, tp.Any],
        *,
        device: jax.Device | None = None,  # type:ignore
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        shard_fns: Mapping[tuple, tp.Callable] | None = None,
        dtype: jnp.dtype = jnp.float16,
        verbose: bool = True,
        callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None,
        remove_state_dict: bool = False,
        lm_head_name: str | None = None,
        uses_tie_word_embedding: bool = False,
        consolidated_moe_keys: set[str] | None = None,
        reform_param: dict | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Base conversion from a PyTorch state dict to EasyDeL nested dict format.

        Iterates over all keys in ``state_dict``, applies per-tensor
        processing (key renaming, axis transposition, dtype casting),
        optional shard functions, and a user callback.

        Args:
            state_dict: PyTorch model ``state_dict()``.
            device: Target JAX device for parameter placement.
            embedding_layer_names: Parameter name substrings identifying embeddings.
            layernorm_names: Parameter name substrings identifying layer norms.
            moe_block_names: Names of MoE block modules.
            moe_names: Names of individual MoE expert sub-modules.
            shard_fns: Optional mapping of key tuples to sharding functions.
            dtype: Target JAX dtype for converted parameters.
            verbose: Whether to display a progress bar.
            callback: Optional function called on each converted array.
            remove_state_dict: Whether to delete the input dict after conversion.
            lm_head_name: Name of the language model head parameter.
            uses_tie_word_embedding: Whether embeddings are tied with lm_head.
            consolidated_moe_keys: Set of keys that were consolidated from
                per-expert weights.
            reform_param: Optional parameter splitting/merging rules.

        Returns:
            Nested EasyDeL parameter dictionary.
        """
        try:
            import torch

            _clear = torch.cuda.empty_cache if torch.cuda.is_available() else gc.collect
        except ModuleNotFoundError:
            _clear = gc.collect

        config = {
            "embedding_layer_names": set(embedding_layer_names or []),
            "layernorm_names": set(layernorm_names or []),
            "moe_block_names": set(moe_block_names or []),
            "moe_names": set(moe_names or []),
            "lm_head_name": lm_head_name,
            "uses_tie_word_embedding": uses_tie_word_embedding,
            "dtype": dtype,
            "consolidated_moe_keys": consolidated_moe_keys or set(),
            "reform_param": reform_param,
        }

        with jax.default_device(device) if device is not None and shard_fns is None else contextlib.nullcontext():
            parameters_dict = {}
            with tqdm(total=len(state_dict), disable=not verbose, desc="Converting Model") as pbar:
                keys = sorted(state_dict.keys())
                for key in keys:
                    tensor = state_dict.get(key)
                    try:
                        # Note: memory_stats() returns None on CPU devices
                        def get_memory_bytes(device_idx):
                            """Return current bytes-in-use for a local JAX device.

                            Args:
                                device_idx: Index into ``jax.local_devices()``.

                            Returns:
                                ``stats["bytes_in_use"]`` when available,
                                otherwise ``0`` (e.g. on CPU devices that
                                don't expose ``memory_stats()``).
                            """
                            stats = jax.local_devices()[device_idx].memory_stats()
                            return stats["bytes_in_use"] if stats is not None else 0

                        bytesi = {i: get_memory_bytes(i) for i in range(jax.local_device_count())}
                        results = StateDictConverter.process_tensor(key, tensor, config)
                        if results is not None:
                            for key_tuple, jax_array in results:
                                if shard_fns and key_tuple in shard_fns:
                                    jax_array = shard_fns[key_tuple](jax_array)
                                if callback is not None:
                                    jax_array = callback(jax_array, key_tuple)
                                bytesn = {i: get_memory_bytes(i) for i in range(jax.local_device_count())}
                                change = {i: bytesn[i] - bytesi[i] for i in range(jax.local_device_count())}
                                divider = 1024**3
                                change_gb = {i: round(change[i] / divider, 4) for i in change}
                                usage_gb = {i: round(bytesn[i] / divider, 4) for i in bytesn}
                                strm = f"Sharding {'.'.join([str(i) for i in key_tuple])} change_gb: {change_gb} current_gb: {usage_gb}"
                                logger.debug(strm)
                                parameters_dict[key_tuple] = jax_array
                    except Exception as e:
                        logger.error(f"Error processing key {key}: {e!s}")
                    pbar.update(1)

            if remove_state_dict:
                del state_dict
                _clear()

            return unflatten_dict(parameters_dict)

    @staticmethod
    def apply_moe_transformations(
        state_dict: dict[str, tp.Any],
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        tensor_transform: tp.Callable | None = None,
        reform_param: dict | None = None,
        debug: bool = False,
    ) -> tuple[dict[str, tp.Any], set[str]]:
        """
        Transform MoE weights from HuggingFace format (separate experts) to EasyDel format (stacked experts).
        Converts from:
            model.layers.3.block_sparse_moe.experts.0.w3.weight -> shape (128, 256)
            model.layers.3.block_sparse_moe.experts.1.w3.weight -> shape (128, 256)
            ...
        To:
            model.layers.3.block_sparse_moe.experts.w3.weight -> shape (num_experts, 128, 256)

        Args:
            state_dict: HuggingFace ``state_dict``-style mapping. Mutated in
                place (entries are popped) for memory efficiency.
            moe_block_names: Tail names of MoE block modules (e.g.
                ``"block_sparse_moe"``).
            moe_names: Tail names of expert sub-modules (e.g.
                ``["w1", "w2", "w3"]``).
            moe_block_path: Full dotted paths to each MoE block in the model
                graph.
            moe_path: Full dotted paths to expert modules. Used to derive the
                expert container name (e.g. ``"experts"``).
            tensor_transform: Optional callable applied to each stacked tensor
                before it is written into the output dict.
            reform_param: Optional split-rule mapping reused from the rest of
                the converter; keys mentioning the experts container provide
                fallback names when the strict rule does not match.

        Returns:
            ``(new_state_dict, consolidated_keys)`` where ``new_state_dict``
            contains stacked expert tensors (and any non-expert leaves passed
            through unchanged) and ``consolidated_keys`` lists the new stacked
            keys for downstream MoE-aware logic.

        Raises:
            ValueError: If the required ``moe_path`` / ``moe_names`` /
                ``moe_block_path`` arguments are ``None``.
        """
        if not all([moe_block_names, moe_names, moe_block_path]):
            return state_dict, set()

        import torch

        if moe_path is None:
            raise ValueError("moe_path cannot be None")
        if moe_names is None:
            raise ValueError("moe_names cannot be None")
        if moe_block_path is None:
            raise ValueError("moe_block_path cannot be None")

        expected_expert_name = moe_path[0].split(".")[-2]
        expert_prefix = f".{expected_expert_name}."

        moe_names_set = set(moe_names)
        moe_stacked_paths = {
            f"{block_path}.{expected_expert_name}.{moe_name}" for block_path in moe_block_path for moe_name in moe_names
        }
        reform_param = reform_param or {}
        reform_fusion_source_paths = {
            (source[:-7] if source.endswith(".weight") else source)
            for value in reform_param.values()
            if "sources" in value
            for source in value["sources"]
            if f".{expected_expert_name}." in source
        }
        stackable_moe_paths = moe_stacked_paths | reform_fusion_source_paths
        if debug:
            logger.info("MoE converter stackable paths: %s", sorted(stackable_moe_paths))
            if reform_fusion_source_paths:
                logger.info("MoE converter reform source paths: %s", sorted(reform_fusion_source_paths))
        fallback_reform_keys = {
            key[:-1] if key.endswith("$") else key
            for key, value in reform_param.items()
            if "splits" in value and f".{expected_expert_name}." in key
        }
        sibling_expert_parents = {path.rsplit(".", 1)[0] for path in moe_path if f".{expected_expert_name}." in path}

        new_state_dict = {}
        moe_groups = {path: {} for path in stackable_moe_paths}
        consolidated_moe_keys = set()

        for key in tqdm(list(state_dict.keys()), desc="Applying MoE Transformations"):
            is_moe_expert = False
            value = state_dict.pop(key)
            if expert_prefix not in key:
                new_state_dict[key] = value
                continue

            for block_path in moe_block_path:
                block_expert_prefix = block_path + expert_prefix
                if key.startswith(block_expert_prefix):
                    remainder = key[len(block_expert_prefix) :]

                    dot_idx = remainder.find(".")
                    if dot_idx <= 0:
                        continue

                    expert_part = remainder[:dot_idx]
                    if not expert_part.isdigit():
                        continue

                    expert_idx = int(expert_part)
                    moe_name_part = remainder[dot_idx + 1 :]
                    moe_name = moe_name_part[:-7] if moe_name_part.endswith(".weight") else moe_name_part

                    target_path = f"{block_path}.{expected_expert_name}.{moe_name}"
                    if moe_name in moe_names_set or target_path in stackable_moe_paths:
                        moe_groups[target_path][expert_idx] = value
                        is_moe_expert = True
                        break

            if not is_moe_expert:
                match = re.match(rf"^(.*\.{expected_expert_name})\.(\d+)\.([^.]+)\.weight$", key)
                if match:
                    expert_parent, expert_idx_str, moe_name = match.groups()
                    target_path = f"{expert_parent}.{moe_name}"
                    if expert_parent in sibling_expert_parents and (
                        moe_name in moe_names_set
                        or target_path in fallback_reform_keys
                        or target_path in stackable_moe_paths
                    ):
                        moe_groups.setdefault(target_path, {})[int(expert_idx_str)] = value
                        is_moe_expert = True

            if not is_moe_expert:
                if debug and expert_prefix in key:
                    logger.info("MoE converter left expert key unclaimed: %s", key)
                new_state_dict[key] = value
        for target_path, expert_dict in moe_groups.items():
            if not expert_dict:
                continue

            expert_indices = sorted(expert_dict.keys())
            num_experts = len(expert_indices)
            first_tensor = expert_dict[expert_indices[0]]
            new_key = f"{target_path}.weight"

            try:
                if isinstance(first_tensor, torch.Tensor):
                    if first_tensor.device.type != "meta":
                        meta_sample = torch.empty_like(first_tensor, device="meta")
                    else:
                        meta_sample = first_tensor
                    stacked_shape = (num_experts, *meta_sample.shape)
                    stacked_tensor = torch.empty(
                        stacked_shape,
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )

                    for i, idx in enumerate(expert_indices):
                        stacked_tensor[i] = expert_dict[idx]

                else:
                    import numpy as np

                    expert_tensors = [expert_dict[idx] for idx in expert_indices]
                    stacked_tensor = np.stack(expert_tensors, axis=0)

                if tensor_transform is not None:
                    stacked_tensor = tensor_transform(stacked_tensor)

                new_state_dict[new_key] = stacked_tensor
                consolidated_moe_keys.add(new_key)
            except Exception as e:
                logger.error(f"Failed to stack MoE tensors for {target_path}: {e}")
                for idx, tensor in expert_dict.items():
                    fallback_key = (
                        f"{target_path.replace(f'.{expected_expert_name}.', f'.{expected_expert_name}.{idx}.')}.weight"
                    )
                    new_state_dict[fallback_key] = tensor

        return new_state_dict, consolidated_moe_keys

    @staticmethod
    def huggingface_to_easydel(
        state_dict: dict[str, tp.Any],
        *,
        device: jax.Device | None = None,  # type:ignore
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        shard_fns: Mapping[tuple, tp.Callable] | None = None,
        dtype: jnp.dtype = jnp.float16,
        verbose: bool = True,
        callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None,
        remove_state_dict: bool = False,
        lm_head_name: str | None = None,
        uses_tie_word_embedding: bool = False,
        reform_param: dict | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Convert a PyTorch state dict to EasyDeL format with MoE support.

        If MoE parameters are present, first stacks per-expert weights into
        consolidated tensors, then delegates to ``_base_huggingface_to_easydel``
        for the standard conversion pipeline.

        Args:
            state_dict: PyTorch model ``state_dict()``.
            device: Target JAX device.
            embedding_layer_names: Substrings identifying embedding layers.
            layernorm_names: Substrings identifying layer norm layers.
            moe_block_names: Names of MoE block modules.
            moe_names: Names of individual expert sub-modules.
            moe_block_path: Full dot-paths to MoE blocks in the model.
            moe_path: Full dot-paths to expert modules.
            shard_fns: Optional sharding functions per key tuple.
            dtype: Target JAX dtype.
            verbose: Whether to show progress.
            callback: Optional per-array callback.
            remove_state_dict: Whether to delete input dict after conversion.
            lm_head_name: Language model head parameter name.
            uses_tie_word_embedding: Whether embeddings are tied.
            reform_param: Optional splitting/merging rules.

        Returns:
            Nested EasyDeL parameter dictionary.
        """
        consolidated_moe_keys = set()
        debug = bool(kwargs.pop("debug", False))
        if moe_block_names is not None and moe_names is not None:
            state_dict, consolidated_moe_keys = StateDictConverter.apply_moe_transformations(
                state_dict=state_dict,
                moe_names=moe_names,
                moe_path=moe_path,
                moe_block_names=moe_block_names,
                moe_block_path=moe_block_path,
                reform_param=reform_param,
                debug=debug,
            )

        fused_counts = StateDictConverter.apply_reform_param_fusions(state_dict, reform_param)
        for label, count in sorted(fused_counts.items()):
            logger.info("Fused %d reform_param %s into runtime merged weights.", count, label)

        return StateDictConverter._base_huggingface_to_easydel(
            state_dict,
            device=device,
            embedding_layer_names=embedding_layer_names,
            layernorm_names=layernorm_names,
            moe_names=moe_names,
            moe_path=moe_path,
            moe_block_names=moe_block_names,
            moe_block_path=moe_block_path,
            shard_fns=shard_fns,
            dtype=dtype,
            verbose=verbose,
            callback=callback,
            remove_state_dict=remove_state_dict,
            lm_head_name=lm_head_name,
            uses_tie_word_embedding=uses_tie_word_embedding,
            consolidated_moe_keys=consolidated_moe_keys,
            reform_param=reform_param,
            **kwargs,
        )

    @staticmethod
    def apply_moe_transformations_reverse(
        state_dict: dict[str, tp.Any],
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        tensor_transform: tp.Callable | None = None,
    ) -> dict[str, tp.Any]:
        """
        Transform MoE weights from EasyDel format (stacked experts) to HuggingFace format (separate experts).

        Converts from:
            model.layers.3.block_sparse_moe.experts.w3.weight -> shape (num_experts, 128, 256)
        To:
            model.layers.3.block_sparse_moe.experts.0.w3.weight -> shape (128, 256)
            model.layers.3.block_sparse_moe.experts.1.w3.weight -> shape (128, 256)
            ...

        Args:
            state_dict: PyTorch-style state dict containing stacked-MoE tensors.
            moe_block_names: Tail names of MoE block modules.
            moe_names: Tail names of expert sub-modules.
            moe_block_path: Full dotted paths to each MoE block.
            moe_path: Full dotted paths to expert modules; used to detect the
                expert-container name.
            tensor_transform: Optional callable applied to each per-expert
                tensor before it is written to the output dict.

        Returns:
            A new dict with each stacked tensor split into one ``...{i}.{name}.weight``
            entry per expert.

        Raises:
            ValueError: If ``moe_names`` or ``moe_block_path`` is ``None``.
        """
        if not all([moe_block_names, moe_names, moe_block_path]):
            return state_dict

        if moe_names is None:
            raise ValueError("moe_names cannot be None")
        if moe_block_path is None:
            raise ValueError("moe_block_path cannot be None")

        new_state_dict = {}
        processed_keys = set()
        expected_expert_name = moe_path[0].split(".")[-2] if moe_path else "experts"
        sibling_expert_parents = {
            path.rsplit(".", 1)[0] for path in moe_path or [] if f".{expected_expert_name}." in path
        }

        for key, value in state_dict.items():
            is_stacked_moe = False
            for block_path in moe_block_path:
                if key.startswith(block_path):
                    remainder = key[len(block_path) + 1 :]
                    parts = remainder.split(".")
                    if (
                        len(parts) == 3
                        and parts[0] == expected_expert_name
                        and parts[1] in moe_names
                        and parts[2] == "weight"
                    ):
                        is_stacked_moe = True
                        moe_name = parts[1]
                        if hasattr(value, "shape") and len(value.shape) >= 3:
                            num_experts = value.shape[0]

                            for expert_idx in range(num_experts):
                                expert_tensor = value[expert_idx]
                                if tensor_transform is not None:
                                    expert_tensor = tensor_transform(expert_tensor)
                                new_key = f"{block_path}.{expected_expert_name}.{expert_idx}.{moe_name}.weight"
                                new_state_dict[new_key] = expert_tensor

                            processed_keys.add(key)
                            break

            if not is_stacked_moe:
                match = re.match(rf"^(.*\.{expected_expert_name})\.([^.]+)\.weight$", key)
                if match:
                    expert_parent, moe_name = match.groups()
                    if expert_parent in sibling_expert_parents and hasattr(value, "shape") and len(value.shape) >= 3:
                        num_experts = value.shape[0]
                        for expert_idx in range(num_experts):
                            expert_tensor = value[expert_idx]
                            if tensor_transform is not None:
                                expert_tensor = tensor_transform(expert_tensor)
                            new_key = f"{expert_parent}.{expert_idx}.{moe_name}.weight"
                            new_state_dict[new_key] = expert_tensor
                        processed_keys.add(key)
                        is_stacked_moe = True

            if not is_stacked_moe:
                new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def reconcile_to_target_shapes(state_dict: dict, target_shapes: dict) -> dict:
        """Fix 2-D weight orientation against the HF target's real shapes (data-driven).

        The generic EasyDeL->torch export transposes every dense 2-D weight assuming the
        JAX ``[in, out]`` convention. Some layers do not follow it (e.g. GPT-2 ``Conv1D``
        already stores ``[out, in]``), and tied heads can be exported in a flipped layout.
        Rather than special-casing each model, this consults the HF model's own parameter
        shapes: for any key whose tensor shape does not match the target but whose
        transpose does, transpose it. Square or already-correct tensors are left untouched.

        Args:
            state_dict: Converted torch state dict (keys already in HF naming).
            target_shapes: ``{key: shape}`` from the instantiated HF model's ``state_dict``.

        Returns:
            The same dict with mis-oriented 2-D weights transposed in place.
        """
        for key, tensor in list(state_dict.items()):
            tgt = target_shapes.get(key)
            if tgt is None or not hasattr(tensor, "shape"):
                continue
            cur = tuple(tensor.shape)
            tgt = tuple(tgt)
            if cur == tgt:
                continue
            # Only the unambiguous 2-D transpose case is safe to auto-correct here: a
            # weight stored as the exact reverse of the target with two DISTINCT dims.
            # (Higher-rank / repeated-dim permutations are ambiguous — which axis maps to
            # which can't be inferred from shape alone — so they are left to the model's
            # own reform_param rules rather than guessed.)
            if tensor.ndim == 2 and cur == tgt[::-1] and cur[0] != cur[1]:
                state_dict[key] = tensor.transpose(0, 1).contiguous()
        return state_dict

    @staticmethod
    def easydel_to_torch(
        module: EasyDeLBaseModule, dtype: jnp.dtype | None = jnp.float16, **kwargs
    ) -> dict[str, tp.Any]:
        """Convert an EasyDeL module's parameters to a PyTorch state dict.

        Flattens the module's parameter tree, transposes weight axes back
        to PyTorch conventions, renames keys (``.kernel`` -> ``.weight``,
        ``.embedding`` -> ``.weight``, ``.scale`` -> ``.weight``), and
        un-stacks MoE expert weights if present.

        Args:
            module: EasyDeL module whose parameters will be exported.
            dtype: Target dtype for the exported tensors.

        Returns:
            Dictionary mapping PyTorch-style parameter names to tensors.
        """
        if dtype is None:
            dtype = module.param_dtype

        # ``module.parameters`` is a spectrax ``State`` (post-migration). Its ``flatten()``
        # yields ``{"<collection>/<dotted.path>": array}``; strip the leading collection
        # ("parameters/") to get the ``{dotted.path: array}`` the converter expects.
        flat_state = module.parameters.flatten()
        model_parameters = {key.split("/", 1)[-1]: value for key, value in flat_state.items()}

        from easydel.layers import BaseMoeModule, Embed, ParallelMoELinear
        from easydel.utils import traversals

        md = ParallelMoELinear
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, md)]
        md = BaseMoeModule
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, md)]

        # spectrax names every leaf ".weight" (no kernel/embedding/scale distinction), so the
        # export transpose must detect dense weights by module type, exactly like the load path:
        # transpose 2D+ weights EXCEPT embeddings (and layernorm scales, which are 1D and skip the
        # permute branches anyway). This feeds correctly-oriented tensors to the qkv/gate-up split.
        embedding_names = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, Embed)]
        try:
            from flax import nnx as _nnx

            embedding_names += [
                ".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, _nnx.Embed)
            ]
        except Exception:
            pass

        moe_names = list(set([names.split(".")[-1] for names in moe_path])) if moe_path else None
        moe_block_names = list(set([names.split(".")[-1] for names in moe_block_path])) if moe_block_path else None

        stacked_moe_keys = set()
        if moe_block_names and moe_names and moe_block_path:
            for block_path in moe_block_path:
                for moe_name in moe_names:
                    # spectrax leaves are ".weight" (was ".kernel"); match that so stacked
                    # expert tensors get the expert-specific permute (0, 2, 1), not the dense one.
                    potential_key = f"{block_path}.experts.{moe_name}.weight"
                    if potential_key in model_parameters:
                        stacked_moe_keys.add(potential_key)
        torch_state_dict = {}
        with tqdm(model_parameters.items(), desc=f"Converting {module.__class__.__name__} to torch") as pbar:
            for key, tensor in pbar:
                if tensor is None:
                    continue
                if hasattr(tensor, "materialize"):
                    tensor = tensor.materialize()
                if hasattr(tensor, "value") and hasattr(tensor.value, "materialize"):
                    tensor = tensor.value.materialize()
                if tensor.dtype != DtypeHandler.get_dtype(dtype):
                    tensor = tensor.astype(DtypeHandler.get_dtype(dtype))
                tensor = TensorConverter.jax_to_pytorch(jax.block_until_ready(tensor))
                is_stacked_moe = key in stacked_moe_keys
                is_embedding = any(emb in key for emb in embedding_names)

                # Transpose dense weights JAX [in, out] -> torch [out, in] (higher-rank
                # analogues). Skip embeddings (kept [vocab, hidden]); 1D layernorm scales hit
                # no permute branch. Keyed on module type, not the (now uniform) ".weight" name.
                if not is_embedding:
                    if not is_stacked_moe:
                        if tensor.ndim == 2:
                            tensor = tensor.permute(1, 0)
                        elif tensor.ndim == 3:
                            tensor = tensor.permute(2, 1, 0)
                        elif tensor.ndim == 4:
                            tensor = tensor.permute(3, 2, 0, 1)
                        elif tensor.ndim == 5:
                            tensor = tensor.permute(4, 3, 0, 1, 2)
                        elif tensor.ndim == 6:
                            tensor = tensor.permute(5, 4, 3, 2, 0, 1)
                    else:
                        if tensor.ndim == 3:
                            tensor = tensor.permute(0, 2, 1)

                key = key.replace(".kernel", ".weight").replace(".embedding", ".weight").replace(".scale", ".weight")
                torch_state_dict[key] = tensor

        if moe_block_names and moe_names and moe_block_path and moe_path:
            torch_state_dict = StateDictConverter.apply_moe_transformations_reverse(
                state_dict=torch_state_dict,
                moe_names=moe_names,
                moe_path=moe_path,
                moe_block_names=moe_block_names,
                moe_block_path=moe_block_path,
            )

        reform_param = kwargs.get("reform_param", None)
        if reform_param:
            StateDictConverter.validate_reform_param_schema(reform_param)
            for key_check, value_check in reform_param.items():
                if "splits" not in value_check:
                    continue
                inverse_spliter = value_check.get("inverse_spliter", None)
                if inverse_spliter:
                    anchor_to_end = key_check.endswith("$")
                    match_target = key_check[:-1] if anchor_to_end else key_check
                    candidates = {}  # (prefix, suffix) -> {split_name: tensor}

                    splits = value_check["splits"]
                    split_names = [s["name"] for s in splits]

                    keys_to_remove = []

                    for key in torch_state_dict.keys():
                        for split_name in split_names:
                            match_index = key.find(split_name)
                            if match_index != -1:
                                after_match = key[match_index + len(split_name) :]
                                if anchor_to_end and after_match:
                                    continue
                                if not after_match or after_match.startswith("."):
                                    before_match = key[:match_index]
                                    if not before_match or before_match.endswith("."):
                                        original_key_candidate = f"{before_match}{match_target}{after_match}"
                                        if original_key_candidate.replace(match_target, split_name) == key:
                                            prefix = before_match
                                            suffix = after_match

                                            group_key = (prefix, suffix)
                                            if group_key not in candidates:
                                                candidates[group_key] = {}

                                            candidates[group_key][split_name] = key

                    for (prefix, suffix), found_splits in candidates.items():
                        if len(found_splits) == len(split_names):
                            tensors_to_merge = []
                            for split in splits:
                                split_name = split["name"]
                                key = found_splits[split_name]
                                tensors_to_merge.append(torch_state_dict[key])
                                keys_to_remove.append(key)

                            torch_module = TensorConverter.get_torch()

                            positional_params = [
                                p
                                for p in inspect.signature(inverse_spliter).parameters.values()
                                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
                            ]
                            wants_torch = bool(positional_params) and positional_params[0].name == "torch"
                            if wants_torch:
                                merged_tensor = inverse_spliter(torch_module, *tensors_to_merge)
                            else:
                                merged_tensor = inverse_spliter(*tensors_to_merge)
                            original_key = f"{prefix}{match_target}{suffix}"
                            torch_state_dict[original_key] = merged_tensor
                            if original_key in keys_to_remove:
                                keys_to_remove.remove(original_key)

                    for key in keys_to_remove:
                        del torch_state_dict[key]

            for key_check, value_check in reform_param.items():
                if "sources" not in value_check or "inverse_fuser" not in value_check:
                    continue
                anchor_to_end = key_check.endswith("$")
                match_target = key_check[:-1] if anchor_to_end else key_check
                source_names = tuple(value_check["sources"])
                if not source_names:
                    continue

                additions: dict[str, tp.Any] = {}
                keys_to_remove: set[str] = set()
                for key, tensor in list(torch_state_dict.items()):
                    match_index = key.find(match_target)
                    if match_index == -1:
                        continue
                    after_match = key[match_index + len(match_target) :]
                    if anchor_to_end and after_match:
                        continue
                    if after_match and not after_match.startswith("."):
                        continue
                    before_match = key[:match_index]
                    if before_match and not before_match.endswith("."):
                        continue

                    outputs = StateDictConverter.inverse_fuse_reform_param_tensor(value_check, tensor)
                    if isinstance(outputs, Mapping):
                        source_tensors = {str(name): value for name, value in outputs.items()}
                    else:
                        if not isinstance(outputs, tuple | list):
                            outputs = (outputs,)
                        if len(outputs) != len(source_names):
                            raise ValueError(
                                f"inverse_fuser for {match_target} returned {len(outputs)} tensors, "
                                f"expected {len(source_names)}."
                            )
                        source_tensors = dict(zip(source_names, outputs, strict=True))

                    for source_name, source_tensor in source_tensors.items():
                        additions[f"{before_match}{source_name}{after_match}"] = source_tensor
                    keys_to_remove.add(key)

                for key in keys_to_remove:
                    torch_state_dict.pop(key, None)
                torch_state_dict.update(additions)

        return torch_state_dict


class ModelConverter:
    """High-level orchestrator for two-way EasyDeL ↔ HuggingFace model conversion.

    Where :class:`StateDictConverter` operates on raw parameter mappings,
    :class:`ModelConverter` brings model classes and configs into the
    picture: it instantiates a fresh ``transformers`` ``PreTrainedModel``
    from the equivalent EasyDeL config, feeds it the converted state dict,
    and validates parameter shape parity along the way. Used by the public
    ``module.to_torch()`` / ``module.from_pretrained(... save_torch=True)``
    paths.

    The class exposes only ``@staticmethod`` callables — there is no
    instance state. Memory-conscious conversions can be run under
    ``torch.device("meta")`` (the default) so the HuggingFace skeleton is
    materialised lazily.
    """

    @staticmethod
    def easydel_to_huggingface(
        module: EasyDeLBaseModule,
        config: EasyDeLBaseConfig,
        base_huggingface_module: PreTrainedModel,
        base_huggingface_module_kwarguments: dict | None = None,
        dtype: jnp.dtype = jnp.float16,
        use_meta_torch: bool = True,
        reform_param: dict | None = None,
        **kw,
    ) -> tp.Any:
        """Convert an EasyDeL module to a HuggingFace ``PreTrainedModel``.

        Creates a HuggingFace model instance, converts the EasyDeL
        parameters to a PyTorch state dict via ``easydel_to_torch``,
        and loads the weights into the HuggingFace model.

        Args:
            module: Source EasyDeL module.
            config: EasyDeL configuration to derive the HuggingFace config.
            base_huggingface_module: HuggingFace model class to instantiate.
            base_huggingface_module_kwarguments: Extra kwargs for the HF
                model constructor.
            dtype: Target dtype for the conversion.
            use_meta_torch: Whether to use ``torch.device("meta")`` for
                memory-efficient model construction.
            reform_param: Optional parameter splitting/merging rules.

        Returns:
            Instantiated HuggingFace model with loaded weights.
        """

        import torch

        if base_huggingface_module_kwarguments is None:
            base_huggingface_module_kwarguments = {}

        state_dict = StateDictConverter.easydel_to_torch(module=module, dtype=dtype, reform_param=reform_param)
        base_config = base_huggingface_module.config_class.from_dict(config.to_dict())
        with torch.device("meta") if use_meta_torch else contextlib.nullcontext():
            model: torch.nn.Module = base_huggingface_module(config=base_config, **base_huggingface_module_kwarguments)
            target_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items() if hasattr(v, "shape")}
            # Reconcile each converted tensor against the HF target's actual shape, driven
            # by data (not model-specific rules): when a 2-D weight's orientation is the
            # transpose of what HF expects, flip it. This self-corrects layers whose JAX vs
            # torch storage convention matches (e.g. GPT-2 Conv1D, already [out, in]) where
            # the generic export transpose would otherwise mis-orient them, and any tied
            # head whose export layout differs — no per-model casing required.
            state_dict = StateDictConverter.reconcile_to_target_shapes(state_dict, target_shapes)
            if len(target_shapes) != len(state_dict):
                warnings.warn(
                    f"converted state_dict has {len(state_dict)} keys, HF model expects {len(target_shapes)}.",
                    stacklevel=1,
                )
            for key, shape in target_shapes.items():
                if key in state_dict and tuple(state_dict[key].shape) != shape:
                    warnings.warn(
                        f"Shape conflict at {key}: have {tuple(state_dict[key].shape)}, expected {shape}.",
                        stacklevel=1,
                    )
            model.load_state_dict(state_dict, assign=True, strict=True)

        return model
