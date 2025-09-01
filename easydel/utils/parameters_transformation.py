# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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


from __future__ import annotations

import contextlib
import functools
import gc
import os
import typing as tp
import warnings

import jax
import jax.extend
import numpy as np
from jax import dlpack
from jax import numpy as jnp
from tqdm.autonotebook import tqdm

from easydel.utils.helpers import check_bool_flag, get_logger

from .analyze_memory import SMPMemoryMonitor
from .traversals import flatten_dict, unflatten_dict

if tp.TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.infra.base_module import EasyDeLBaseModule


mem_ops = SMPMemoryMonitor(5)
logger = get_logger(__name__)
EASYDEL_PERFRED_HOST_COPY_INDEX = int(os.getenv("EASYDEL_PERFRED_HOST_COPY_INDEX", "0"))
EASYDEL_PERFRED_HOST_COPY = str(os.getenv("EASYDEL_PERFRED_HOST_COPY", "cpu")).lower()
EASYDEL_PERFRED_HOST_COPY = None if EASYDEL_PERFRED_HOST_COPY == "none" else EASYDEL_PERFRED_HOST_COPY


class DtypeHandler:
    """Handles dtype conversions and operations."""

    @staticmethod
    def get_dtype(dtype: str | jnp.dtype) -> jnp.dtype:
        """Convert string dtype representation to JAX dtype."""
        if isinstance(dtype, str):
            dtype_map = {
                "bf16": jnp.bfloat16,
                "bfloat16": jnp.bfloat16,
                "fp16": jnp.float16,
                "float16": jnp.float16,
                "fp32": jnp.float32,
                "float32": jnp.float32,
                "fp64": jnp.float64,
                "float64": jnp.float64,
                "fp8": jnp.float8_e5m2,
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
            dtype = dtype_map[dtype]
        return dtype

    @staticmethod
    def float_tensor_to_dtype(tensor: tp.Any, dtype: str | jnp.dtype | None) -> tp.Any:
        """Convert float tensor to specified dtype."""
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
    """Handles tensor conversions between PyTorch and JAX."""

    @staticmethod
    def convert_pytorch_to_jax(tensor: tp.Any, dtype: jnp.dtype) -> jnp.ndarray:
        """Convert PyTorch tensor to JAX array."""
        if "bfloat16" in str(tensor.dtype):
            tensor = tensor.float()
        return jnp.asarray(tensor.cpu().detach().numpy(), dtype=dtype)

    @staticmethod
    @functools.lru_cache
    def get_torch():
        """Import and return torch module (cached)."""
        import torch

        return torch

    @staticmethod
    def jax_to_pytorch(x: jax.Array) -> tp.Any:
        """Convert JAX array to PyTorch tensor."""
        if check_bool_flag("EASY_SAFE_TRANSFER", True):
            x = jax.device_get(x)
            return TensorConverter.get_torch().from_numpy(np.array(x.tolist(), dtype=x.dtype))
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
                        jax.devices(EASYDEL_PERFRED_HOST_COPY)[EASYDEL_PERFRED_HOST_COPY_INDEX],
                    ),
                    stream=None,
                )
            return dlpack_pt.from_dlpack(dl_pack_jax)

    @staticmethod
    def pytorch_to_jax(x: tp.Any) -> jnp.ndarray:
        """Convert PyTorch tensor to JAX array."""
        return jnp.asarray(x.detach().cpu().numpy())


class StateDictConverter:
    """Handles conversion between PyTorch and EasyDeL state dictionaries."""

    @staticmethod
    def match_keywords(string: str, required: list[str], forbidden: list[str]) -> bool:
        """Check if string contains all required keywords and none of the forbidden ones."""
        return all(t in string for t in required) and not any(n in string for n in forbidden)

    @staticmethod
    def process_tensor(key: str, tensor: tp.Any, config: dict[str, tp.Any]) -> tuple[tuple, jnp.ndarray] | None:
        """Process a single tensor and return its processed key and value."""
        new_key = key

        if any(layer_name in key for layer_name in config["embedding_layer_names"]):
            new_key = f"{key[: -len('.weight')]}.embedding"

        elif any(layer_norm in key for layer_norm in config["layernorm_names"]):
            new_key = key.replace(".weight", ".scale")

        elif "weight" in key:
            is_moe_expert = key in config.get("consolidated_moe_keys", set())
            ndim = len(tensor.shape)
            if not is_moe_expert:
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
            else:
                if ndim == 3:
                    tensor = tensor.permute(0, 2, 1)
            new_key = key.replace(".weight", ".kernel")

        key_tuple = tuple(int(n) if n.isdigit() else n for n in new_key.split("."))

        if config["uses_tie_word_embedding"] and config["lm_head_name"] and key_tuple[0] == config["lm_head_name"]:
            return None

        array = TensorConverter.convert_pytorch_to_jax(tensor, config["dtype"])
        return key_tuple, array

    @staticmethod
    def _base_huggingface_to_easydel(
        state_dict: dict[str, tp.Any],
        *,
        device: jax.Device | None = None,  # type:ignore
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        shard_fns: tp.Mapping[tuple, tp.Callable] | None = None,
        dtype: jnp.dtype = jnp.float16,
        verbose: bool = True,
        callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None,
        remove_state_dict: bool = False,
        lm_head_name: str | None = None,
        uses_tie_word_embedding: bool = False,
        consolidated_moe_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Base conversion function from PyTorch state dict to EasyDeL format."""
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
        }

        with jax.default_device(device) if device is not None and shard_fns is None else contextlib.nullcontext():
            flax_dict = {}
            with tqdm(total=len(state_dict), disable=not verbose, desc="Converting Model") as pbar:
                for key, tensor in state_dict.items():
                    try:
                        result = StateDictConverter.process_tensor(key, tensor, config)
                        if result is not None:
                            key_tuple, jax_array = result
                            if shard_fns and key_tuple in shard_fns:
                                jax_array = shard_fns[key_tuple](jax_array)
                            if callback is not None:
                                jax_array = callback(jax_array, key_tuple)
                            flax_dict[key_tuple] = jax_array
                    except Exception as e:
                        logger.error(f"Error processing key {key}: {e!s}")
                    pbar.update(1)

            if remove_state_dict:
                del state_dict
                _clear()

            return unflatten_dict(flax_dict)

    @staticmethod
    def apply_moe_transformations(
        state_dict: dict[str, tp.Any],
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        tensor_transform: tp.Callable | None = None,
    ) -> tuple[dict[str, tp.Any], set[str]]:
        """
        Transform MoE weights from HuggingFace format (separate experts) to EasyDel format (stacked experts).
        Converts from:
            model.layers.3.block_sparse_moe.experts.0.w3.weight -> shape (128, 256)
            model.layers.3.block_sparse_moe.experts.1.w3.weight -> shape (128, 256)
            ...
        To:
            model.layers.3.block_sparse_moe.experts.w3.weight -> shape (num_experts, 128, 256)
        """
        if not all([moe_block_names, moe_names, moe_block_path]):
            return state_dict, set()

        import torch

        excepted_expert_name = moe_path[0].split(".")[-2]
        expert_prefix = f".{excepted_expert_name}."

        moe_names_set = set(moe_names)
        moe_stacked_paths = {
            f"{block_path}.{excepted_expert_name}.{moe_name}" for block_path in moe_block_path for moe_name in moe_names
        }

        new_state_dict = {}
        moe_groups = {path: {} for path in moe_stacked_paths}
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

                    if moe_name in moe_names_set:
                        target_path = f"{block_path}.{excepted_expert_name}.{moe_name}"
                        moe_groups[target_path][expert_idx] = value
                        is_moe_expert = True
                        break

            if not is_moe_expert:
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
                        f"{target_path.replace(f'.{excepted_expert_name}.', f'.{excepted_expert_name}.{idx}.')}.weight"
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
        shard_fns: tp.Mapping[tuple, tp.Callable] | None = None,
        dtype: jnp.dtype = jnp.float16,
        verbose: bool = True,
        callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None,
        remove_state_dict: bool = False,
        lm_head_name: str | None = None,
        uses_tie_word_embedding: bool = False,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Convert PyTorch state dict to EasyDeL format with MoE transformations."""
        consolidated_moe_keys = set()
        if moe_block_names is not None and moe_names is not None:
            state_dict, consolidated_moe_keys = StateDictConverter.apply_moe_transformations(
                state_dict=state_dict,
                moe_names=moe_names,
                moe_path=moe_path,
                moe_block_names=moe_block_names,
                moe_block_path=moe_block_path,
            )

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
        """
        if not all([moe_block_names, moe_names, moe_block_path]):
            return state_dict

        new_state_dict = {}
        processed_keys = set()
        excepted_expert_name = moe_path[0].split(".")[-2] if moe_path else "experts"

        for key, value in state_dict.items():
            is_stacked_moe = False
            for block_path in moe_block_path:
                if key.startswith(block_path):
                    remainder = key[len(block_path) + 1 :]
                    parts = remainder.split(".")
                    if (
                        len(parts) == 3
                        and parts[0] == excepted_expert_name
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
                                new_key = f"{block_path}.{excepted_expert_name}.{expert_idx}.{moe_name}.weight"
                                new_state_dict[new_key] = expert_tensor

                            processed_keys.add(key)
                            break

            if not is_stacked_moe:
                new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def easydel_to_torch(module: EasyDeLBaseModule, dtype: jnp.dtype = jnp.float16) -> dict[str, tp.Any]:
        """Convert EasyDeL module to PyTorch state dict."""
        if dtype is None:
            dtype = module.param_dtype

        graphtree = unflatten_dict(module.parameters)
        model_parameters = flatten_dict(graphtree, sep=".")

        from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
        from easydel.utils import traversals

        md = ParallelMoELinear
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, md)]
        md = BaseMoeModule
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, md)]

        moe_names = list(set([names.split(".")[-1] for names in moe_path])) if moe_path else None
        moe_block_names = list(set([names.split(".")[-1] for names in moe_block_path])) if moe_block_path else None

        stacked_moe_keys = set()
        if moe_block_names and moe_names and moe_block_path:
            for block_path in moe_block_path:
                for moe_name in moe_names:
                    potential_key = f"{block_path}.experts.{moe_name}.kernel"
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

                if key.endswith(".kernel"):
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

        return torch_state_dict


class ModelConverter:
    """Handles model conversions between EasyDeL and HuggingFace formats."""

    @staticmethod
    def easydel_to_huggingface(
        module: EasyDeLBaseModule,
        config: EasyDeLBaseConfig,
        base_huggingface_module: PreTrainedModel,
        base_huggingface_module_kwarguments: dict | None = None,
        dtype: jnp.dtype = jnp.float16,
        use_meta_torch: bool = True,
        **kw,
    ) -> tp.Any:
        """Convert EasyDeL module to HuggingFace model."""

        import torch

        if base_huggingface_module_kwarguments is None:
            base_huggingface_module_kwarguments = {}

        state_dict = StateDictConverter.easydel_to_torch(module=module, dtype=dtype)
        base_config = base_huggingface_module.config_class.from_dict(config.to_dict())
        with torch.device("meta") if use_meta_torch else contextlib.nullcontext():
            model: torch.nn.Module = base_huggingface_module(config=base_config, **base_huggingface_module_kwarguments)
            key_shape_checks = {k: v.shape for k, v in model.state_dict().items() if hasattr(v, "shape")}
            if len(list(key_shape_checks.keys())) != len(list(state_dict.keys())):
                warnings.warn("There might be an issue with converted `state_dict`.", stacklevel=1)
            for key, shape in key_shape_checks.items():
                if state_dict[key].shape != shape:
                    warnings.warn(f"Shape conflict at {key}.", stacklevel=1)
            model.load_state_dict(state_dict, assign=True, strict=True)

        return model
