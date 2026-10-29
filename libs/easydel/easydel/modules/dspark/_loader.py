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

"""Family-agnostic loader for DSpark draft models from native ``mtp.*`` safetensors shards (DeepSeek-V4 / V4.1).

Composes the existing conversion pieces instead of a new path:

- keys: a family ``key_normalizer`` (``mtp.N.*`` -> ``stages.N.*``; unowned keys map to ``None``);
- pre-quantized weights: the bridge's checkpoint quant decoder (block-fp8 dense, MXFP4 routed experts via the
  ``deepseek_v4_mixed`` adapter), decoded to dense floats;
- layout: :meth:`StateDictConverter.huggingface_to_easydel` with the draft module's own MoE paths, embedding names and
  ``reform_param`` rules (expert stacking, 2-D transposes, shared-expert ``gate_up`` fusion interleaved for the
  config's tensor-parallel size at conversion time);
- placement: every leaf's ``NamedSharding`` is resolved from the variable metadata of an abstract (``eval_shape``)
  module, so no random-initialized copy of the draft model is ever materialized on device.

Shards are processed one draft stage at a time, so host memory holds one stage's dense tensors at most. With
``stage_shards`` the shard files themselves are staged per stage too (only one stage's files on disk at a time).
Decoding and conversion run with the host CPU as JAX's default device, so only the final per-leaf placement touches
accelerators.
"""

from __future__ import annotations

import collections
import contextlib
import typing as tp
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import spectrax as spx

from easydel.utils.helpers import get_logger

logger = get_logger(__name__)

KeyNormalizer = tp.Callable[[str], "str | None"]
#: ``stage -> context manager yielding that stage's shard paths`` (entered while the stage loads, then exited).
StageShards = tp.Callable[[int], tp.ContextManager[tp.Sequence[str]]]


def scan_mtp_keys(shard_paths: tp.Sequence[str]) -> dict[int, list[tuple[str, str]]]:
    """Group the ``mtp.N.*`` keys of the given safetensors files by draft stage: ``{N: [(path, key), ...]}``."""
    from safetensors import safe_open

    stages: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
    for path in shard_paths:
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                if key.startswith("mtp."):
                    stages[int(key.split(".")[1])].append((str(path), key))
    return dict(stages)


def load_dspark_model(
    config: tp.Any,
    shard_paths: tp.Sequence[str] | None,
    *,
    build: tp.Callable[[], tp.Any],
    key_normalizer: KeyNormalizer,
    quantization_config: tp.Mapping[str, tp.Any] | None = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    stage_shards: StageShards | None = None,
    scan_keys: tp.Callable[[tp.Sequence[str]], dict[int, list[tuple[str, str]]]] | None = None,
) -> tp.Any:
    """Build a DSpark draft model on ``config.mesh`` holding the weights of native ``mtp.*`` shards.

    Args:
        config: The backbone config (its mesh and sharding settings are used for placement).
        shard_paths: Safetensors files containing the ``mtp.*`` tensors (other keys are ignored); ``None`` when
            ``stage_shards`` provides them.
        build: Zero-argument constructor of the draft model (called under ``eval_shape``); the model exposes
            ``num_stages``.
        key_normalizer: Native ``mtp.N.*`` key -> draft-model parameter name, or ``None`` for unowned keys.
        quantization_config: The checkpoint's ``quantization_config`` (plus hints such as ``expert_dtype``); ``None``
            for an unquantized checkpoint.
        param_dtype: Parameter storage dtype (fp8 / fp4 values decode exactly into bf16).
        stage_shards: Optional per-stage shard provider (``stage -> context manager of paths``) used instead of
            ``shard_paths``; each stage's context is exited before the next stage is staged.
        scan_keys: Optional shard scanner ``paths -> {stage: [(path, key), ...]}``; defaults to
            :func:`scan_mtp_keys` (native ``mtp.N.*`` keys). Families whose draft layers live elsewhere in the
            checkpoint (e.g. GLM's ``layers.{num_hidden_layers}.*`` nextn layer) pass their own.

    Returns:
        The draft model with every parameter loaded and placed on its sharding.

    Raises:
        ValueError: If the shards miss a draft stage or a parameter, carry an unexpected parameter, or a shape
            disagrees with the draft model.
    """
    import torch
    from spectrax import nn
    from spectrax.sharding.partition import named_sharding_for_variable

    from easydel.infra.base_module import EasyDeLBaseModule, _parameter_init_sharding_context
    from easydel.layers import BaseMoeModule, Embed, ParallelMoELinear
    from easydel.utils import traversals

    if (shard_paths is None) == (stage_shards is None):
        raise ValueError("Pass exactly one of `shard_paths` and `stage_shards`.")
    scanner = scan_mtp_keys if scan_keys is None else scan_keys
    mesh = config.mesh
    with mesh, _parameter_init_sharding_context(config):
        model = spx.eval_shape(build)
        shardings = {path: named_sharding_for_variable(var, mesh) for path, var in spx.live_variables(model)}
    graphdef, state = spx.export(model)
    targets = {path: (collection, leaf) for collection, path, leaf in state.items()}

    def _paths(cls: type) -> list[str]:
        return [".".join(map(str, p)) for p, _ in traversals.iter_module_search(model, cls)]

    moe_path = _paths(ParallelMoELinear)
    moe_block_path = _paths(BaseMoeModule)
    try:
        host = jax.devices("cpu")[0]
    except RuntimeError:
        host = None
    convert_kwargs = dict(
        embedding_layer_names=_paths(nn.Embed) + _paths(Embed),
        layernorm_names=[],
        moe_names=sorted({p.split(".")[-1] for p in moe_path}),
        moe_block_names=sorted({p.split(".")[-1] for p in moe_block_path}),
        moe_block_path=moe_block_path,
        moe_path=moe_path,
        dtype=param_dtype,
        reform_param=EasyDeLBaseModule._get_reform_param(model),
        device=host,
        verbose=False,
    )
    torch_dtype = torch.bfloat16 if jnp.dtype(param_dtype) == jnp.bfloat16 else None
    expected_stages = list(range(int(model.num_stages)))

    @contextlib.contextmanager
    def _stage_keys(stage: int) -> tp.Iterator[list[tuple[str, str]]]:
        if stage_shards is None:
            yield all_keys.get(stage, [])
            return
        with stage_shards(stage) as paths:
            yield scanner(list(paths)).get(stage, [])

    all_keys: dict[int, list[tuple[str, str]]] = {}
    if shard_paths is not None:
        all_keys = scanner(shard_paths)
        if sorted(all_keys) != expected_stages:
            raise ValueError(f"mtp shards hold stages {sorted(all_keys)}; the draft model needs {expected_stages}.")

    placed: dict[str, jax.Array] = {}
    unexpected: list[str] = []
    # Decode / convert on the host: otherwise jnp work in the quant decoder lands on the default device (TPU:0).
    host_scope = jax.default_device(host) if host is not None else contextlib.nullcontext()
    with host_scope:
        for stage in expected_stages:
            with _stage_keys(stage) as keys:
                if not keys:
                    raise ValueError(f"no mtp.{stage}.* tensors found for draft stage {stage}.")
                _load_stage(
                    stage,
                    keys,
                    key_normalizer=key_normalizer,
                    quantization_config=quantization_config,
                    convert_kwargs=convert_kwargs,
                    torch_dtype=torch_dtype,
                    targets=targets,
                    shardings=shardings,
                    placed=placed,
                    unexpected=unexpected,
                )

    missing = sorted(set(targets) - set(placed))
    if missing or unexpected:
        raise ValueError(
            f"DSpark load mismatch: missing {missing[:8]} (of {len(missing)}), unexpected {unexpected[:8]}."
        )
    for path, array in placed.items():
        state.set(targets[path][0], path, array)
    return spx.bind(graphdef, state)


def _load_stage(
    stage: int,
    keys: list[tuple[str, str]],
    *,
    key_normalizer: KeyNormalizer,
    quantization_config: tp.Mapping[str, tp.Any] | None,
    convert_kwargs: dict[str, tp.Any],
    torch_dtype: tp.Any,
    targets: dict[str, tuple[tp.Any, tp.Any]],
    shardings: dict[str, tp.Any],
    placed: dict[str, jax.Array],
    unexpected: list[str],
) -> None:
    """Decode, convert and place one draft stage (fills ``placed`` / ``unexpected``)."""
    import torch
    from safetensors import safe_open

    from easydel.infra.mixins.bridge import _QUANT_CONSUMED, _build_quant_decoder
    from easydel.utils import traversals
    from easydel.utils.parameters_transformation import StateDictConverter

    # Native key -> file. The quant decoder runs on NATIVE keys (as in the bridge): after renaming,
    # ``mtp.N.hc_attn_scale`` becomes ``stages.N.attn_hc.scale`` and would be mistaken for a quant scale.
    files = {key: path for path, key in keys}
    decoder = (
        _build_quant_decoder(SimpleNamespace(quantization_config=dict(quantization_config)), files)
        if quantization_config is not None
        else None
    )
    handles = {path: safe_open(path, "pt") for path in set(files.values())}

    def fetch(key: str):
        path = files.get(key)
        return None if path is None else handles[path].get_tensor(key)

    state_dict: dict[str, tp.Any] = {}
    loaded = 0
    for key in sorted(files):
        name = key_normalizer(key)
        if name is None:
            continue
        tensor = fetch(key)
        if decoder is not None:
            decoded = decoder(key, tensor, fetch)
            if decoded is _QUANT_CONSUMED:
                continue
            if decoded is not None:
                tensor = decoded
        loaded += 1
        # Decoded fp8/fp4 values are exact in bf16; narrowing here bounds the per-stage host footprint.
        if torch_dtype is not None and tensor.dtype == torch.float32 and tensor.ndim >= 2:
            tensor = tensor.to(torch_dtype)
        state_dict[name] = tensor
    handles.clear()  # release the file handles before conversion (``fetch`` closes over this dict)

    converted = StateDictConverter.huggingface_to_easydel(state_dict, **convert_kwargs)
    del state_dict
    for key_tuple, array in traversals.flatten_dict(converted).items():
        path = ".".join(map(str, key_tuple))
        if path not in targets:
            unexpected.append(path)
            continue
        leaf = targets[path][1]
        if tuple(array.shape) != tuple(leaf.shape):
            raise ValueError(f"{path}: converted shape {tuple(array.shape)} != draft model {tuple(leaf.shape)}")
        placed[path] = jax.device_put(jnp.asarray(array, dtype=leaf.dtype), shardings[path])
    del converted
    logger.info(f"DSpark stage {stage}: {loaded} checkpoint tensors loaded.")


__all__ = ["KeyNormalizer", "StageShards", "load_dspark_model", "scan_mtp_keys"]
