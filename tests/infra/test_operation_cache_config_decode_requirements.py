from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import typing
from pathlib import Path
from types import SimpleNamespace


def _load_symbols():
    """Import OperationCacheMixin in-place or via a lightweight stubbed loader.

    The full EasyDeL package has optional dependencies; this test focuses only
    on the config-based cache inference logic and can run without them.
    """

    try:
        from easydel.infra.mixins.operation_cache import OperationCacheMixin
        from easydel.operations.requirements import CacheType, MetadataField

        return OperationCacheMixin, CacheType, MetadataField
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        easydel_dir = repo_root / "easydel"

        def _ensure_pkg(name: str, path: Path) -> types.ModuleType:
            module = sys.modules.get(name)
            if module is None:
                module = types.ModuleType(name)
                sys.modules[name] = module
            module.__path__ = [str(path)]
            return module

        _ensure_pkg("easydel", easydel_dir)
        _ensure_pkg("easydel.layers", easydel_dir / "layers")
        ops_pkg = _ensure_pkg("easydel.layers.operations", easydel_dir / "layers" / "operations")
        _ensure_pkg("easydel.infra", easydel_dir / "infra")
        _ensure_pkg("easydel.infra.mixins", easydel_dir / "infra" / "mixins")

        reqs_mod = importlib.import_module("easydel.layers.operations.requirements")
        CacheType = reqs_mod.CacheType
        ExecutionMode = reqs_mod.ExecutionMode
        MetadataField = reqs_mod.MetadataField
        OperationRequirements = reqs_mod.OperationRequirements

        class _VanillaOp:
            @classmethod
            def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):  # type:ignore
                return OperationRequirements.create(
                    name="vanilla",
                    required_metadata=MetadataField.basic(),
                    supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
                )

        class _AutoRegressiveDecodeOp:
            @classmethod
            def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):  # type:ignore
                return OperationRequirements.create(
                    name="autoregressive_decodeattn",
                    required_metadata=MetadataField.basic() | MetadataField.CONTEXT_LENS,
                    supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
                )

        class _RaggedPageV2Op:
            @classmethod
            def get_requirements(cls, mode: ExecutionMode = ExecutionMode.MIXED):  # type:ignore
                return OperationRequirements.create(
                    name="ragged_page_attention_v2",
                    required_metadata=MetadataField.ragged(),
                    supported_cache=CacheType.RAGGED_PAGES,
                )

        class _OperationRegistry:
            _ops: typing.ClassVar = {
                "vanilla": _VanillaOp,
                "autoregressive_decodeattn": _AutoRegressiveDecodeOp,
                "ragged_page_attention_v2": _RaggedPageV2Op,
            }

            @classmethod
            def get(cls, name: str):
                return cls._ops.get(name.lower())

        ops_pkg.OperationRegistry = _OperationRegistry

        operation_cache_path = easydel_dir / "infra" / "mixins" / "operation_cache.py"
        spec = importlib.util.spec_from_file_location(
            "easydel.infra.mixins.operation_cache",
            operation_cache_path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module.OperationCacheMixin, CacheType, MetadataField


OperationCacheMixin, CacheType, MetadataField = _load_symbols()


class _Dummy(OperationCacheMixin):
    def __init__(self, config):
        self.config = config


def test_layer_types_merges_decode_metadata_requirements():
    model = _Dummy(
        SimpleNamespace(
            attn_mechanism="vanilla",
            decode_attn_mechanism="autoregressive_decodeattn",
            layer_types=["full_attention", "full_attention"],
            num_hidden_layers=2,
        )
    )

    info = model._get_operations_cache_info_from_config()

    assert MetadataField.CONTEXT_LENS in info.combined_metadata


def test_layer_types_intersects_decode_cache_requirements():
    model = _Dummy(
        SimpleNamespace(
            attn_mechanism="ragged_page_attention_v2",
            decode_attn_mechanism="vanilla",
            layer_types=["full_attention"],
            num_hidden_layers=1,
        )
    )

    info = model._get_operations_cache_info_from_config()

    assert info.combined_cache_types == CacheType.NONE
