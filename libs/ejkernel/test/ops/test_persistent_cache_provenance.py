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

"""Provenance guards for the shared on-disk config cache.

The persistent cache file is keyed by ``device_fingerprint|op_id|call_key``,
so every process on a host that calls the same op at the same shapes shares
one entry. A benchmark process that passed explicit configs (or fell back to
heuristics) used to write those unmeasured configs into the shared file, and
a production run with no explicit config would then inherit them instead of
autotuning (the v5p blocksparse 15.5s -> 27-35s step-time incident).

These tests pin the contract at the ops/config-cache layer:
    - explicit-cfg calls never write the persistent cache (in-memory only),
    - heuristic fallbacks never write the persistent cache,
    - autotune-measured winners persist with ``provenance='autotune'`` and are
      reused by fresh caches (a "new process") without re-tuning,
    - legacy entries without a provenance envelope, entries with untrusted
      provenance, and corrupt files are ignored on load and re-tuned.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import pytest

from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Kernel,
    PersistentCache,
    Tuner,
)
from ejkernel.ops.config.persistent import PROVENANCE_AUTOTUNE, PROVENANCE_KEY


@dataclass
class DummyConfig:
    """Minimal JSON-serializable config with an origin-revealing algorithm tag."""

    algorithm: str = "generic"
    block_size: int = 128


class DummyKernel(Kernel[DummyConfig, jax.Array]):
    """XLA/CPU-runnable kernel whose config origin is visible in the output."""

    def __init__(self):
        super().__init__("test_provenance_kernel")

    def run(self, x, y, cfg: DummyConfig) -> jax.Array:
        return x + y + jnp.asarray(cfg.block_size, dtype=x.dtype)

    def heuristic_cfg(self, inv) -> DummyConfig:
        return DummyConfig(algorithm="heuristic", block_size=128)

    def candidate_cfgs(self, inv):
        return [
            DummyConfig(algorithm="tuned", block_size=64),
            DummyConfig(algorithm="tuned", block_size=128),
        ]


def _make_executor(
    path,
    *,
    cache_miss_fallback: str = "autotune",
    allow_autotune: bool = True,
    tuner: Tuner | None = None,
) -> tuple[Executor, ConfigSelectorChain]:
    selector = ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=allow_autotune, cache_miss_fallback=cache_miss_fallback),
        tuner=tuner or Tuner(warmup=0, iters=1),
        persistent=PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig),
    )
    return Executor(selector), selector


class ExplodingTuner(Tuner[DummyConfig]):
    """Fails the test if any autotune measurement is attempted."""

    def autotune(self, make_fn, args, kwargs, candidates):
        raise AssertionError("autotune ran, but a trusted persistent entry should have been reused")


def test_explicit_cfg_call_does_not_write_persistent_cache(tmp_path):
    path = tmp_path / "test-provenance-op.json"
    executor, selector = _make_executor(path, cache_miss_fallback="autotune")
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.arange(4, dtype=jnp.float32)

    out = executor(DummyKernel(), x, y, cfg=DummyConfig(algorithm="explicit", block_size=512))

    assert jnp.allclose(out, x + y + 512.0)
    # nothing was persisted for other processes to inherit
    assert not path.exists()
    # in-memory per-process caching of the explicit config is still allowed
    assert selector.cache.size() == 1
    (cached,) = [selector.cache.get(*key) for key in selector.cache.keys()]
    assert cached == DummyConfig(algorithm="explicit", block_size=512)


def test_heuristic_fallbacks_do_not_write_persistent_cache(tmp_path):
    path = tmp_path / "test-provenance-op.json"
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.arange(4, dtype=jnp.float32)

    # Executor heuristics-only route (policy.cache_miss_fallback == "heuristics"):
    # both the explicit-override branch and the heuristic-fallback branch.
    executor, _ = _make_executor(path, cache_miss_fallback="heuristics")
    out = executor(DummyKernel(), x, y, cfg=DummyConfig(algorithm="explicit", block_size=512))
    assert jnp.allclose(out, x + y + 512.0)
    out = executor(DummyKernel(), jnp.arange(8, dtype=jnp.float32), jnp.arange(8, dtype=jnp.float32))
    assert jnp.allclose(out, jnp.arange(8, dtype=jnp.float32) * 2 + 128.0)

    # ConfigSelectorChain.choose heuristics tier (autotune disallowed).
    executor, _ = _make_executor(path, cache_miss_fallback="autotune", allow_autotune=False)
    out = executor(DummyKernel(), x, y)
    assert jnp.allclose(out, x + y + 128.0)

    assert not path.exists()


def test_autotuned_config_persists_with_provenance_and_is_reused(tmp_path):
    path = tmp_path / "test-provenance-op.json"
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.arange(4, dtype=jnp.float32)

    executor, _ = _make_executor(path, cache_miss_fallback="autotune")
    kernel = DummyKernel()
    out = executor(kernel, x, y)

    data = json.loads(path.read_text())
    assert len(data) == 1
    ((key, entry),) = data.items()
    # incident-shaped key: "<device fingerprint>|<op>@v<version>|<hash>"
    # (the device fingerprint may itself contain "|", e.g. "cpu|cpu")
    assert f"|{kernel.op_id}@v{kernel.version}|" in key
    # provenance envelope with the measured winner inside
    assert entry[PROVENANCE_KEY] == PROVENANCE_AUTOTUNE
    assert entry["cfg"]["algorithm"] == "tuned"
    assert entry["cfg"]["block_size"] in (64, 128)
    assert jnp.allclose(out, x + y + entry["cfg"]["block_size"])

    # a "new process": fresh in-memory cache and a freshly loaded persistent
    # file must reuse the entry without any re-tuning
    executor2, _ = _make_executor(path, cache_miss_fallback="autotune", tuner=ExplodingTuner())
    cfg2 = executor2.choose_config(DummyKernel(), x, y)
    assert isinstance(cfg2, DummyConfig)
    assert asdict(cfg2) == entry["cfg"]
    out2 = executor2(DummyKernel(), x, y)
    assert jnp.allclose(out2, out)


def test_legacy_provenance_less_entry_is_ignored_and_retuned(tmp_path):
    path = tmp_path / "test-provenance-op.json"
    x = jnp.arange(4, dtype=jnp.float32)
    y = jnp.arange(4, dtype=jnp.float32)

    # learn the exact key a real run uses, then rewrite the file in the legacy
    # format (raw config dict, no provenance envelope) with a polluted config
    executor, _ = _make_executor(path, cache_miss_fallback="autotune")
    executor(DummyKernel(), x, y)
    ((key, _),) = json.loads(path.read_text()).items()
    path.write_text(json.dumps({key: {"algorithm": "polluted", "block_size": 999}}))

    executor2, _ = _make_executor(path, cache_miss_fallback="autotune")
    cfg = executor2.choose_config(DummyKernel(), x, y)
    assert cfg.algorithm == "tuned"
    assert cfg.block_size != 999

    # re-tuning overwrote the polluted key with a provenance-carrying entry
    data = json.loads(path.read_text())
    assert data[key][PROVENANCE_KEY] == PROVENANCE_AUTOTUNE
    assert data[key]["cfg"]["algorithm"] == "tuned"


def test_untrusted_provenance_entry_is_ignored_on_get(tmp_path):
    path = tmp_path / "test-provenance-op.json"
    cache = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)

    cache.put("dev", "op@v0", "key", DummyConfig(algorithm="explicit", block_size=512), provenance="override")
    assert cache.get("dev", "op@v0", "key") is None

    cache.put("dev", "op@v0", "key2", DummyConfig(algorithm="tuned", block_size=64), provenance=PROVENANCE_AUTOTUNE)
    assert cache.get("dev", "op@v0", "key2") == DummyConfig(algorithm="tuned", block_size=64)

    # both survive a reload; only the trusted one is served
    reloaded = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)
    assert reloaded.get("dev", "op@v0", "key") is None
    assert reloaded.get("dev", "op@v0", "key2") == DummyConfig(algorithm="tuned", block_size=64)


def test_corrupt_or_non_dict_cache_file_degrades_to_miss(tmp_path):
    path = tmp_path / "test-provenance-op.json"

    path.write_text("{this is not json")
    cache = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)
    assert cache.get("dev", "op@v0", "key") is None
    # writes still work after a corrupt load and round-trip cleanly
    cache.put("dev", "op@v0", "key", DummyConfig(algorithm="tuned", block_size=64), provenance=PROVENANCE_AUTOTUNE)
    reloaded = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)
    assert reloaded.get("dev", "op@v0", "key") == DummyConfig(algorithm="tuned", block_size=64)

    path.write_text(json.dumps(["not", "a", "mapping"]))
    cache = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)
    assert cache.get("dev", "op@v0", "key") is None


def test_put_requires_explicit_provenance(tmp_path):
    """No default: every call site must make an explicit, reviewable choice.

    This is the safety property itself -- if `put` ever regains a default
    (e.g. back to `PROVENANCE_AUTOTUNE`), a caller that forgets `provenance=`
    silently writes a trusted entry, reproducing the original incident one
    layer down from the ConfigSelectorChain call sites this module guards.
    """
    path = tmp_path / "test-provenance-op.json"
    cache = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)
    with pytest.raises(TypeError):
        cache.put("dev", "op@v0", "key", DummyConfig(algorithm="tuned", block_size=64))


def test_non_hashable_provenance_value_degrades_to_miss_not_crash(tmp_path):
    """A malformed entry (foreign process, disk corruption) must never crash a reader.

    `provenance` is checked via a set-membership test; a list/dict value there
    would raise TypeError on `in` if not guarded, crashing every reader of that
    key instead of degrading to a cache miss as the class docstring promises.
    """
    path = tmp_path / "test-provenance-op.json"
    path.write_text(json.dumps({"dev|op@v0|key": {PROVENANCE_KEY: ["autotune"], "cfg": {"block_size": 999}}}))
    cache = PersistentCache("test-provenance-op", path=str(path), cfg_type=DummyConfig)
    assert cache.get("dev", "op@v0", "key") is None
