"""Test-suite compatibility hooks."""

from __future__ import annotations


def _patch_removed_jax_config_flags() -> None:
    try:
        import jax
    except Exception:
        return
    update = getattr(jax.config, "update", None)
    if update is None or getattr(update, "_easydel_tests_removed_flag_patch", False):
        return

    removed_flags = {"jax_pmap_shmap_merge"}

    def _patched_update(name, value):
        if name in removed_flags:
            return None
        return update(name, value)

    _patched_update._easydel_tests_removed_flag_patch = True  # type: ignore[attr-defined]
    jax.config.update = _patched_update


_patch_removed_jax_config_flags()
