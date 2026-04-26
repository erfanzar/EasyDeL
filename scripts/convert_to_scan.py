#!/usr/bin/env python3
"""Convert model for-loops over self.layers to use ModuleList.scan."""

from __future__ import annotations

import re
from pathlib import Path

# Models that should be skipped (heterogeneous layers or other issues)
SKIP_MODELS = {
    "qwen3_next",
    "gemma4",
    "llama4",
    "arctic",
    "glm4_moe_lite",
    "glm4_moe",
    "qwen3_omni_moe",
    "kimi_linear",
    "qwen3_vl_moe",
    "glm_moe_dsa",
    "falcon_h1",
    "qwen3_moe",
    "qwen2_moe",
    "phimoe",
    "glm4v_moe",
    "gpt_oss",
    "mamba",
    "mamba2",
    "falcon_mamba",
    "minimax",
    "gidd",
    "exaone4",
    "cohere2",
}

SCAN_IMPORT = """        from easydel.infra.layer_scan import (
            _get_view_at_index,
            _try_stack_views,
            _update_stacked_view,
            scan_layers,
        )

        views = past_key_values.views if past_key_values is not None else None
        stacked_views = _try_stack_views(views)
"""


def make_standard_replacement(block_call: str, has_router_logits: bool = False) -> str:
    """Generate scan replacement for standard models."""
    # Extract kwargs from block call
    kwargs_match = re.search(r"block\((.*?)\n            \)", block_call, re.DOTALL)
    if not kwargs_match:
        return None
    kwargs_block = kwargs_match.group(1).strip()

    # Build scan step function kwargs
    scan_kwargs = kwargs_block.replace("cache_view=past_key_values.views[idx]", "cache_view=_get_view_at_index(sv, idx)")
    loop_kwargs = kwargs_block.replace(
        "cache_view=past_key_values.views[idx]", "cache_view=views[idx] if views is not None else None"
    )

    if has_router_logits:
        scan_kwargs = scan_kwargs.replace("output_router_logits=output_router_logits,", "")

    result = SCAN_IMPORT
    result += (
        """
        def _layer_step(block, carry):
            hs, sv, idx = carry
            layer_outputs = block(
                """
        + scan_kwargs
        + """
            )
            hs = layer_outputs.hidden_states
            if sv is not None and layer_outputs.cache_view is not None:
                sv = _update_stacked_view(sv, idx, layer_outputs.cache_view)
            return hs, sv, idx + 1
"""
    )

    if has_router_logits:
        result += (
            """
        def _layer_loop(block, carry):
            hs, ah, aa, ar, idx = carry
            if output_hidden_states:
                ah = ah + (hs,)
            layer_outputs = block(
                """
            + loop_kwargs
            + """
            )
            hs = layer_outputs.hidden_states
            if output_attentions:
                aa = aa + (layer_outputs.attention_weight,)
            if output_router_logits and layer_outputs.router_logits is not None:
                ar = ar + (layer_outputs.router_logits,)
            if past_key_values is not None and layer_outputs.cache_view is not None:
                past_key_values[idx] = layer_outputs.cache_view
            return hs, ah, aa, ar, idx + 1

        if output_hidden_states or output_attentions or output_router_logits or stacked_views is None:
            init_carry = (hidden_states, all_hidden_states, all_attentions, all_router_logits, 0)
            for block in self.layers:
                init_carry = _layer_loop(block, init_carry)
            hidden_states, all_hidden_states, all_attentions, all_router_logits, _ = init_carry
        else:
            init_carry = (hidden_states, stacked_views, 0)
            hidden_states, final_sv, _ = scan_layers(
                self.layers, _layer_step, init_carry, fallback=True
            )
            if past_key_values is not None and final_sv is not None:
                past_key_values.views = [
                    jax.tree.map(lambda leaf: leaf[i], final_sv)
                    for i in range(len(views))
                ]
"""
        )
    else:
        result += (
            """
        def _layer_loop(block, carry):
            hs, ah, aa, idx = carry
            if output_hidden_states:
                ah = ah + (hs,)
            layer_outputs = block(
                """
            + loop_kwargs
            + """
            )
            hs = layer_outputs.hidden_states
            if output_attentions:
                aa = aa + (layer_outputs.attention_weight,)
            if past_key_values is not None and layer_outputs.cache_view is not None:
                past_key_values[idx] = layer_outputs.cache_view
            return hs, ah, aa, idx + 1

        if output_hidden_states or output_attentions or stacked_views is None:
            init_carry = (hidden_states, all_hidden_states, all_attentions, 0)
            for block in self.layers:
                init_carry = _layer_loop(block, init_carry)
            hidden_states, all_hidden_states, all_attentions, _ = init_carry
        else:
            init_carry = (hidden_states, stacked_views, 0)
            hidden_states, final_sv, _ = scan_layers(
                self.layers, _layer_step, init_carry, fallback=True
            )
            if past_key_values is not None and final_sv is not None:
                past_key_values.views = [
                    jax.tree.map(lambda leaf: leaf[i], final_sv)
                    for i in range(len(views))
                ]
"""
        )
    return result


def process_file(filepath: Path) -> bool:
    """Process a single model file. Returns True if modified."""
    content = filepath.read_text()

    # Check if it has the pattern
    if "for idx, block in enumerate(self.layers):" not in content:
        return False

    # Skip models in skip list
    model_name = filepath.parent.name
    if model_name in SKIP_MODELS:
        print(f"Skipping {model_name} (heterogeneous or special case)")
        return False

    # Find the for loop block
    pattern = r"        for idx, block in enumerate\(self\.layers\):\n((?:\n|.*?))\n        (hidden_states = self\.norm|hidden_states = checkpoint_name\(self\.norm)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Could not match loop in {filepath}")
        return False

    loop_block = match.group(1)
    post_line = match.group(2)

    # Check for router_logits
    has_router_logits = "output_router_logits" in loop_block

    # Extract block call
    block_call_match = re.search(r"(layer_outputs = block\(\n.*?\n            \))", loop_block, re.DOTALL)
    if not block_call_match:
        print(f"Could not extract block call in {filepath}")
        return False

    block_call = block_call_match.group(1)

    # Generate replacement
    replacement = make_standard_replacement(block_call, has_router_logits=has_router_logits)
    if replacement is None:
        print(f"Failed to generate replacement for {filepath}")
        return False

    # Build full replacement including the post-line
    full_replacement = replacement + "\n        " + post_line

    # Replace in content
    old_block = match.group(0)
    new_content = content.replace(old_block, full_replacement)

    if new_content == content:
        print(f"No change for {filepath}")
        return False

    filepath.write_text(new_content)
    print(f"Converted {filepath} (router_logits={has_router_logits})")
    return True


def main():
    modules_dir = Path("/home/erfan/EasyDeL-SpecTrax/easydel/modules")
    converted = 0
    for filepath in sorted(modules_dir.glob("*/modeling_*.py")):
        if process_file(filepath):
            converted += 1
    print(f"\nConverted {converted} files")


if __name__ == "__main__":
    main()
