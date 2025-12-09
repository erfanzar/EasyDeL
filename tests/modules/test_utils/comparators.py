"""Output comparison utilities for EasyDeL model testing.

This module provides functions to compare outputs between HuggingFace (PyTorch)
and EasyDeL (JAX) models, including logits, hidden states, and loss values.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing model outputs."""

    success: bool
    max_error: float
    correct_percentage: float
    outputs_match: bool
    loss_match: bool
    details: str


def _color(text: str, color_code: str) -> str:
    """Apply ANSI color to text."""
    return f"\x1b[{color_code}m{text}\x1b[0m"


def compare_outputs(
    name: str,
    hf_output: np.ndarray,
    ed_output: jnp.ndarray,
    hf_loss: float | None = None,
    ed_loss: float | None = None,
    hf_aux_loss: float = 0.0,
    ed_aux_loss: float = 0.0,
    atol: float = 0.125,
    rtol: float = 0.0,
    strict_check: bool = False,
) -> ComparisonResult:
    """Compare HuggingFace and EasyDeL outputs with detailed metrics.

    Args:
        name: Model name for display
        hf_output: HuggingFace output array (logits or hidden states)
        ed_output: EasyDeL output array
        hf_loss: HuggingFace loss value (optional)
        ed_loss: EasyDeL loss value (optional)
        hf_aux_loss: HuggingFace auxiliary loss (for MoE models)
        ed_aux_loss: EasyDeL auxiliary loss (for MoE models)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        strict_check: If True, raise assertion on mismatch

    Returns:
        ComparisonResult with detailed comparison metrics
    """
    # Convert to numpy for comparison
    hf_arr = np.asarray(hf_output)
    ed_arr = np.asarray(ed_output)

    if strict_check:
        np.testing.assert_allclose(hf_arr, ed_arr, atol=atol, rtol=rtol)

    # Check if outputs are close
    outputs_match = bool(jnp.allclose(hf_arr, ed_arr, atol=atol, rtol=rtol))

    # Calculate error metrics
    correct_percentage = float(
        jnp.mean(jnp.where(jnp.isclose(hf_arr, ed_arr, atol=atol, rtol=rtol), 1, 0))
    )
    max_error = float(jnp.abs(hf_arr - ed_arr).max())

    # Find location of maximum difference
    diff = np.abs(hf_arr - ed_arr)
    max_flat = diff.argmax()
    max_idx = np.unravel_index(max_flat, diff.shape)
    max_hf = hf_arr[max_idx]
    max_ed = ed_arr[max_idx]

    # Compare losses if provided
    loss_match = True
    if hf_loss is not None and ed_loss is not None:
        # Adjust for auxiliary loss
        adjusted_ed_loss = ed_loss - ed_aux_loss if name not in ["gpt_oss"] else ed_loss
        loss_match = bool(jnp.allclose(hf_loss, adjusted_ed_loss, atol=0.125, rtol=0))

    # Build detail table (simple format without tabulate)
    table_lines = [
        "| Metric          | HuggingFace              | EasyDeL                  |",
        "|-----------------|--------------------------|--------------------------|",
        f"| Last 5 elements | {str(hf_arr[0, -1, -5:])[:24]:24} | {str(ed_arr[0, -1, -5:])[:24]:24} |",
    ]
    if hf_loss is not None and ed_loss is not None:
        table_lines.append(f"| Loss            | {str(hf_loss)[:24]:24} | {str(ed_loss)[:24]:24} |")
        table_lines.append(f"| AUX Loss        | {str(hf_aux_loss)[:24]:24} | {str(ed_aux_loss)[:24]:24} |")

    table = "\n".join(table_lines)

    # Format colored output
    loss_close_str = _color(str(loss_match), "32" if loss_match else "31")
    max_error_str = _color(f"{max_error:.6f}", "32" if max_error < 1e-2 else "31")
    correct_pct_str = _color(
        f"{correct_percentage:.2%}", "32" if correct_percentage > 0.99 else "31"
    )

    details = f"""
{_color(name, '36;1')}
{table}

{_color('Additional Information:', '33;1')}
Correct %: {correct_pct_str}
Max Error: {max_error_str}
Losses Close: {loss_close_str}
Max diff index: {max_idx}, HF={max_hf}, ED={max_ed}
"""

    # Success criteria: outputs match OR correct percentage > 99.5%
    success = outputs_match or correct_percentage > 0.995

    return ComparisonResult(
        success=success,
        max_error=max_error,
        correct_percentage=correct_percentage,
        outputs_match=outputs_match,
        loss_match=loss_match,
        details=details,
    )


def compare_logits(
    name: str,
    hf_logits: np.ndarray,
    ed_logits: jnp.ndarray,
    hf_loss: float | None = None,
    ed_loss: float | None = None,
    hf_aux_loss: float = 0.0,
    ed_aux_loss: float = 0.0,
    atol: float = 0.125,
    rtol: float = 0.0,
) -> ComparisonResult:
    """Compare logits between HuggingFace and EasyDeL models.

    This is the primary comparison for CAUSAL_LM and other task-specific models.
    """
    return compare_outputs(
        name=name,
        hf_output=hf_logits,
        ed_output=ed_logits,
        hf_loss=hf_loss,
        ed_loss=ed_loss,
        hf_aux_loss=hf_aux_loss,
        ed_aux_loss=ed_aux_loss,
        atol=atol,
        rtol=rtol,
    )


def compare_hidden_states(
    name: str,
    hf_hidden: np.ndarray,
    ed_hidden: jnp.ndarray,
    atol: float = 0.125,
    rtol: float = 0.0,
) -> ComparisonResult:
    """Compare hidden states for BASE_MODULE testing.

    Args:
        name: Model name for display
        hf_hidden: HuggingFace last_hidden_state
        ed_hidden: EasyDeL last_hidden_state
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        ComparisonResult with detailed comparison metrics
    """
    return compare_outputs(
        name=f"{name} (hidden_states)",
        hf_output=hf_hidden,
        ed_output=ed_hidden,
        atol=atol,
        rtol=rtol,
    )


def compare_loss(
    hf_loss: float,
    ed_loss: float,
    hf_aux: float = 0.0,
    ed_aux: float = 0.0,
    atol: float = 0.125,
) -> bool:
    """Compare loss values between HuggingFace and EasyDeL.

    Args:
        hf_loss: HuggingFace loss
        ed_loss: EasyDeL loss
        hf_aux: HuggingFace auxiliary loss
        ed_aux: EasyDeL auxiliary loss
        atol: Absolute tolerance

    Returns:
        True if losses are close within tolerance
    """
    adjusted_ed = ed_loss - ed_aux
    adjusted_hf = hf_loss - hf_aux
    return bool(jnp.allclose(adjusted_hf, adjusted_ed, atol=atol, rtol=0))


def print_comparison_result(result: ComparisonResult) -> None:
    """Print comparison result details to stdout."""
    print(result.details)
