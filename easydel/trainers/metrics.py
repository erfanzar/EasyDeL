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

import abc
import re
import time
import typing as tp
from collections import defaultdict

import jax
import numpy as np
from eformer.pytree import auto_pytree
from rich.progress import Progress, ProgressColumn, Task, TaskID
from rich.text import Text
from tqdm.autonotebook import tqdm

from easydel.infra.loss_utils import LossMetrics
from easydel.utils import traversals
from easydel.utils.compiling_utils import ejit
from easydel.utils.traversals import flatten_dict

try:
    import wandb  # type:ignore
except ImportError:
    wandb = None


from eformer.loggings import get_logger
from jax import numpy as jnp

logger = get_logger("TrainerMetrics")


class StepMetrics:
    """Handles calculation and tracking of training metrics."""

    def __init__(self, arguments):
        self.arguments = arguments
        self.start_time = time.time()
        self.step_start_time = time.time()

    def start_step(self):
        """Mark the start of a training step."""
        self.step_start_time = time.time()

    def calculate(
        self,
        metrics: LossMetrics,
        current_step: int,
        epoch: int,
        flops_per_token: float,
        extra_flops_per_token: float,
        batch_size: int,
        seq_length: int,
        learning_rate: float,
        mode: tp.Literal["eval", "train"] | None = None,
        **extras,
    ) -> dict[str, float]:
        """Calculate comprehensive metrics for the training step."""
        step_time = time.time() - self.step_start_time
        total_time = time.time() - self.start_time

        execution_time = metrics.execution_time
        flops = flops_per_token * seq_length
        total_flops = flops * batch_size

        extra_flops = extra_flops_per_token * seq_length
        total_flops += extra_flops * batch_size

        tflops = (total_flops / execution_time) / 1e12
        total_tokens = batch_size * seq_length
        visited_tokens = total_tokens * current_step
        throughput = total_tokens / execution_time
        perf_key = mode + "-mlperf"
        mlperf_metrics = {
            f"{perf_key}/execution_time": float(execution_time),
            f"{perf_key}/flops": float(flops),
            f"{perf_key}/flops_per_token": float(flops_per_token),
            f"{perf_key}/extra_flops": float(extra_flops),
            f"{perf_key}/extra_flops_per_token": float(extra_flops_per_token),
            f"{perf_key}/step_time": float(step_time),
            f"{perf_key}/tflops": float(tflops),
            f"{perf_key}/throughput": throughput,
            f"{perf_key}/total_flops": float(total_flops),
            f"{perf_key}/total_time": float(total_time),
            f"{perf_key}/total_tokens": total_tokens,
        }

        loss = metrics.loss
        z_loss = metrics.z_loss

        basic_metrics = {
            "epoch": int(epoch),
            "execution_time": float(execution_time),
            "learning_rate": float(np.array(learning_rate).item()),
            "loss": float(loss),
            "perplexity": float(jnp.exp(loss)),
            f"{mode}_step": int(current_step),
            f"{mode}_step_time": float(step_time),
            "tflops": float(tflops),
            "visited_tokens": visited_tokens,
            "z_loss": float(z_loss) if z_loss is not None else None,
            **extras,
        }

        if metrics.accuracy is not None:
            basic_metrics["accuracy"] = float(metrics.accuracy)
        if metrics.chosen_rewards is not None:
            basic_metrics["chosen_rewards"] = float(jnp.mean(metrics.chosen_rewards).item())
        if metrics.rejected_rewards is not None:
            basic_metrics["rejected_rewards"] = float(jnp.mean(metrics.rejected_rewards).item())
        if metrics.other_metrics is not None:
            basic_metrics.update(metrics.other_metrics)
        if not self.arguments.performance_mode and (mode == "train" or mode is None):
            detailed_metrics = self._calculate_detailed_metrics(metrics)
            basic_metrics.update(detailed_metrics)
        if mode is not None:
            basic_metrics = {f"{mode}/{k}": v for k, v in basic_metrics.items()}
        basic_metrics.update(mlperf_metrics)

        return basic_metrics

    def _calculate_detailed_metrics(self, metrics: LossMetrics):
        """Calculate additional detailed metrics."""
        detailed_metrics = {}
        getattr_in = lambda x: x if not hasattr(x, "value") else x.value  # noqa
        if self.arguments.log_grad_norms:
            if metrics.max_grad_norm is not None:
                detailed_metrics.update({"train/max_grad_norm": getattr_in(metrics.max_grad_norm).tolist()})

            if metrics.mean_grad_norm is not None:
                detailed_metrics.update({"train/mean_grad_norm": getattr_in(metrics.mean_grad_norm).tolist()})

            # Add per-layer gradient norms
            if metrics.grad_norms is not None:
                detailed_metrics.update(
                    {
                        f"grad_norm/{'.'.join([str(s) for s in layer_name])}": getattr_in(grad_norm).tolist()
                        for layer_name, grad_norm in flatten_dict(metrics.grad_norms).items()
                        if getattr_in(grad_norm) is not None
                    }
                )

        return detailed_metrics


class MetricsTracker:
    """Tracks and aggregates training metrics over time."""

    def __init__(self):
        self.loss_sum = None
        self.accuracy_sum = None
        self.metrics_history = defaultdict(list)
        self.step_offset = 0

    def update(self, loss, accuracy, step):
        """Update tracked metrics with new values."""
        self.loss_sum = loss if self.loss_sum is None else self.loss_sum + loss
        mean_loss = self.loss_sum / (step - self.step_offset)
        if accuracy != float("inf"):
            if accuracy is None:
                accuracy = 0.0
            self.accuracy_sum = accuracy if self.accuracy_sum is None else self.accuracy_sum + accuracy
            mean_accuracy = self.accuracy_sum / (step - self.step_offset)

            return float(mean_loss), float(mean_accuracy)
        return float(mean_loss)

    def reset(self, step):
        """Reset tracked metrics."""
        self.loss_sum = None
        self.accuracy_sum = None
        self.step_offset = step


class MetricsColumn(ProgressColumn):
    """A custom progress column for displaying metrics."""

    def __init__(self, metrics_to_show=None):
        super().__init__()
        self.metrics_to_show = metrics_to_show

    def render(self, task: Task) -> Text:
        """Render the metrics in a organized way."""
        if not task.fields.get("metrics"):
            return Text("")

        metrics = task.fields["metrics"]
        display_items = []

        for key, value in metrics.items():
            if self.metrics_to_show is None:
                if isinstance(value, float):
                    if abs(value) < 0.01 or abs(value) > 1000:
                        formatted_value = f"{value:.4e}"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)

                display_items.append(f"{key}={formatted_value}")
            else:
                if any(metric in key for metric in self.metrics_to_show):
                    if isinstance(value, float):
                        if abs(value) < 0.01 or abs(value) > 1000:
                            formatted_value = f"{value:.4e}"
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)

                    display_items.append(f"{key}={formatted_value}")

        return Text(" • ".join(display_items), style="cyan")


class BaseProgressBar(abc.ABC):
    """Abstract base class for progress bar implementations."""

    @abc.abstractmethod
    def update(self, n: int = 1) -> None:
        pass

    @abc.abstractmethod
    def set_postfix(self, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class NullProgressBar(BaseProgressBar):
    """Dummy progress bar that does nothing - useful for multiprocessing."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, **kwargs) -> None:
        pass

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class TqdmProgressBar(BaseProgressBar):
    """Wrapper for tqdm progress bar."""

    def __init__(self, pbar: tqdm):
        self.pbar = pbar

    def update(self, n: int = 1) -> None:
        self.pbar.update(n)

    def set_postfix(self, **kwargs) -> None:
        for k in list(kwargs.keys()):
            val = kwargs.get(k)
            if isinstance(val, float) and k != "learning_rate":
                kwargs[k] = round(val, 3)
        self.pbar.set_postfix(**kwargs)

    def reset(self) -> None:
        self.pbar.n = 0
        self.pbar.start_t = self.pbar._time()

    def close(self) -> None:
        self.pbar.close()


class JSONProgressBar(BaseProgressBar):
    """Wrapper for JSON"""

    def __init__(self, desc=""):
        self.desc = desc

    def update(self, n: int = 1) -> None: ...

    def set_postfix(self, **kwargs) -> None:
        for k in list(kwargs.keys()):
            val = kwargs.get(k)
            if hasattr(val, "size") and val.size == 1:
                kwargs[k] = val.item()
            if isinstance(val, float) and k != "learning_rate":
                kwargs[k] = round(val, 3)
        logger.info(kwargs)

    def reset(self) -> None: ...

    def close(self) -> None: ...


class RichProgressBar(BaseProgressBar):
    """Wrapper for rich progress bar."""

    def __init__(self, progress: Progress, task_id: TaskID):
        """Initialize RichProgressBar with an existing Progress instance and task_id."""
        self.progress = progress
        self.task_id = task_id
        self._postfix = {}

    def update(self, n: int = 1) -> None:
        self.progress.update(self.task_id, advance=n)

    def set_postfix(self, **kwargs) -> None:
        self._postfix.update(kwargs)
        self.progress.update(self.task_id, metrics=self._postfix)

    def reset(self) -> None:
        self.progress.reset(self.task_id)
        self._postfix = {}

    def close(self) -> None:
        try:
            self.progress.remove_task(self.task_id)
        except KeyError:
            pass


@auto_pytree
class MetricsHistogram:
    """Compute and store histogram data for model weights or activations.

    This class provides a PyTree-compatible way to compute histograms and statistics
    for JAX arrays, optimized for use within JIT-compiled functions.
    """

    bin_counts: jax.Array
    bin_edges: jax.Array

    size: int

    min: jax.Array
    max: jax.Array
    sum: jax.Array

    sum_squares: jax.Array

    @staticmethod
    @ejit
    def _create_histogram_bin_edges(arr: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Create histogram bins and counts.

        Args:
            arr: Input array to create histogram from

        Returns:
            Tuple of (bin_counts, bin_edges)
        """
        bin_edges = jnp.histogram_bin_edges(arr, 64)
        left_edges = bin_edges[:-1, None]
        right_edges = bin_edges[1:, None]
        index = ((arr >= left_edges) & (arr < right_edges)).astype(arr.dtype)
        out = index.sum(axis=1, dtype=arr.dtype)
        out = jax.lax.cond(
            out.size >= 1,
            lambda o: o.at[-1].add(jnp.sum(arr == arr.max())),
            lambda o: o,
            out,
        )
        return out, bin_edges

    @classmethod
    def from_array(cls, arr: jax.Array) -> MetricsHistogram:
        """Create a histogram from an array.

        Args:
            arr: Input array

        Returns:
            MetricsHistogram instance
        """
        flat_arr = arr.reshape(-1)
        bin_counts, bin_edges = cls._create_histogram_bin_edges(flat_arr)
        return cls(
            bin_counts=bin_counts,
            bin_edges=bin_edges,
            size=flat_arr.size,
            min=jnp.min(flat_arr),
            max=jnp.max(flat_arr),
            sum=jnp.sum(flat_arr),
            sum_squares=jnp.sum(flat_arr**2),
        )

    def numpy_histogram(self) -> tuple[jax.Array, jax.Array]:
        """Return histogram data in numpy-compatible format.

        Returns:
            Tuple of (bin_counts, bin_edges)
        """
        return self.bin_counts, self.bin_edges

    @property
    def mean(self) -> jax.Array:
        """Calculate mean of the original array.

        Returns:
            Mean value
        """
        return self.sum / self.size

    @property
    def variance(self) -> jax.Array:
        """Calculate variance of the original array.

        Returns:
            Variance value
        """
        mean = self.mean
        mean_of_squares = self.sum_squares / self.size
        variance = mean_of_squares - (mean**2)
        return variance

    @property
    def std(self) -> jax.Array:
        """Calculate standard deviation of the original array.

        Returns:
            Standard deviation value
        """
        return jnp.sqrt(self.variance).reshape(-1)


@ejit(static_argnums=(1,))
def compute_weight_stats(
    params: dict[str, tp.Any],
    repattern: str,
) -> dict[str, MetricsHistogram]:
    """Compute statistics for model weights in a JIT-compatible way.

    Args:
        params: Model parameters
        repattern: Regular expression pattern to match parameter paths

    Returns:
        Dictionary of weight statistics with keys formatted as 'path/to/param/histogram'
    """
    stats = {}
    for path, param in traversals.flatten_dict(params).items():
        weight = param.value if hasattr(param, "value") else param
        pattern_search = ".".join([str(p) for p in path])
        output_path = "/".join([str(p) for p in path])
        if re.match(repattern, pattern_search):
            stats[f"{output_path}/histogram"] = MetricsHistogram.from_array(weight)

    return stats
