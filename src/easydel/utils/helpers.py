import typing
import warnings
import time
import contextlib

import flax.metrics.tensorboard
import jax
import jax.numpy as jnp
import termcolor

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0.0
        self.started = False
        self.start_time = 0.0

    def start(self):
        if self.started:
            raise RuntimeError(f"Timer '{self.name}' is already running")
        self.start_time = time.time()
        self.started = True

    def stop(self):
        if not self.started:
            raise RuntimeError(f"Timer '{self.name}' is not running")
        self.elapsed += time.time() - self.start_time
        self.started = False

    def reset(self):
        self.elapsed = 0.0
        self.started = False
        self.start_time = 0.0

    def elapsed_time(self, reset=True):
        if self.started:
            self.stop()
        total_time = self.elapsed
        if reset:
            self.reset()
        return total_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


Color = typing.Literal[
    "black",
    "grey",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "light_grey",
    "dark_grey",
    "light_red",
    "light_green",
    "light_yellow",
    "light_blue",
    "light_magenta",
    "light_cyan",
    "white",
]


def prefix_print(prefix, string, prefix_color: typing.Optional[Color] = "red"):
    print(
        termcolor.colored(f"{prefix} : ", color=prefix_color, force_color=True) + string
    )


class Timers:
    def __init__(
        self, use_wandb, tensorboard_writer: flax.metrics.tensorboard.SummaryWriter
    ):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed_time(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                if wandb is None:
                    warnings.warn(
                        "`wandb` is not installed use `pip install wandb` (use_wandb=True will be ignored)"
                    )
                    self.use_wandb = False
                else:
                    wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        assert normalizer > 0.0

        if isinstance(names, str):
            names = [names]
        for name in names:
            elapsed_time = (
                self.timers[name].elapsed_time(reset=reset) * 1000.0 / normalizer
            )
            self._print_log(name, elapsed_time)

    def _print_log(self, name, elapsed_time):
        termcolor.cprint(
            f"Time Took to Complete Task {name} (milliseconds) : "
            f"{termcolor.colored(elapsed_time, color='white', force_color=True)}",
            color="red",
            force_color=True,
        )

    @contextlib.contextmanager
    def timed(self, name, log=True, reset=True):
        timer = self(name)
        try:
            timer.start()
            yield timer
        finally:
            timer.stop()
            if log:
                elapsed_time = (
                    timer.elapsed_time(reset=reset) * 1000.0
                )  # Convert to milliseconds
                self._print_log(name, elapsed_time)


class RNG:

    def __init__(self, seed):
        self.rng = jax.random.PRNGKey(seed)

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def get_mesh(
    shape: typing.Sequence[int] = (1, -1, 1, 1),
    axis_names: typing.Sequence[str] = ("dp", "fsdp", "tp", "sp"),
):
    """The get_mesh function is a helper function that creates a JAX Mesh object.

    Args:
        shape: typing.Sequence[int]: Specify the shape of the array that
            is used to create the mesh
        axis_names: typing.Sequence[int]: Specify the Axis Names in mesh

    Returns:
        A mesh object
    """
    from jax.sharding import Mesh
    from jax.experimental import mesh_utils

    array = jnp.ones((len(jax.devices()), 1)).reshape(shape)
    return Mesh(mesh_utils.create_device_mesh(array.shape), axis_names)
