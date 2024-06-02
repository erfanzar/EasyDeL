import typing
import warnings

import flax.metrics.tensorboard
import jax
import jax.numpy as jnp
import time

import termcolor

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class Timer:

    def __init__(self, name):
        """The __init__ function is called when the class is instantiated.
        It sets up the object with a name and initializes other variables.

        Args:
            self: Represent the instance of the class
            name: Give the timer a name

        Returns:
            An instance of the class
        """
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """The start function starts the timer.
                Args:
                    None

        Args:
            self: Access the attributes and methods of the class in
                python

        Returns:
            Nothing
        """
        assert not self.started_, "timer has already been started"
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """The stop function stops the timer and adds the time elapsed since start was called to the total elapsed time.

        Args:
            self: Represent the instance of the class

        Returns:
            The time elapsed since the start function was called
        """
        assert self.started_, "timer is not started"
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """The reset function sets the elapsed time to 0.0 and the started flag to False.

        Args:
            self: Represent the instance of the class

        Returns:
            True if the timer was running, false otherwise
        """
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """The elapsed function returns the elapsed time in seconds since the timer was started.
        If reset is True, then it also resets the timer to zero and restarts it.
        If reset is False, then it leaves the timer running.

        Args:
            self: Represent the instance of the class
            reset: Reset the timer

        Returns:
            The elapsed time in seconds
        """
        started_ = self.started_
        if self.started_:
            self.stop()
        elapsed_ = self.elapsed_
        if reset:
            self.reset()
        if started_:
            self.start()
        return elapsed_


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


def prefix_print(
        prefix,
        string,
        prefix_color: typing.Optional[Color] = "red"
):
    print(
        termcolor.colored(
            f"{prefix} : ",
            color=prefix_color,
            force_color=True
        ) + string
    )


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer: flax.metrics.tensorboard.SummaryWriter):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):

        """The write function is used to write the elapsed time of a timer to Tensorboard and/or Weights &amp; Biases.

        Args:
            self: Make the function a method of the class
            names: Specify which timer(s) to write
            iteration: Keep track of the number of iterations
            normalizer: Normalize the time elapsed by a certain value
            reset: Reset the timer after it has been written to
                tensorboard

        Returns:
            Nothing
        """
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                if wandb is None:
                    warnings.warn("`wandb` is not installed use `pip install wandb` (use_wandb=True will be ignored)")
                    self.use_wandb = False
                else:
                    wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """The log function is used to print the time elapsed for a given function.

        Args:
            self: Represent the instance of the class
            names: Specify the name of the timer that we want to log
            normalizer: Normalize the time taken to run a function
            reset: Reset the timer after logging

        Returns:
            The time taken for the given name
        """
        assert normalizer > 0.0

        if isinstance(names, str):
            names = [names]
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            termcolor.cprint(
                f"Time Took to Complete Task {name} (microseconds) : "
                f"{termcolor.colored(elapsed_time, color='white', force_color=True)}",
                color="cyan",
                force_color=True
            )


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
        axis_names: typing.Sequence[str] = ("dp", "fsdp", "tp", "sp")
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


