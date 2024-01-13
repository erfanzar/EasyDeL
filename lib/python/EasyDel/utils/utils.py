import typing

import jax
import jax.numpy as jnp
import os
import time

import termcolor
import wandb
from jax.experimental.pjit import pjit, with_sharding_constraint as wsc
from jax.interpreters import pxla


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """
    The make_shard_and_gather_fns function takes in a partition_specs and dtype_specs,
    and returns two functions: shard_fns and gather_fns. The shard function is used to
    shard the input tensor into the specified partitions, while the gather function is used to
    gather all of those shards back together. This allows us to use different data types for each
    partition (e.g., float16 for weights, float32 for activations)

    :param partition_specs: Specify the partitioning of each tensor in the model
    :param dtype_specs: Specify the dtype of the tensor
    :return: A tuple of two functions:
    
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, "dtype", None) in float_dtypes:
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, "dtype") and hasattr(tensor, "dtype"):
                return tensor.astype(dtype_spec.dtype)
            return tensor

        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )

        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()

        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )

        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))

        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns


def get_names_from_partition_spec(partition_specs):
    """
    The get_names_from_partition_spec function takes a partition_specs argument, which is either a dictionary or list.
    If it's a dictionary, the function converts it to a list of values. Then for each item in the partition_specs list:
        If the item is None, continue (do nothing) and move on to next iteration of loop.
        If the item is an instance of str (i.e., if it's just one string), add that string to names set and move on 
        to next iteration of loop.
        Otherwise, (if not None or str), call get_names_from_partition_spec recurs
    
    :param partition_specs: Specify the partitioning of a table
    :return: A list of names from a partition spec
    
    """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def names_in_mesh(*names):
    """
    The names_in_mesh function is a decorator that can be used to check whether
    the names of the axes passed into a function are valid.  It will raise an
    exception if any of the axis names are not in the physical mesh.  For example,
    if you have a function that takes two axes as arguments, and you want to make sure they're both in your mesh:

    :param *names: Pass in a variable number of arguments
    :return: A boolean indicating whether all of the given names are in the physical mesh
    
    """
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def with_sharding_constraint(x, partition_specs):
    """
    The with_sharding_constraint function is used to ensure that the sharding of a tensor
    is consistent with the sharding of its inputs.  This function should be called on any
    tensor which has been created by an operation which does not automatically handle this,
    such as tf.concat or tf.split.

    :param x: Pass in the tensor that is to be sharded
    :param partition_specs: Specify the axis names and partition sizes
    :return: The same value as the original function
    
    """
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_mesh(*axis_names):
        x = wsc(x, partition_specs)
    return x


class Timer:

    def __init__(self, name):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the object with a name and initializes other variables.

        :param self: Represent the instance of the class
        :param name: Give the timer a name
        :return: An instance of the class
        
        """
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """
        The start function starts the timer.
                Args:
                    None

        :param self: Access the attributes and methods of the class in python
        :return: Nothing
        
        """
        assert not self.started_, "timer has already been started"
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """
        The stop function stops the timer and adds the time elapsed since start was called to the total elapsed time.


        :param self: Represent the instance of the class
        :return: The time elapsed since the start function was called
        
        """
        assert self.started_, "timer is not started"
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """
        The reset function sets the elapsed time to 0.0 and the started flag to False.

        :param self: Represent the instance of the class
        :return: True if the timer was running, false otherwise
        
        """
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """
        The elapsed function returns the elapsed time in seconds since the timer was started.
        If reset is True, then it also resets the timer to zero and restarts it.
        If reset is False, then it leaves the timer running.

        :param self: Represent the instance of the class
        :param reset: Reset the timer
        :return: The elapsed time in seconds
        
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
        prefix_color: Color | None = "red"
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

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):

        """
        The write function is used to write the elapsed time of a timer to Tensorboard and/or Weights &amp; Biases.

        :param self: Make the function a method of the class
        :param names: Specify which timer(s) to write
        :param iteration: Keep track of the number of iterations
        :param normalizer: Normalize the time elapsed by a certain value
        :param reset: Reset the timer after it has been written to tensorboard
        :return: Nothing
        
        """
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """
        The log function is used to print the time elapsed for a given function.

        :param self: Represent the instance of the class
        :param names: Specify the name of the timer that we want to log
        :param normalizer: Normalize the time taken to run a function
        :param reset: Reset the timer after logging
        :return: The time taken for the given name
        
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
    """
    The get_mesh function is a helper function that creates a JAX Mesh object.
    
    :param shape: typing.Sequence[int]: Specify the shape of the array that is used to create the mesh
    :param axis_names: typing.Sequence[int]: Specify the Axis Names in mesh
    :return: A mesh object
    
    """
    from jax.sharding import Mesh
    from jax.experimental import mesh_utils
    array = jnp.ones((len(jax.devices()), 1)).reshape(shape)
    return Mesh(mesh_utils.create_device_mesh(array.shape), axis_names)
