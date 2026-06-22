# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""Device-remote decorators for TPU/GPU execution."""

import functools
from collections.abc import Callable

import ray

from ..core.cluster import SliceInfo
from ..pool.slice import SlicePoolManager
from ..resources.configs import TpuAcceleratorConfig
from .executor import _build_multislice_envs


class TpuRemoteManager:
    """Session-based manager for TPU remote execution across multiple slices.

    Provides a high-level interface for managing TPU resources and executing
    functions or class methods across multiple TPU slices. Each execution runs
    in a short-lived worker process to avoid TPU handle leaks.

    Key features:
    - Automatic slice scaling and warming
    - Persistent DeviceHostActors for efficient execution
    - Broadcasting of function/method calls across all hosts
    - Ephemeral worker processes for each execution
    - MegaScale coordination for multi-slice workloads

    Attributes:
        tpu_version: TPU version string (e.g., 'v4', 'v5p')
        pod_count: Number of TPU pods/slices to use
        base_env: Base environment variables for all workers
        runtime_env: Ray runtime environment configuration
        coord_port: Port for MegaScale coordinator (default 8081)

    Example:
        >>> manager = TpuRemoteManager(
        ...     tpu_version='v4',
        ...     pod_count=2,
        ...     base_env={'MY_VAR': 'value'}
        ... )
        >>> manager.ensure()
        >>> results = manager.run_function(my_func, arg1, arg2)
        >>> manager.close()
    """

    def __init__(
        self,
        tpu_version: str,
        pod_count: int,
        *,
        base_env: dict[str, str] | None = None,
        runtime_env: dict | None = None,
        coord_port: int = 8081,
    ):
        self.tpu_version = tpu_version
        self.pod_count = int(pod_count)
        self.base_env = dict(base_env or {})
        self.runtime_env = dict(runtime_env or {})
        self.coord_port = int(coord_port)

        self._pool = SlicePoolManager(tpu_type=tpu_version)
        self._members = None
        self._slice_infos: list[SliceInfo] = []
        self._per_slice_envs: list[dict[str, str]] = []
        self._host_handles_by_slice: list[list[ray.actor.ActorHandle]] = []

    def ensure(self):
        """Initialize and prepare TPU resources if not already done.

        This method:
        1. Scales the slice pool to the requested pod_count
        2. Prepares all slices (creates placement groups, etc.)
        3. Ensures DeviceHostActors are created on each host
        4. Builds per-slice environment variables including MegaScale config
        5. Caches host actor handles for efficient execution

        Raises:
            RuntimeError: If no SliceActors are available after scaling.

        Note:
            This method is idempotent - calling it multiple times has no
            additional effect after the first successful call.
        """
        if self._members is not None:
            return

        self._pool.scale_multislice(self.pod_count)
        self._pool.prepare_all_slices()

        members = self._pool.get_all_pool_members()
        if not members:
            raise RuntimeError("No SliceActors available after scaling.")
        ray.get([m.actor.ensure_host_pool.remote() for m in members])

        slice_infos = ray.get([m.actor.get_info.remote() for m in members])

        base_env = dict(self.base_env)
        base_env.update(
            TPU_VERSION=self.tpu_version,
            TPU_POD_COUNT=str(len(members)),
        )
        self._per_slice_envs = _build_multislice_envs(slice_infos, base_env, self.coord_port)

        host_handles_by_slice = ray.get([m.actor.get_all_actors_in_pool.remote() for m in members])

        self._members = members
        self._slice_infos = slice_infos
        self._host_handles_by_slice = host_handles_by_slice

    def close(self):
        """Clean up all TPU resources and reset state.

        Performs cleanup in the following order:
        1. Cancels any current work on all host actors
        2. Drains the actor pool (terminates all actors)
        3. Resets internal state

        Note:
            This method is safe to call multiple times and will suppress
            any errors during cleanup to ensure resources are freed.
        """

        try:
            if self._members:
                for m in self._members:
                    try:
                        hosts = ray.get(m.actor.get_all_actors_in_pool.remote())
                        ray.get([h.cancel_current.remote() for h in hosts], timeout=10)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self._pool.drain_actor_pool()
        finally:
            self._members = None
            self._slice_infos = []
            self._per_slice_envs = []
            self._host_handles_by_slice = []

    def run_function(self, fn: Callable, *args, flatten: bool = True, **kwargs):
        """Execute a Python function on every TPU host across all slices.

        Runs the provided function in ephemeral worker processes on each host
        to avoid TPU handle leaks. The function is executed with the same
        arguments on every host.

        Args:
            fn: Python callable to execute on each host. Should be pickleable.
            *args: Positional arguments to pass to the function.
            flatten: If True, returns a flat list of results from all hosts.
                If False, returns nested lists where outer list represents
                slices and inner lists contain results from hosts within
                each slice.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            If flatten=True: List of results from all hosts (flattened).
            If flatten=False: List of lists, where each inner list contains
                results from hosts within a slice.

        Note:
            Each execution runs in a new process to avoid TPU handle leaks.
            The function must be pickleable for Ray serialization.
        """
        self.ensure()

        outer_refs_per_slice = []
        for sl, hosts in enumerate(self._host_handles_by_slice):
            env = self._per_slice_envs[sl]

            outer = [
                h.run_remote_fn.remote(
                    fn,
                    f_args=args,
                    f_kwargs=kwargs,
                    runtime_env=self.runtime_env,
                    env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                )
                for host_idx, h in enumerate(hosts)
            ]
            outer_refs_per_slice.append(outer)

        inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
        if flatten:
            flat_refs = [ref for refs in inner_per_slice for ref in refs]
            return ray.get(flat_refs)
        else:
            return [ray.get(refs) for refs in inner_per_slice]

    def run_class_method(
        self,
        cls_obj: type,
        init_args: tuple,
        init_kwargs: dict,
        method_name: str,
        *call_args,
        flatten: bool = True,
        **call_kwargs,
    ):
        """Execute a class method on every TPU host across all slices.

        Instantiates the class and calls the specified method in ephemeral
        worker processes on each host. The class is reconstructed inside
        each worker to ensure clean TPU state.

        Args:
            cls_obj: The class object (not an instance). Must be pickleable.
            init_args: Positional arguments for class initialization.
            init_kwargs: Keyword arguments for class initialization.
            method_name: Name of the method to call on the instantiated object.
            *call_args: Positional arguments for the method call.
            flatten: If True, returns a flat list of results from all hosts.
                If False, returns nested lists by slice.
            **call_kwargs: Keyword arguments for the method call.

        Returns:
            If flatten=True: List of results from all hosts (flattened).
            If flatten=False: List of lists, where each inner list contains
                results from hosts within a slice.

        Example:
            >>> results = manager.run_class_method(
            ...     MyModel,
            ...     init_args=(config,),
            ...     init_kwargs={'checkpoint': 'path/to/ckpt'},
            ...     method_name='train',
            ...     epochs=10,
            ...     batch_size=32
            ... )

        Note:
            The class is instantiated fresh in each worker process,
            avoiding TPU handle leaks from persistent objects.
        """
        self.ensure()

        def _class_method_entry(cls_obj, i_args, i_kwargs, mname, c_args, c_kwargs):
            obj = cls_obj(*i_args, **i_kwargs)
            return getattr(obj, mname)(*c_args, **c_kwargs)

        outer_refs_per_slice = []
        for sl, hosts in enumerate(self._host_handles_by_slice):
            env = self._per_slice_envs[sl]
            payload = (cls_obj, init_args, init_kwargs, method_name, call_args, call_kwargs)
            outer = [
                h.run_remote_fn.remote(
                    _class_method_entry,
                    f_args=payload,
                    runtime_env=self.runtime_env,
                    env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                )
                for host_idx, h in enumerate(hosts)
            ]
            outer_refs_per_slice.append(outer)

        inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
        if flatten:
            flat_refs = [ref for refs in inner_per_slice for ref in refs]
            return ray.get(flat_refs)
        else:
            return [ray.get(refs) for refs in inner_per_slice]


class _BoundRemoteClass:
    """Bound handle for remote class execution on TPU hosts.

    Internal class that provides a proxy interface for calling methods on
    a class that will be instantiated and executed on TPU hosts. Supports
    both synchronous execution (returning values) and asynchronous execution
    (returning Ray ObjectRefs).

    This class is created by the tpu_remote decorator when applied to classes.

    Attributes:
        _manager: TpuRemoteManager instance for resource management
        _cls: The class object to instantiate on remote hosts
        _init_args: Positional arguments for class initialization
        _init_kwargs: Keyword arguments for class initialization
        _flatten: Whether to flatten results from multiple hosts
    """

    def __init__(self, manager: TpuRemoteManager, cls: type, init_args: tuple, init_kwargs: dict, flatten: bool):
        self._manager = manager
        self._cls = cls
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._flatten = flatten

    def __getattr__(self, name: str):
        def _call_method(*args, **kwargs):
            return self._manager.run_class_method(
                self._cls,
                self._init_args,
                self._init_kwargs,
                name,
                *args,
                flatten=self._flatten,
                **kwargs,
            )

        _call_method.remote = self._remote(name)
        return _call_method

    def _remote(self, name: str):
        def _remote_impl(*args, **kwargs):
            def _class_method_entry(cls_obj, i_args, i_kwargs, mname, c_args, c_kwargs):
                obj = cls_obj(*i_args, **i_kwargs)
                return getattr(obj, mname)(*c_args, **c_kwargs)

            self._manager.ensure()
            outer_refs_per_slice = []
            for sl, hosts in enumerate(self._manager._host_handles_by_slice):
                env = self._manager._per_slice_envs[sl]
                payload = (self._cls, self._init_args, self._init_kwargs, name, args, kwargs)
                outer = [
                    h.run_remote_fn.remote(
                        _class_method_entry,
                        f_args=payload,
                        runtime_env=self._manager.runtime_env,
                        env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                    )
                    for host_idx, h in enumerate(hosts)
                ]
                outer_refs_per_slice.append(outer)

            inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
            if self._flatten:
                return [ref for refs in inner_per_slice for ref in refs]
            else:
                return inner_per_slice

        return _remote_impl

    def close(self):
        """Close the manager and clean up TPU resources.

        Delegates to the underlying TpuRemoteManager to terminate all actors
        and free TPU resources. Safe to call multiple times.
        """
        self._manager.close()


class _RemoteClassWrapper:
    """Wrapper that converts a regular class into a TPU-remote class.

    Internal class used by the tpu_remote decorator to wrap classes for
    remote execution. When the wrapped class is instantiated, it returns
    a _BoundRemoteClass that broadcasts method calls to TPU hosts.

    Attributes:
        _manager: TpuRemoteManager instance for resource management
        _flatten: Whether to flatten results from multiple hosts
    """

    def __init__(self, manager: TpuRemoteManager, flatten: bool):
        self._manager = manager
        self._flatten = flatten

    def __call__(self, cls: type):
        @functools.wraps(cls)
        def ctor(*args, **kwargs):
            return _BoundRemoteClass(self._manager, cls, args, kwargs, self._flatten)

        return ctor


class _RemoteFunctionWrapper:
    """Wrapper that converts a regular function into a TPU-remote function.

    Internal class used by the tpu_remote decorator to wrap functions for
    remote execution. The wrapped function broadcasts calls to all TPU hosts
    and supports both synchronous and asynchronous execution modes.

    Attributes:
        _manager: TpuRemoteManager instance for resource management
        _flatten: Whether to flatten results from multiple hosts
    """

    def __init__(self, manager: TpuRemoteManager, flatten: bool):
        self._manager = manager
        self._flatten = flatten

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self._manager.run_function(fn, *args, flatten=self._flatten, **kwargs)

        def _remote(*args, **kwargs):
            self._manager.ensure()
            outer_refs_per_slice = []
            for sl, hosts in enumerate(self._manager._host_handles_by_slice):
                env = self._manager._per_slice_envs[sl]
                outer = [
                    h.run_remote_fn.remote(
                        fn,
                        f_args=args,
                        f_kwargs=kwargs,
                        runtime_env=self._manager.runtime_env,
                        env=dict(env, EXECUTOR_CALL_INDEX=str(host_idx)),
                    )
                    for host_idx, h in enumerate(hosts)
                ]
                outer_refs_per_slice.append(outer)
            inner_per_slice = [ray.get(outer) for outer in outer_refs_per_slice]
            if self._flatten:
                return [ref for refs in inner_per_slice for ref in refs]
            else:
                return inner_per_slice

        wrapper.remote = _remote  # type: ignore[attr-defined]
        return wrapper


def device_remote(*, accelerator_config: TpuAcceleratorConfig, flatten: bool = True):
    """Decorator for TPU-remote execution of functions or classes.

    Transforms a regular Python function or class into a TPU-remote version
    that automatically broadcasts execution across all TPU hosts in the
    specified configuration. Each execution runs in an ephemeral worker
    process to avoid TPU handle leaks.

    Args:
        accelerator_config: TPU configuration specifying version, pod count,
            and other execution parameters.
        flatten: If True (default), returns flat list of results from all
            hosts. If False, returns nested lists where outer list represents
            slices and inner lists contain results from hosts within each slice.

    Returns:
        Decorator function that wraps the target function or class.

    Example for functions:
        >>> @device_remote(accelerator_config=tpu_config)
        >>> def compute(x, y):
        ...     return jax.numpy.dot(x, y)
        >>>
        >>>
        >>> results = compute(array1, array2)
        >>>
        >>>
        >>> refs = compute.remote(array1, array2)
        >>> results = ray.get(refs)

    Example for classes:
        >>> @tpu_remote(accelerator_config=tpu_config)
        >>> class Model:
        ...     def __init__(self, config):
        ...         self.config = config
        ...
        ...     def train(self, data):
        ...
        ...         return metrics
        >>>
        >>> model = Model(config)
        >>> metrics = model.train(data)
        >>> refs = model.train.remote(data)

    Note:
        - Functions/classes must be pickleable for Ray serialization
        - Each method call creates new worker processes on TPU hosts
        - The decorator manages slice scaling and resource allocation
        - Call model.close() to clean up resources when done
    """
    full_runtime_env = dict(accelerator_config.execution_env or {})
    env_vars = dict(full_runtime_env.get("env_vars", {}))
    full_runtime_env["env_vars"] = env_vars

    mgr = TpuRemoteManager(
        tpu_version=accelerator_config.tpu_version,
        pod_count=accelerator_config.pod_count,
        base_env=env_vars,
        runtime_env=full_runtime_env,
        coord_port=8081,
    )

    def decorator(obj: Callable | type):
        if isinstance(obj, type):
            return _RemoteClassWrapper(mgr, flatten)(obj)
        elif callable(obj):
            return _RemoteFunctionWrapper(mgr, flatten)(obj)
        else:
            raise TypeError("tpu_remote can only decorate a function or a class.")

    return decorator
