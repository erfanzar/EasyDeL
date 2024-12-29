# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import functools
import hashlib
import os
import pickle
import typing as tp
import warnings

import jax
from jax.experimental.serialize_executable import deserialize_and_load, serialize

from .helpers import get_cache_dir

if tp.TYPE_CHECKING:
	from jax._src.stages import Compiled, Lowered
else:
	Compiled, Lowered = tp.Any, tp.Any

RECOMPILE_FORCE = os.environ.get("RECOMPILE_FORCE", "false") in ["true", "1", "on"]
ECACHE_COMPILES = os.environ.get("ECACHE_COMPILES", "true") in ["true", "1", "on"]

CACHE_DIR = get_cache_dir()
COMPILE_FUNC_DIR = CACHE_DIR / "compiled_funcs"
COMPILE_FUNC_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_FILE_NAME = "compiled.func"

COMPILED_CACHE: tp.Dict[tp.Tuple, tp.Any] = {}


def is_jit_wrapped(fn):
	return all(
		[
			hasattr(fn, "_fun"),
			hasattr(fn, "lower"),
			hasattr(fn, "eval_shape"),
			hasattr(fn, "trace"),
		]
	)


def cjit(fn, static_argnames=None):
	assert is_jit_wrapped(fn=fn), "function should be jit wrapped already"

	@functools.wraps(fn)
	def wrapped(**kwargs):  # kwargs only !
		signature = get_signature((), kwargs)
		cache_key = (fn, signature)
		if cache_key in COMPILED_CACHE:
			if static_argnames is not None:
				for key in static_argnames:
					kwargs.pop(key)
			return COMPILED_CACHE[cache_key](**kwargs)

		lowered_func: Lowered = fn.lower(**kwargs)
		compiled_func = smart_compile(lowered_func, "cjit")
		COMPILED_CACHE[cache_key] = compiled_func

		if static_argnames is not None:
			for key in static_argnames:
				kwargs.pop(key)
		return compiled_func(**kwargs)

	return wrapped


def hash_fn(self) -> int:
	shu = "".join(
		str(cu)
		for cu in self.__dict__.values()
		if isinstance(cu, (float, int, float, bool, dict, list))
	)
	return get_safe_hash_int(shu)


# @functools.lru_cache(maxsize=2048)
def get_safe_hash_int(text, algorithm="md5"):
	try:
		text_str = str(text)
		hash_object = getattr(hashlib, algorithm)(text_str.encode())
		return int.from_bytes(hash_object.digest(), byteorder="big")
	except AttributeError as e:
		raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
	except Exception as e:
		raise Exception(f"Error generating hash: {str(e)}") from e


def get_signature(args, kwargs) -> tp.Tuple:
	"""Get a hashable signature of args/kwargs shapes and dtypes."""

	def get_array_signature(x):
		if hasattr(x, "shape") and hasattr(x, "dtype"):
			return (tuple(x.shape), str(x.dtype))
		return str(type(x))

	args_sig = tuple(get_array_signature(arg) for arg in args)
	kwargs_sig = tuple((k, get_array_signature(v)) for k, v in sorted(kwargs.items()))
	return (args_sig, kwargs_sig)


def get_hash_of_lowering(lowered_func: Lowered):
	text_representation = lowered_func.as_text()
	hash_object = hashlib.sha256(text_representation.encode("utf-8"))
	hash_digest = hash_object.hexdigest()
	return hash_digest


def smart_compile(lowered_func: Lowered, tag: tp.Optional[str] = None):
	func_hash = get_hash_of_lowering(lowered_func)
	foldername = str(func_hash) if tag is None else f"{tag}-{func_hash}"
	func_dir = COMPILE_FUNC_DIR / foldername
	filepath = func_dir / COMPILED_FILE_NAME
	if filepath.exists() and not RECOMPILE_FORCE:
		try:
			(serialized, in_tree, out_tree) = pickle.load(open(filepath, "rb"))
			compiled_func = deserialize_and_load(
				serialized=serialized,
				in_tree=in_tree,
				out_tree=out_tree,
			)
			return compiled_func
		except Exception as e:
			warnings.warn(f"couldn't load compiled function due to {e}", stacklevel=4)
			compiled_func: Compiled = lowered_func.compile()
			if ECACHE_COMPILES:
				serialized, in_tree, out_tree = serialize(compiled_func)
				func_dir.mkdir(parents=True, exist_ok=True)
				try:
					pickle.dump((serialized, in_tree, out_tree), open(filepath, "wb"))
				except Exception as e:  # noqa
					warnings.warn(f"couldn't save compiled function due to {e}", stacklevel=4)
			return compiled_func
	else:
		compiled_func: Compiled = lowered_func.compile()
		if ECACHE_COMPILES:
			serialized, in_tree, out_tree = serialize(compiled_func)
			func_dir.mkdir(parents=True, exist_ok=True)
			try:
				pickle.dump((serialized, in_tree, out_tree), open(filepath, "wb"))
			except Exception as e:  # noqa
				warnings.warn(f"couldn't save compiled function due to {e}", stacklevel=4)
		return compiled_func


def save_compiled_fn(
	path: tp.Union[str, os.PathLike],
	fn: Compiled,
	prefix: tp.Optional[str] = None,
):
	path.mkdir(parents=True, exist_ok=True)
	prefix = prefix or ""
	filename = path / (prefix + "-" + COMPILED_FILE_NAME)
	serialized, in_tree, out_tree = serialize(fn)
	try:
		pickle.dump((serialized, in_tree, out_tree), open(filename, "wb"))
	except Exception as e:  # noqa
		warnings.warn(f"couldn't save compiled function due to {e}", stacklevel=4)


def load_compiled_fn(
	path: tp.Union[str, os.PathLike],
	prefix: tp.Optional[str] = None,
):
	prefix = prefix or ""
	filename = path / (prefix + "-" + COMPILED_FILE_NAME)
	(serialized, in_tree, out_tree) = pickle.load(open(filename, "rb"))
	return deserialize_and_load(
		serialized=serialized,
		in_tree=in_tree,
		out_tree=out_tree,
	)


def cache_compiles(
	tag: tp.Optional[str] = None,
	static_argnames: tp.Optional[tp.List[str]] = None,
):
	static_argnames = static_argnames or []

	def create_wrapper(func: tp.Callable, tag: tp.Optional[str] = None) -> tp.Callable:
		original_func = getattr(func, "_fun", func)
		func_id = str(
			hashlib.sha256(
				original_func.__code__.co_code,
			)
			.hexdigest()
			.encode("utf-8")
		)

		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			signature = (func_id, get_signature(args, kwargs))
			if signature in COMPILED_CACHE:
				for static_key in static_argnames:
					kwargs.pop(static_key)
				return COMPILED_CACHE[signature](*args, **kwargs)
			if hasattr(func, "lower"):
				lowered = func.lower(*args, **kwargs)
				for static_key in static_argnames:
					kwargs.pop(static_key)
				func_hash = get_hash_of_lowering(lowered)
				sig_hash = hashlib.sha256(str(signature).encode()).hexdigest()[:8]
				foldername = (
					f"{tag}-{func_hash}-{sig_hash}" if tag else f"{func_hash}-{sig_hash}"
				)
				func_dir = COMPILE_FUNC_DIR / foldername
				filepath = func_dir / "compiled.func"

				if filepath.exists() and not RECOMPILE_FORCE:
					with open(filepath, "rb") as f:
						serialized, in_tree, out_tree = pickle.load(f)
					compiled_func = deserialize_and_load(
						serialized=serialized,
						in_tree=in_tree,
						out_tree=out_tree,
					)
					COMPILED_CACHE[signature] = compiled_func
					return compiled_func(*args, **kwargs)

				compiled_func = lowered.compile()
				COMPILED_CACHE[signature] = compiled_func

				try:
					serialized, in_tree, out_tree = serialize(compiled_func)
					func_dir.mkdir(parents=True, exist_ok=True)
					with open(filepath, "wb") as f:
						pickle.dump((serialized, in_tree, out_tree), f)
				except Exception as e:
					print(f"Failed to cache compilation: {e}")

				return compiled_func(*args, **kwargs)
			return func(*args, **kwargs)

		wrapper._COMPILED_CACHE = COMPILED_CACHE
		return wrapper

	def decorator(func: tp.Callable) -> tp.Callable:
		return create_wrapper(func, tag)

	return decorator


def lower_function(
	func,
	func_input_args,
	func_input_kwargs,
	mesh=None,
	in_shardings=None,
	out_shardings=None,
	static_argnums=None,
	donate_argnums=None,
):
	"""
	lower a JAX function with optional sharding and mesh configuration.

	Args:
	    func: The JAX function to compile.
	    func_input_args: Input arguments for the function.
	    func_input_kwargs: Input keyword arguments for the function.
	    mesh: tp.Optional JAX mesh for distributed execution.
	    in_shardings: tp.Optional input sharding specifications.
	    out_shardings: tp.Optional output sharding specifications.
	    static_argnums: Indices of static arguments.
	    donate_argnums: Indices of arguments to donate.

	Returns:
	    lowered JAX function.
	"""
	if mesh is None:
		return jax.jit(
			func,
			in_shardings=in_shardings,
			out_shardings=out_shardings,
			static_argnums=static_argnums,
			donate_argnums=donate_argnums,
		).lower(*func_input_args, **func_input_kwargs)
	with mesh:
		return jax.jit(
			func,
			in_shardings=in_shardings,
			out_shardings=out_shardings,
			static_argnums=static_argnums,
			donate_argnums=donate_argnums,
		).lower(*func_input_args, **func_input_kwargs)


def compile_function(
	func,
	func_input_args,
	func_input_kwargs,
	mesh=None,
	in_shardings=None,
	out_shardings=None,
	static_argnums=None,
	donate_argnums=None,
):
	"""
	Compiles a JAX function with optional sharding and mesh configuration.

	Args:
	    func: The JAX function to compile.
	    func_input_args: Input arguments for the function.
	    func_input_kwargs: Input keyword arguments for the function.
	    mesh: tp.Optional JAX mesh for distributed execution.
	    in_shardings: tp.Optional input sharding specifications.
	    out_shardings: tp.Optional output sharding specifications.
	    static_argnums: Indices of static arguments.
	    donate_argnums: Indices of arguments to donate.

	Returns:
	    Compiled JAX function.
	"""
	return lower_function(
		func,
		func_input_args,
		func_input_kwargs,
		mesh=mesh,
		in_shardings=in_shardings,
		out_shardings=out_shardings,
		static_argnums=static_argnums,
		donate_argnums=donate_argnums,
	).compile()


if __name__ == "__main__":
	jnp = jax.numpy

	@cjit
	@jax.jit
	def my_function(x, y):
		return x * y + x

	a = jnp.array([1, 2, 3], dtype=jnp.float32)
	b = jnp.array([4, 5, 6], dtype=jnp.float32)

	result1 = my_function(a, b)  # Compiles and caches on first call
	result2 = my_function(a, b)  # Returns cached result

	c = jnp.array([1, 2, 3], dtype=jnp.float32)
	d = jnp.array([1, 1, 1], dtype=jnp.float32)
	result3 = my_function(c, d)
	print(result1, result2, result3)
