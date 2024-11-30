from __future__ import annotations

import functools
import hashlib
import os
import pickle
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
from jax._src.stages import Compiled, Lowered
from jax.experimental.serialize_executable import deserialize_and_load, serialize

from easydel.utils.helpers import get_cache_dir

RECOMPILE_FORCE = os.environ.get("RECOMPILE_FORCE", "false") in ["true", "1", "on"]
ECACHE_COMPILES = os.environ.get("ECACHE_COMPILES", "true") in ["true", "1", "on"]

CACHE_DIR = get_cache_dir()
COMPILE_FUNC_DIR = CACHE_DIR / "compiled_funcs"
COMPILE_FUNC_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_FILE_NAME = "compiled.func"

COMPILED_CACHE: Dict[Tuple, Any] = {}


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


def get_signature(args, kwargs) -> Tuple:
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


def smart_compile(lowered_func: Lowered, tag: Optional[str] = None):
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
	path: Union[str, os.PathLike],
	fn: Compiled,
	prefix: Optional[str] = None,
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
	path: Union[str, os.PathLike],
	prefix: Optional[str] = None,
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
	tag: Optional[str] = None,
	static_argnames: Optional[List[str]] = None,
):
	static_argnames = static_argnames or []

	def create_wrapper(func: Callable, tag: Optional[str] = None) -> Callable:
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

	def decorator(func: Callable) -> Callable:
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
	    mesh: Optional JAX mesh for distributed execution.
	    in_shardings: Optional input sharding specifications.
	    out_shardings: Optional output sharding specifications.
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
	    mesh: Optional JAX mesh for distributed execution.
	    in_shardings: Optional input sharding specifications.
	    out_shardings: Optional output sharding specifications.
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
