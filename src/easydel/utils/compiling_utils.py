from __future__ import annotations
import os
import hashlib
import pickle
from typing import Optional
import jax
from jax._src.stages import Compiled, Lowered
from jax.experimental.serialize_executable import deserialize_and_load, serialize

from easydel.utils.helpers import get_cache_dir

RECOMPILE_FORCE = os.environ.get("RECOMPILE_FORCE", "false") in ["true", "1", "on"]
CACHE_DIR = get_cache_dir()
COMPILE_FUNC_DIR = CACHE_DIR / "compiled_funcs"
COMPILE_FUNC_DIR.mkdir(parents=True, exist_ok=True)


def get_hash_of_lowering(lowered_func: Lowered):
	text_representation = lowered_func.as_text()
	hash_object = hashlib.sha256(text_representation.encode("utf-8"))
	hash_digest = hash_object.hexdigest()
	return hash_digest


def smart_compile(lowered_func: Lowered, tag: Optional[str] = None):
	func_hash = get_hash_of_lowering(lowered_func)
	foldername = str(func_hash) if tag is None else f"{tag}-{func_hash}"
	func_dir = COMPILE_FUNC_DIR / foldername
	filepath = func_dir / "compiled.func"
	if filepath.exists() and not RECOMPILE_FORCE:
		(serialized, in_tree, out_tree) = pickle.load(open(filepath, "rb"))
		compiled_func = deserialize_and_load(
			serialized=serialized,
			in_tree=in_tree,
			out_tree=out_tree,
		)
		return compiled_func
	else:
		compiled_func: Compiled = lowered_func.compile()
		serialized, in_tree, out_tree = serialize(compiled_func)
		func_dir.mkdir(parents=True, exist_ok=True)
		pickle.dump((serialized, in_tree, out_tree), open(filepath, "wb"))
		return compiled_func


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
