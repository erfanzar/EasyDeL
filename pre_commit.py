import os
import sys

# Setup paths: add current directory and src/ subdirectory to sys.path
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(os.path.join(dirname, "src"))

# Global cache to record module paths (keys as tuples representing the hierarchy)
cache = {}

# Directives to include in automodule docs.
static_joins = "\n\t:members:\n\t:undoc-members:\n\t:show-inheritance:"


def flatten_dict(d, parent_key="", sep="-"):
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, dict):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)


def unflatten_dict(xs, sep=None):
	"""
	Convert a flat dictionary with compound keys (tuples) into a nested dictionary.
	If an intermediate key already has a non-dict value, we promote it by storing its value
	under the reserved key "__self__".
	"""
	assert isinstance(xs, dict), f"input is not a dict; it is a {type(xs)}"
	result = {}
	for compound_key, value in xs.items():
		# compound_key is assumed to be a tuple
		keys = compound_key
		cursor = result
		for key in keys[:-1]:
			if key not in cursor:
				cursor[key] = {}
			elif not isinstance(cursor[key], dict):
				cursor[key] = {"__self__": cursor[key]}
			cursor = cursor[key]
		final_key = keys[-1]
		if final_key in cursor and not isinstance(cursor[final_key], dict):
			cursor[final_key] = {"__self__": cursor[final_key]}
		else:
			cursor[final_key] = value
	return result


def get_inner(path: str):
	return [
		os.path.join(path, o)
		for o in os.listdir(path)
		if os.path.exists(os.path.join(path, o))
	]


def run(project_locations="easydel/", current_head="easydel", base_head=None):
	"""
	Recursively traverse the project directory and add each Python module to the cache.
	The cache keys are tuples reflecting the full hierarchy.

	For directories, if an __init__.py exists, we record the package (without the trailing .__init__)
	so that when unflattening we can later store the package module name under "__self__".
	"""
	global cache
	if base_head is None:
		base_head = current_head

	try:
		for current_file in get_inner(project_locations):
			if os.path.isdir(current_file):
				# Check if the directory is a package by looking for __init__.py
				init_file = os.path.join(current_file, "__init__.py")
				if os.path.exists(init_file):
					module_path = (
						init_file.replace(".py", "").replace(os.path.sep, ".").replace("/", ".")
					).replace("src.", "")
					# Remove trailing .__init__ so that the package is named properly.
					if module_path.endswith(".__init__"):
						module_path = module_path[: -len(".__init__")]
					base_dot = base_head.replace(os.path.sep, ".").replace("/", ".") + "."
					categorical_name = module_path.replace(base_dot, "")
					category_tuple = tuple(categorical_name.split("."))
					edited_category_tuple = tuple(
						" ".join(word.capitalize() for word in part.split("_") if word)
						for part in category_tuple
					)
					# Record the package module name under its tuple key.
					cache[edited_category_tuple] = module_path

				# Recurse into the directory.
				new_head = current_head + "." + os.path.basename(current_file)
				run(current_file, current_head=new_head, base_head=base_head)
			elif current_file.endswith(".py"):
				# Process Python files.
				module_path = (
					current_file.replace(".py", "").replace(os.path.sep, ".").replace("/", ".")
				).replace("src.", "")
				base_dot = base_head.replace(os.path.sep, ".").replace("/", ".") + "."
				categorical_name = module_path.replace(base_dot, "")
				category_tuple = tuple(categorical_name.split("."))
				edited_category_tuple = tuple(
					" ".join(word.capitalize() for word in part.split("_") if word)
					for part in category_tuple
				)
				cache[edited_category_tuple] = module_path
	except NotADirectoryError:
		pass


# Helper to compute a consistent RST file name from a module dotted name.
def get_rst_filename(module_dotted, fallback):
	if module_dotted:
		fname = module_dotted.replace("easydel.", "").replace("src.", "")
		# Replace dots with underscores and lower-case the result.
		fname = fname.replace(".", "_").lower()
		return fname
	else:
		return fallback.lower()


def create_rst(name, children, output_dir, module_dotted=None):
	"""
	Create an RST file for a package or module.

	Parameters:
	  - name: A display name for the module/package (often the key from the hierarchy).
	  - children: A dict (if a package) or a string (if a module).
	  - output_dir: Where to write the file.
	  - module_dotted: The full module dotted name (e.g. "easydel.callib.triton_call").
	    This is used to compute a consistent file name.
	"""
	# For packages, use the package's own module (if available) from __self__
	if isinstance(children, dict):
		package_mod = children.get("__self__", module_dotted)
		rst_filename = get_rst_filename(package_mod, name)
	else:
		rst_filename = get_rst_filename(module_dotted, name)
	rst_path = os.path.join(output_dir, f"{rst_filename}.rst")

	if isinstance(children, dict):
		# This is a package.
		title = name.replace("_", " ")
		with open(rst_path, "w") as rst_file:
			rst_file.write(f"{title}\n{'=' * len(title)}\n\n")
			# Build a list of child entries.
			entries = []
			for child_key, child_val in children.items():
				if child_key == "__self__":
					continue
				# For a child, get its module dotted name if available.
				if isinstance(child_val, str):
					child_mod = child_val
				else:
					child_mod = child_val.get("__self__", None)
				child_fname = get_rst_filename(child_mod, child_key)
				entries.append(child_fname)
			if entries:
				rst_file.write(".. toctree::\n   :maxdepth: 2\n\n")
				for entry in sorted(entries):
					rst_file.write(f"   {entry}\n")
			else:
				rst_file.write("No documented modules available.\n")
		# Recursively create RST files for children.
		for child_key, child_val in children.items():
			if child_key == "__self__":
				continue
			if isinstance(child_val, dict):
				child_mod = child_val.get("__self__", None)
				create_rst(child_key, child_val, output_dir, module_dotted=child_mod)
			else:
				create_rst(child_key, child_val, output_dir, module_dotted=child_val)
	else:
		# This is an individual module.
		mod_disp = module_dotted or name
		title = mod_disp
		with open(rst_path, "w") as rst_file:
			rst_file.write(f"{title}\n{'=' * len(title)}\n\n")
			rst_file.write(
				f".. automodule:: {module_dotted}\n"
				f"    :members:\n"
				f"    :undoc-members:\n"
				f"    :show-inheritance:\n"
			)


def generate_api_docs(structure, output_dir):
	"""
	Recursively generate RST files from the nested structure.
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	for root_name, children in structure.items():
		mod = None
		if isinstance(children, dict):
			mod = children.get("__self__", None)
		elif isinstance(children, str):
			mod = children
		create_rst(root_name, children, output_dir, module_dotted=mod)


def main():
	global cache

	base_api_docs = "docs/api_docs/"
	# Remove any existing files in the output directory.
	if os.path.exists(base_api_docs):
		for current_file in get_inner(base_api_docs):
			os.remove(current_file)
	else:
		os.makedirs(base_api_docs)

	# Traverse the project starting at the "easydel/" directory.
	run("easydel/", current_head="easydel")

	# Optionally, prepend a base tuple ("APIs",) to all keys.
	cache_adjusted = {("APIs",) + k: v for k, v in cache.items()}

	# Unflatten the cache into a nested structure.
	pages = unflatten_dict(cache_adjusted)
	generate_api_docs(pages, base_api_docs)

	# Create an index RST file that links to the documented APIs.
	uf_f = flatten_dict(pages)
	st = set()
	for k, _ in uf_f.items():
		parts = k.split("-")
		if len(parts) > 1:
			st.add(parts[1])
	apis_index = """EasyDeL APIs 🔮
====
    
.. toctree::
   :maxdepth: 2
   
   {form}
   """.format(form="\n   ".join(sorted(st)))
	with open(os.path.join(base_api_docs, "apis.rst"), "w", encoding="utf-8") as f:
		f.write(apis_index)


if __name__ == "__main__":
	main()
