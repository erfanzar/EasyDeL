import os
import sys

dirname = os.path.dirname(os.path.basename(__file__))
sys.path.append(dirname)
sys.path.append(os.path.join(dirname, "src"))
static_joins = "\n\t:members:\n\t:undoc-members:\n\t:show-inheritance:"

cache = {}


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
	assert isinstance(xs, dict), f"input is not a dict; it is a {type(xs)}"
	result = {}
	for path, value in xs.items():
		if sep is not None:
			path = path.split(sep)
		cursor = result
		for key in path[:-1]:
			if key not in cursor:
				cursor[key] = {}
			cursor = cursor[key]
		cursor[path[-1]] = value
	return result


def get_inner(path: str):
	return [
		os.path.join(path, o)
		for o in os.listdir(path)
		if os.path.exists(os.path.join(path, o))
	]


def get_dirs(path: str):
	return [
		os.path.join(path, o)
		for o in os.listdir(path)
		if os.path.exists(os.path.join(path, o)) and os.path.isdir(os.path.join(path, o))
	]


def get_files(path: str):
	return [
		os.path.join(path, o)
		for o in os.listdir(path)
		if os.path.exists(os.path.join(path, o))
		and not os.path.isdir(os.path.join(path, o))
	]


def run(
	project_locations="src/easydel/",
	start_head="easydel",
):
	global cache
	try:
		for current_file in get_inner(project_locations):
			if (
				not current_file.endswith("__init__.py")
				and not os.path.isdir(current_file)
				and current_file.endswith(".py")
			):
				# print(current_file)
				doted = start_head.replace(os.path.sep, ".").replace("/", ".") + "."
				name = (
					current_file.replace(".py", "").replace(os.path.sep, ".").replace("/", ".")
				).replace("src.", "")
				# markdown_documentation = f"{name.replace(doted, '')}\n========\n.. automodule:: {name}" + static_joins
				categorical_name = name.replace(doted, "")
				markdown_filename = name.replace(doted, "") + ".rst"

				# with open(docs_file + markdown_filename, "w") as buffer:
				#     buffer.write(markdown_documentation)
				category_tuple = tuple(categorical_name.split("."))
				edited_category_tuple = ()

				for key in category_tuple:
					key = key.split("_")
					capitalized_words = [word.capitalize() for word in key if word != ""]
					edited_category_tuple += (" ".join(capitalized_words),)
				#
				cache[edited_category_tuple] = (
					start_head.replace("/", ".") + "." + markdown_filename.replace(".py", "")
				)
			else:
				run(current_file, start_head=start_head)
	except NotADirectoryError:
		...


def create_rst(name, children, output_dir):
	"""Create an .rst file with a toctree linking its children."""
	rst_path = os.path.join(output_dir, f"{name}.rst")
	# print(rst_path)
	if isinstance(children, dict):
		with open(rst_path, "w") as rst_file:
			rst_file.write(f"{name.replace('_', ' ')}\n{'=' * len(name)}\n\n")
			rst_file.write(".. toctree::\n   :maxdepth: 2\n\n")
			for child_name, child_value in children.items():
				if isinstance(child_value, str):
					children_name = child_value.replace("easydel.", "")
					rst_file.write(
						f"   {children_name.replace('.rst', '').replace('src.', '')}\n"
					)
				else:
					assert isinstance(child_value, dict)
					child_value = flatten_dict(child_value)
				create_rst(child_name, child_value, output_dir)
	else:
		children_name = children.replace("easydel.", "")
		ca = ".".join([s.strip() for s in children_name.split(".")[1:-1]])
		name = f"{ca} package"
		with open(os.path.join(output_dir, children_name), "w") as rst_file:
			rst_file.write(f"{name}\n{'=' * len(name)}\n\n")
			children = children.replace(".rst", "")
			rst_file.write(
				# "```{eval-rst}\n"+
				f".. automodule:: {children}\n"
				f"    :members:\n"
				f"    :undoc-members:\n"
				f"    :show-inheritance:\n"
			)


def generate_api_docs(structure, output_dir):
	"""Recursively generate .rst files based on the given dictionary structure."""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for root_name, children in structure.items():
		create_rst(root_name.replace(" ", "_"), children, output_dir)


def main():
	global cache

	for current_file in get_inner("docs/api_docs/"):
		if current_file.startswith("docs/api_docs/"):
			os.remove(current_file)
			# print("Removed Past generated file: " + current_file)
	run()

	cache = {("APIs",) + k: v for k, v in cache.items()}
	# print(cache)
	pages = unflatten_dict(cache)
	base_dir = "docs/api_docs/"
	generate_api_docs(
		pages,
		base_dir,
	)
	uf_f = flatten_dict(pages)
	st = []
	for k, _ in uf_f.items():
		st.append(k.split("-")[1])
	st = set(st)
	apis_index = """EasyDeL APIs 🔮
====

.. toctree::
   :maxdepth: 2
   
   {form}
   """.format(form="\n   ".join(sorted(st)))
	open(os.path.join(str(base_dir), "APIs.rst"), "w", encoding="utf-8").write(apis_index)


if __name__ == "__main__":
	main()
