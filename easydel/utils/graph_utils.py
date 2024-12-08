from flax import nnx as nn
import typing as tp

graph = nn.graph


def iter_module_search(model, instance):
	for path, module in graph.iter_graph(model):
		if isinstance(module, instance):
			yield path, module


def get_module_from_path(model, path: tp.Tuple[str, ...]):
	if not path:
		return
	
	mdl = model
	for _path in path:
		mdl = mdl[_path] if isinstance(_path, int) else getattr(mdl, _path)
	return mdl


def set_module_from_path(model, path: tp.Tuple[str, ...], new_value):
	if not path:
		return

	current = model
	for item in path[:-1]:
		current = current[item] if isinstance(item, int) else getattr(current, item)

	last_item = path[-1]
	if isinstance(last_item, int):
		current[last_item] = new_value
	else:
		setattr(current, last_item, new_value)
