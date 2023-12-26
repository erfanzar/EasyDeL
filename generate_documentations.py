import os
import sys
import yaml

cache = {}


def unflatten_dict(xs, sep=None):
    assert isinstance(xs, dict), f'input is not a dict; it is a {type(xs)}'
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
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.exists(os.path.join(path, o))]


def get_dirs(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if
            os.path.exists(os.path.join(path, o)) and os.path.isdir(os.path.join(path, o))]


def get_files(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if
            os.path.exists(os.path.join(path, o)) and not os.path.isdir(os.path.join(path, o))]


def run(project_locations="lib/python/EasyDel", docs_file="docs/"):
    global cache

    try:
        for current_file in get_inner(project_locations):
            if not current_file.endswith(
                    "__init__.py"
            ) and not os.path.isdir(
                current_file
            ) and current_file.endswith(
                ".py"
            ):

                name = current_file.replace(".py", "").replace("/", ".")

                replace_file_regex = "lib/python/EasyDel".replace("/", ".") + "."
                markdown_documentation = f"# {name.replace(replace_file_regex, '')}\n::: {name}"
                categorical_name = name.replace("lib.python.EasyDel.", "")
                markdown_filename = categorical_name.replace(".", "-") + ".md"

                with open(docs_file + markdown_filename, "w") as buffer:
                    buffer.write(markdown_documentation)
                category_tuple = tuple(categorical_name.split("."))
                cache[category_tuple] = markdown_filename
            else:
                run(current_file)
    except NotADirectoryError:
        ...


def main():
    global cache
    run()
    yaml_data = {
        "nav": unflatten_dict(cache),
        "site_name": "EasyDel",
        "copyright": "Erfan Zare Chavoshi-EasyDel",
        "site_author": "Erfan Zare Chavoshi",
        "repo_url": "https://github.com/erfanzar/EasyDel"
    }


if __name__ == "__main__":
    main()
