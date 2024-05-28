import os

static_joins = "\n\t:members:\n\t:undoc-members:\n\t:show-inheritance:"

cache = {}


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
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.exists(os.path.join(path, o))]


def get_dirs(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if
            os.path.exists(os.path.join(path, o)) and os.path.isdir(os.path.join(path, o))]


def get_files(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if
            os.path.exists(os.path.join(path, o)) and not os.path.isdir(os.path.join(path, o))]


def run(project_locations="src/python/easydel", docs_file="docs/api_docs/", start_head="src/python/easydel"):
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

                doted = (
                        start_head
                        .replace(os.path.sep, ".")
                        .replace("/", ".") + "."
                )

                name = (
                    current_file
                    .replace(".py", "")
                    .replace(os.path.sep, ".")
                    .replace("/", ".")
                )

                markdown_documentation = f"{name.replace(doted, '')}\n========\n.. automodule:: {name}" + static_joins
                categorical_name = name.replace(doted, "")
                markdown_filename = (
                        "generated_" + name
                        .replace(doted, "")
                        .replace(".", "-")
                        + ".rst"
                )

                with open(docs_file + markdown_filename, "w") as buffer:
                    buffer.write(markdown_documentation)
                category_tuple = tuple(categorical_name.split("."))
                edited_category_tuple = ()

                for key in category_tuple:
                    key = key.split("_")
                    capitalized_words = [word.capitalize() for word in key if word != ""]
                    edited_category_tuple += (" ".join(capitalized_words),)
                cache[edited_category_tuple] = markdown_filename
            else:
                run(current_file)
    except NotADirectoryError:
        ...


def generate_index_file(nested_dict, parent_title="", depth=0):
    lines = []

    if parent_title:
        indent = "    " * depth
        lines.append(f"{indent}{parent_title}\n")
        lines.append(f"{indent}{'-' * len(parent_title)}\n")

    for key, value in nested_dict.items():
        if isinstance(value, dict):
            lines.extend(generate_index_file(value, key, depth + 1))
        else:
            indent = "    " * (depth + 1)
            lines.append(f"{indent}- {key} <{value}>\n")

    return lines


def main():
    global cache

    for current_file in get_inner("docs/api_docs/"):
        if current_file.startswith("docs/api_docs/generated_"):
            os.remove(current_file)
            # print("Removed Past generated file: " + current_file)
    run()

    cache = {("APIs",) + k: v for k, v in cache.items()}
    pages = unflatten_dict(cache)
    index_lines = [
        "API Home",
        "====\n",
        ".. toctree::\n",
        "    :maxdepth: 2\n",
        "    :caption: Contents\n"
    ]

    index_lines.extend(generate_index_file(pages))
    index_content = "\n".join(index_lines)

    # Write the index content to a file
    with open("docs/api_docs/index.rst", "w") as f:
        f.write(index_content)


if __name__ == "__main__":
    main()
