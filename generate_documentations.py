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

                # markdown_documentation = f"{name.replace(doted, '')}\n========\n.. automodule:: {name}" + static_joins
                categorical_name = name.replace(doted, "")
                markdown_filename = (
                        name
                        .replace(doted, "")
                        + ".rst"
                )

                # with open(docs_file + markdown_filename, "w") as buffer:
                #     buffer.write(markdown_documentation)
                category_tuple = tuple(categorical_name.split("."))
                edited_category_tuple = ()

                for key in category_tuple:
                    key = key.split("_")
                    capitalized_words = [word.capitalize() for word in key if word != ""]
                    edited_category_tuple += (" ".join(capitalized_words),)

                cache[edited_category_tuple] = start_head.replace("/", ".") + "." + markdown_filename
            else:
                run(current_file)
    except NotADirectoryError:
        ...


def create_rst_pages(data, output_dir=".", parent_page=None):
    """
    Recursively generates .rst pages from a nested dictionary.

    Args:
        data (dict): The dictionary representing the page structure.
        output_dir (str, optional): The directory to write the .rst files to. Defaults to "." (current directory).
        parent_page (str, optional): The name of the parent page (used for creating links). Defaults to None.
    """

    for key, value in data.items():
        # Create a file-system friendly page name
        page_name = key.replace(" ", "_") + ".rst"

        # Create the full path to the .rst file
        file_path = os.path.join(output_dir, page_name)

        # Construct the content of the .rst file
        content = f"{key}\n{'=' * len(key)}\n\n"  # Title with underline

        # Add a link back to the parent page if it exists
        if parent_page:
            content += f".. automodule:: {parent_page.replace('.rst', '')}\n   :members:\n\n"

        # Handle nested dictionaries (sub-pages)
        if isinstance(value, dict):
            create_rst_pages(value, output_dir, page_name)  # Recursive call
            # Add links to the sub-pages
            for sub_key in value.keys():
                sub_page_name = sub_key.replace(" ", "_") + ".rst"
                content += f".. toctree::\n   :maxdepth: 2\n\n   {sub_page_name.replace('.rst','')}\n\n"
        # Handle leaf nodes (links to existing .rst files)
        elif isinstance(value, str):
            content += f".. automodule:: {value.replace('.rst', '')}\n   :members:\n   :undoc-members:\n   :show-inheritance:\n"

            # Write the content to the .rst file
        with open(file_path, "w") as f:
            f.write(content)


def main():
    global cache

    for current_file in get_inner("docs/api_docs/"):
        if current_file.startswith("docs/api_docs/generated_"):
            os.remove(current_file)
            # print("Removed Past generated file: " + current_file)
    run()

    cache = {("APIs",) + k: v for k, v in cache.items()}
    pages = unflatten_dict(cache)
    # index_lines = [
    #     "API Home",
    #     "====\n",
    #     ".. toctree::\n",
    #     "    :maxdepth: 2\n",
    #     "    :caption: Contents\n"
    # ]

    create_rst_pages(pages, "docs/api_docs/")


if __name__ == "__main__":
    main()
