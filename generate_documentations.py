import os
import re

import yaml

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


def run(project_locations="src/python/EasyDel", docs_file="docs/", start_head="src/python/EasyDel"):
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

                markdown_documentation = f"# {name.replace(doted, '')}\n::: {name}"
                categorical_name = name.replace(doted, "")
                markdown_filename = (
                        "generated-" + name
                        .replace(doted, "")
                        .replace(".", "-")
                        + ".md"
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


def main():
    global cache

    for current_file in get_inner("docs/"):
        if current_file.startswith("docs/generated-"):
            os.remove(current_file)
            print("Removed Past generated file: " + current_file)
    run()
    mkdocstrings_options = {
        "mkdocstrings": {
            "handlers": {
                "python": {
                    "options": {
                        "docstring_style": "sphinx"
                    }
                }
            },
        }
    }
    theme_options = {

        "name": "material",
        "highlightjs": True,
        "hljs_languages": [
            "yaml", "python"
        ]

    }

    string_options = """
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: sphinx

repo_url: https://github.com/erfanzar/EasyDel
site_author: Erfan Zare Chavoshi
site_name: EasyDel
copyright: Erfan Zare Chavoshi-EasyDel

theme:
  highlightjs: true
  hljs_languages:
    - yaml
    - python
  name: material
"""

    statics = {
        ("Home",): "index.md",
        ("install",): "Install.md",
        ("AvailableModels",): "AvailableModels.md",
        ("EasyBIT",): "Bits.md",
        ("Examples", "EasyState"): "EasyStateExample.md",
        ("Examples", "LoRA and Transfer Learning"): "LoRA-TransferLearningExample.md",
        ("Examples", "Fine Tuning Example"): "FineTuningExample.md",
        ("Examples", "PytorchServer"): "PyTorchServer.md",
        ("Examples", "JAXServer"): "JAXServer.md",
        ("Examples", "DataProcessing"): "DataProcessing.md",
        ("Examples", "Falcon Models"): "Falcon.md",
        ("Examples", "Llama Models"): "Llama.md",
        ("Examples", "Llama2 Models"): "Llama2.md",
        ("Examples", "Mistral Models"): "Mistral.md",
        ("Examples", "MosaicMPT Models"): "MosaicMPT.md",
        ("Examples", "Easy Attention"): "AttentionModuleExample.md",
        ("Examples", "Model Parameter Quantization"): "Parameter-Quantization.md",
        ("CONTRIBUTING",): "CONTRIBUTING.md"

    }
    cache = {("APIs",) + k: v for k, v in cache.items()}
    cache = statics | cache
    pages = unflatten_dict(cache)

    def custom_sort(page):
        order = {'Home': 0, 'install': 1, "AvailableModels": 2, "Examples": 3}
        return 0, order.get(page, 4), page

    yaml_data = {
        "nav": {key: pages[key] for key in sorted(pages.keys(), key=custom_sort)},
    }
    buff = open("mkdocs.yml", "w")
    yaml.safe_dump(yaml_data, buff)
    chk = open("mkdocs.yml", "r")
    wrote = chk.read()
    output_string = re.sub(r'(\n\s*)(\w[^:\n]*:)(.*?)(?=\n\s*\w[^:\n]*:|\Z)', r'\1- \2\3', str(wrote), flags=re.DOTALL)
    buff = open("mkdocs.yml", "w")
    buff.write(output_string)
    buff.write(string_options)


if __name__ == "__main__":
    main()
