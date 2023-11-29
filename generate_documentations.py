import os
import sys


def get_inner(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.exists(os.path.join(path, o))]


def get_dirs(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if
            os.path.exists(os.path.join(path, o)) and os.path.isdir(os.path.join(path, o))]


def get_files(path: str):
    return [os.path.join(path, o) for o in os.listdir(path) if
            os.path.exists(os.path.join(path, o)) and not os.path.isdir(os.path.join(path, o))]


def run(project_locations="lib/python/EasyDel"):
    try:
        for ps in get_inner(project_locations):
            is_file = not os.path.isdir(ps)
            if not ps.endswith("__init__.py") and is_file and ps.endswith('.py'):
                name = ps.replace(".py", "").replace("/", ".")

                md_doc = f"# {name}\n::: {name}"
                md_file = name.replace(".", "-") + '.md'
                with open("docs/" + md_file, 'w') as buffer:
                    buffer.write(md_doc)

                print(f" - {name.replace('.', '/')} : {md_file}")
            else:
                run(ps)
    except NotADirectoryError:
        ...


def main():
    run()


if __name__ == "__main__":
    main()
