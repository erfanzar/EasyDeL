import os
import ast
import sys


def get_imports(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for alias in node.names:
                imports.add(f"{module}.{alias.name}" if module else alias.name)

    return imports


def crawl_project(project_path):
    all_imports = set()

    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    imports = get_imports(file_path)
                    if imports:
                        print(f"Imports in {file_path}:")
                        for imp in imports:
                            print(f"  - {imp}")
                        print()
                    all_imports.update(imports)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    print("All unique imports in the project:")
    for imp in sorted(all_imports):
        print(f"  - {imp}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <project_directory>")
        sys.exit(1)

    project_path = sys.argv[1]
    if not os.path.isdir(project_path):
        print(f"Error: {project_path} is not a valid directory")
        sys.exit(1)

    crawl_project(project_path)
