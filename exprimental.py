import os

import sys

try:
    import easydel as ed
except ModuleNotFoundError:
    dirname = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(dirname)  # noqa: E402
    sys.path.append(
        os.path.join(
            dirname,
            "src",
        )
    )
    import easydel as ed


def main():
    print(ed.FlaxLlamaForCausalLM)


if __name__ == "__main__":
    main()
