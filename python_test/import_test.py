import os
import sys
import time

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)  # noqa: E402
sys.path.append(
    os.path.join(
        dirname,
        "..",
    )
)  # noqa: E402


def main():
    start = time.time()
    import easydel as ed

    config = ed.LlamaConfig()  # noqa
    arguments = ed.TrainArguments("", 1)  # noqa

    end = time.time()
    print(f"time took for import {end - start}")


if __name__ == "__main__":
    main()
