import time


def main():
    start = time.time()
    from lib.python.EasyDel import DPOTrainer
    end = time.time()
    print(f"time took for import {end - start}")


if __name__ == "__main__":
    main()
