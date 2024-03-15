import time


def main():
    start = time.time()
    import lib.python.EasyDel as ed
    end = time.time()
    print(f"time took for import {end - start}")


if __name__ == "__main__":
    main()
