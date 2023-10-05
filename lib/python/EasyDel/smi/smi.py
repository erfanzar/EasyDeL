import curses
import json
import subprocess
import time

import IPython.display
import jax
import threading

import os


# Edited version of Jax-SMI from https://github.com/ayaka14732/jax-smi/
def run(note_book=None, interval: float = 1, dir_prefix: str = '/dev/shm', dpr=True):
    if note_book is None:
        import os

        def is_notebook():
            """Returns True if the code is being run in a notebook, False otherwise."""
            return os.environ.get("IPYTHON") is not None

        note_book = is_notebook()
    std = curses.initscr() if not note_book else None
    try:
        while True:
            if not note_book and dpr:
                std.clear()
            output = subprocess.run(
                args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.decode('utf-8')
            if not note_book and dpr:
                std.addstr(output)
                std.refresh()
            if note_book and dpr:
                IPython.display.clear_output(True)
                print(output)

            with open(f'{dir_prefix}/memory.json', 'w') as fin:
                json.dump({
                    'log': output
                }, fin)
            time.sleep(interval)
    except KeyboardInterrupt:
        curses.endwin()


def get_mem(dir_prefix: str = '/dev/shm'):
    return subprocess.run(
        args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.decode('utf-8')


def initialise_tracking(interval: float = 1., dir_prefix: str = '/dev/shm') -> None:
    def inner():
        while True:
            jax.profiler.save_device_memory_profile(f'{dir_prefix}/memory.prof.new')
            os.rename(f'{dir_prefix}/memory.prof.new', f'{dir_prefix}/memory.prof')
            time.sleep(interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()
