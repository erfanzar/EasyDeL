import curses
import json
import subprocess
import time

import IPython.display
import jax
import threading

import os
import re


# Edited version of Jax-SMI from https://github.com/ayaka14732/jax-smi/
def run(note_book=None, interval: float = 1, dir_prefix: str = '/dev/shm', dpr=True):
    """
    The run function is a simple wrapper around the go tool pprof command.
    It runs the command every interval seconds and prints out its output to stdout.
    If you are running this in a notebook, it will print to IPython's display instead of stdout.


    :param note_book: Determine whether the program is running in a notebook or not
    :param interval: float: Specify the time interval between each refresh
    :param dir_prefix: str: Specify the directory where the memory
    :param dpr: Control whether the output is displayed in a notebook or not
    :return: The output of the pprof command
    
    """
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
    """
    The get_mem function is a wrapper around the go tool pprof command.
    It takes in an optional argument, dir_prefix, which defaults to /dev/shm.
    The function then runs the go tool pprof command with arguments -tags and dir_prefix/memory.prof,
    and returns its stdout as a string.

    :param dir_prefix: str: Specify the directory where
    :return: A string of the memory profile
    
    """
    return subprocess.run(
        args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.decode('utf-8')


def initialise_tracking(interval: float = 0.5, dir_prefix: str = '/dev/shm') -> None:
    """
    The initialise_tracking function starts a daemon thread that periodically saves the current memory profile to disk.

    :param interval: float: Specify the time interval between each memory profile
    :param dir_prefix: str: Specify the directory where the memory profile will be saved
    :return: Nothing, but it starts a thread that
    
    """

    def inner():
        while True:
            jax.profiler.save_device_memory_profile(f'{dir_prefix}/memory.prof.new')
            os.rename(f'{dir_prefix}/memory.prof.new', f'{dir_prefix}/memory.prof')
            time.sleep(interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()


def get_capacity_matrix(dir_prefix: str = '/dev/shm'):
    pattern = r'(\d+\.\d+\wB) \((\d+\.\d+%)\): (\w+)(\(.*?\))?'

    def calculate_full_size(size, percent):
        size_in_gb = float(re.search(r'(\d+\.\d+)GB', size).group(1))
        percent_value = 100 / float(re.search(r'(\d+\.\d+)%', percent).group(1))
        full_size = size_in_gb * percent_value
        return full_size

    matches = re.findall(pattern, get_mem(dir_prefix=dir_prefix))
    information = {}
    try:
        for match in matches:
            information[match[2]] = {
                "Used": match[0],
                "Usage Percent": match[1],
                "Process": match[3][1:] if match[3] else "∞",
                "Full Capacity": calculate_full_size(match[0], match[1])
            }
    except (ArithmeticError, AttributeError, KeyError, ValueError):
        ...
    return information
