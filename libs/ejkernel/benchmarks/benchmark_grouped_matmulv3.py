#!/usr/bin/env python3
"""Benchmark grouped_matmulv3 across all available implementations."""

import os
import sys

sys.path.append(os.path.dirname(__file__))
from _op_benchmark_registry import run_benchmark

if __name__ == "__main__":
    raise SystemExit(run_benchmark("grouped_matmulv3"))
