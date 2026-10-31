#!/usr/bin/env bash
# Shim: this toolchain moved into eray. Same output, fixed resource labels.
echo "generate-cluster-configs.py has moved into eray:" >&2
echo "    eray autoscale generate [--zones Z1,Z2] [--families v4,v5e,v5p,v6e] [--spot|--on-demand]" >&2
exec eray autoscale generate "$@"
