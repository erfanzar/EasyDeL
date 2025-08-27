#!/bin/bash
# Convenience script to build and release Docker images using pre-commit
# Usage: ./scripts/docker-release.sh [--push] [--hardware cpu|gpu|tpu|all]

# Pass all arguments to the docker-build-push script via environment variables
export DOCKER_BUILD=1
export DOCKER_BUILD_ARGS="$@"

# Run the pre-commit hook manually
pre-commit run docker-build-push --hook-stage manual

# Clean up environment variables
unset DOCKER_BUILD
unset DOCKER_BUILD_ARGS