#!/bin/bash
# Docker build and push script for EasyDeL
# Usage: ./scripts/docker-build-push.sh [--push] [--hardware cpu|cuda|tpu|all]

set -e

# Configuration
REGISTRY="ghcr.io"
NAMESPACE="erfanzar"
IMAGE_NAME="easydel"
VERSION=$(grep '^version = ' libs/easydel/pyproject.toml | sed 's/version = "\(.*\)"/\1/')
DATE=$(date +'%Y%m%d')

# Parse arguments
PUSH=false
HARDWARE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --hardware)
            HARDWARE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--push] [--hardware cpu|cuda|tpu|all]"
            echo "  --push: Push images to registry after building"
            echo "  --hardware: Specify which hardware variant to build (default: all)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to build and optionally push an image
build_and_push() {
    local hardware_type=$1
    local tag_suffix=$2

    echo "Building ${hardware_type} image..."

    # Build the image
    sudo docker build \
        --build-arg HARDWARE_TYPE=${hardware_type} \
        --build-arg VERSION=${VERSION} \
        -t ${IMAGE_NAME}:${hardware_type} \
        -t ${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${VERSION}${tag_suffix} \
        -t ${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:latest${tag_suffix} \
        -t ${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${DATE}${tag_suffix} \
        .

    if [ "$PUSH" = true ]; then
        echo "Pushing ${hardware_type} image to registry..."

        # Push all tags. No `docker pull` login probe: pulling `latest` would retag it to the
        # registry's previous image, so the stale image would be pushed as `latest` (r11 S14-2).
        # A failed push (e.g. not logged in) is reported here instead.
        local tag
        for tag in "${VERSION}" "latest" "${DATE}"; do
            if ! sudo docker push "${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${tag}${tag_suffix}"; then
                echo "Failed to push ${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${tag}${tag_suffix}."
                echo "If you are not logged in to ${REGISTRY}:"
                echo "  echo \$GITHUB_TOKEN | sudo docker login ${REGISTRY} -u USERNAME --password-stdin"
                exit 1
            fi
        done

        echo "Successfully pushed ${hardware_type} images"
    fi
}

# Main build logic
echo "EasyDeL Docker Build Script"
echo "Version: ${VERSION}"
echo "Date: ${DATE}"
echo "Push: ${PUSH}"
echo "Hardware: ${HARDWARE}"
echo "----------------------------"

case $HARDWARE in
    cpu)
        build_and_push "cpu" ""
        ;;
    cuda)
        build_and_push "cuda" "-cuda"
        ;;
    tpu)
        build_and_push "tpu" "-tpu"
        ;;
    all)
        build_and_push "cpu" ""
        build_and_push "cuda" "-cuda"
        build_and_push "tpu" "-tpu"
        ;;
    *)
        echo "Invalid hardware type: ${HARDWARE}"
        echo "Valid options: cpu, cuda, tpu, all"
        exit 1
        ;;
esac

echo "Build complete!"

if [ "$PUSH" = false ]; then
    echo ""
    echo "Images built locally. To push to registry, run:"
    echo "  $0 --push --hardware ${HARDWARE}"
fi
