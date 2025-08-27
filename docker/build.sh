#!/bin/bash
set -e

# EasyDeL Docker Build Script
# Usage: ./docker/build.sh [HARDWARE_TYPE] [TARGET] [OPTIONS]

HARDWARE_TYPE=${1:-cpu}
TARGET=${2:-production}
VERSION=${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo "latest")}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
REGISTRY=${REGISTRY:-ghcr.io/erfanzar}
IMAGE_NAME=${IMAGE_NAME:-easydel}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if buildx is available and properly configured
check_buildx() {
    if ! docker buildx version >/dev/null 2>&1; then
        echo -e "${YELLOW}Docker buildx not found. Using regular docker build.${NC}"
        return 1
    fi

    # Check if we have a builder that supports cache export
    local current_builder=$(docker buildx inspect 2>/dev/null | grep "^Name:" | awk '{print $2}')
    local driver=$(docker buildx inspect 2>/dev/null | grep "^Driver:" | awk '{print $2}')

    if [ "$driver" = "docker" ]; then
        echo -e "${YELLOW}Current buildx driver is 'docker' which doesn't support cache export.${NC}"
        echo -e "${YELLOW}To enable advanced caching, create a new builder:${NC}"
        echo "  docker buildx create --use --driver docker-container --name easydel-builder"
        echo ""
        return 1
    fi

    return 0
}

print_usage() {
    echo "Usage: $0 [HARDWARE_TYPE] [TARGET] [OPTIONS]"
    echo ""
    echo "HARDWARE_TYPE: cpu, gpu, tpu (default: cpu)"
    echo "TARGET: production, development, test (default: production)"
    echo ""
    echo "Options:"
    echo "  --push              Push image to registry"
    echo "  --no-cache          Build without cache"
    echo "  --platform PLATFORM Docker platform (e.g., linux/amd64,linux/arm64)"
    echo "  --tag TAG           Custom tag for the image"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  VERSION             Version tag (default: git describe or 'latest')"
    echo "  PYTHON_VERSION      Python version (default: 3.11)"
    echo "  REGISTRY            Docker registry (default: ghcr.io/erfanzar)"
    echo "  IMAGE_NAME          Image name (default: easydel)"
}

# Parse additional options
PUSH=false
NO_CACHE=""
PLATFORM=""
CUSTOM_TAG=""

shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        --tag)
            CUSTOM_TAG="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate hardware type
if [[ ! "$HARDWARE_TYPE" =~ ^(cpu|gpu|tpu)$ ]]; then
    echo -e "${RED}Invalid hardware type: $HARDWARE_TYPE${NC}"
    print_usage
    exit 1
fi

# Validate target
if [[ ! "$TARGET" =~ ^(production|development|test)$ ]]; then
    echo -e "${RED}Invalid target: $TARGET${NC}"
    print_usage
    exit 1
fi

# Build tag
if [ -n "$CUSTOM_TAG" ]; then
    TAG="$CUSTOM_TAG"
else
    TAG="${IMAGE_NAME}:${VERSION}-${HARDWARE_TYPE}-${TARGET}"
fi

FULL_TAG="${REGISTRY}/${TAG}"

echo -e "${GREEN}Building EasyDeL Docker image${NC}"
echo "  Hardware: ${HARDWARE_TYPE}"
echo "  Target: ${TARGET}"
echo "  Tag: ${FULL_TAG}"
echo "  Python: ${PYTHON_VERSION}"
echo ""

# Build command
BUILD_CMD="docker buildx build \
    --build-arg HARDWARE_TYPE=${HARDWARE_TYPE} \
    --build-arg VERSION=${VERSION} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    --target ${TARGET} \
    --tag ${FULL_TAG} \
    ${NO_CACHE} \
    ${PLATFORM} \
    --progress=plain"

# Add cache configuration
if [ -z "$NO_CACHE" ]; then
    BUILD_CMD="${BUILD_CMD} \
        --cache-from type=registry,ref=${REGISTRY}/${IMAGE_NAME}:cache-${HARDWARE_TYPE} \
        --cache-to type=registry,ref=${REGISTRY}/${IMAGE_NAME}:cache-${HARDWARE_TYPE},mode=max"
fi

# Add build context
BUILD_CMD="${BUILD_CMD} ."

echo -e "${YELLOW}Executing build command:${NC}"
echo "$BUILD_CMD"
echo ""

# Execute build
eval $BUILD_CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"

    # Push if requested
    if [ "$PUSH" = true ]; then
        echo -e "${YELLOW}Pushing image to registry...${NC}"
        docker push ${FULL_TAG}

        # Also push latest tag for production builds
        if [ "$TARGET" = "production" ]; then
            LATEST_TAG="${REGISTRY}/${IMAGE_NAME}:latest-${HARDWARE_TYPE}"
            docker tag ${FULL_TAG} ${LATEST_TAG}
            docker push ${LATEST_TAG}
        fi

        echo -e "${GREEN}Push complete!${NC}"
    fi
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Print usage instructions
echo ""
echo -e "${GREEN}Image built successfully: ${FULL_TAG}${NC}"
echo ""
echo "To run the container:"
echo ""

case $TARGET in
    development)
        echo "  docker run -it --rm -v \$(pwd):/app ${FULL_TAG} bash"
        if [ "$HARDWARE_TYPE" = "gpu" ]; then
            echo "  # With GPU support:"
            echo "  docker run -it --rm --gpus all -v \$(pwd):/app ${FULL_TAG} bash"
        fi
        ;;
    test)
        echo "  docker run --rm ${FULL_TAG}"
        ;;
    production)
        echo "  docker run --rm ${FULL_TAG}"
        if [ "$HARDWARE_TYPE" = "gpu" ]; then
            echo "  # With GPU support:"
            echo "  docker run --rm --gpus all ${FULL_TAG}"
        fi
        ;;
esac

echo ""
echo "Or use Docker Compose:"
echo "  docker-compose up easydel-${HARDWARE_TYPE}$([ "$TARGET" != "production" ] && echo "-${TARGET}")"
