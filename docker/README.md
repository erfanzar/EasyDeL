# EasyDeL Docker Setup

This directory contains Docker configurations for running EasyDeL with different hardware accelerators (CPU, CUDA, TPU).

## Prerequisites

### For CUDA Support

- NVIDIA CUDA-capable hardware (CUDA 12.8)
- NVIDIA drivers installed on host system
- Docker installed

### For TPU Support

- Access to Google Cloud TPUs
- TPU configuration credentials

## Building Docker Images

### Build for CUDA (12.8)

```bash
# Using the build script
./docker/build.sh cuda

# Or directly with docker
sudo docker build --build-arg HARDWARE_TYPE=cuda -t easydel:cuda .
```

### Build for TPU

```bash
# Using the build script
./docker/build.sh tpu

# Or directly with docker
sudo docker build --build-arg HARDWARE_TYPE=tpu -t easydel:tpu .
```

### Build for CPU

```bash
# Using the build script
./docker/build.sh cpu

# Or directly with docker
sudo docker build --build-arg HARDWARE_TYPE=cpu -t easydel:cpu .
```

## Running Containers

### CUDA Container

#### Using Docker Compose

```bash
# Start CUDA container with all necessary mounts
docker compose -f docker-compose.cuda.yml up

# Run in detached mode
docker compose -f docker-compose.cuda.yml up -d

# Execute commands in running container
docker compose -f docker-compose.cuda.yml exec easydel-cuda python your_script.py
```

#### Using Docker Run

```bash
# Sanity Check

docker compose -f docker-compose.cuda.yml exec easydel-cuda python -c "import jax; print('Devices:', jax.devices())"

# Basic CUDA container
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    easydel:cuda

# With all CUDA capabilities
docker run --gpus all --privileged \
    --network host \
    --ipc host \
    -v $(pwd):/workspace \
    -v /dev:/dev \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    easydel:cuda
```

### TPU Container

- Using Docker Compose

```bash
# Start TPU container
docker compose -f docker-compose.tpu.yml up

# Run in detached mode
sudo docker compose -f docker-compose.tpu.yml up -d
sudo docker compose -f docker-compose.tpu.yml run --rm easydel-tpu python -c "import jax; print('Devices:', jax.devices())"
```

- Using Docker Run

```bash
# TPU container with auto-detection
docker run -it --rm \
    --network host \
    --privileged \
    -v $(pwd):/workspace \
    -v ~/.config/gcloud:/root/.config/gcloud:ro \
    easydel:tpu

# TPU container with explicit configuration
docker run -it --rm \
    --network host \
    --privileged \
    -e TPU_NAME=your-tpu-name \
    -e TPU_ZONE=us-central1-a \
    -v $(pwd):/workspace \
    easydel:tpu
```

### CPU Container

```bash
# Using docker-compose
docker-compose up

# Using docker run
docker run -it --rm -v $(pwd):/workspace easydel:cpu
```

## Development Mode

For development work with additional tools:

```bash
# Build development image
docker build --target development --build-arg HARDWARE_TYPE=cuda -t easydel:dev-cuda .

# Run with development tools
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    easydel:dev-cuda bash
```

## Testing

Run tests in containerized environment:

```bash
# Build test image
docker build --target test --build-arg HARDWARE_TYPE=cuda -t easydel:test-cuda .

# Run tests
docker run --gpus all --rm easydel:test-cuda
```

## Multi-Stage Build Targets

The Dockerfile supports multiple build stages:

- **base**: Base image with Python and system dependencies
- **production**: Production-ready image with EasyDeL installed
- **development**: Includes development tools (vim, ipython, pytest, etc.)
- **test**: Configured for running tests

## Hardware-Specific Features

### CUDA Features

- CUDA 12.8 with cuDNN support
- Automatic CUDA library detection
- JAX backend configured for CUDA devices
- PyTorch with CUDA support

### TPU Features

- TPU runtime libraries
- JAX TPU backend configured
- Automatic TPU detection when running on GCP
- PyTorch XLA support

## Troubleshooting

### CUDA Not Detected

1. Ensure NVIDIA drivers are installed: `nvidia-smi`
2. Check Docker CUDA support: `docker run --gpus all nvidia/cuda:12.8.0-base nvidia-smi`
3. Verify JAX can see CUDA devices:

   ```python
   import jax
   print(jax.devices())  # Should show CudaDevice(id=0)
   ```

### TPU Connection Issues

1. Ensure you're authenticated with gcloud: `gcloud auth login`
2. Check TPU is accessible: `gcloud compute tpus list --zone=your-zone`
3. Verify JAX TPU backend:

   ```python
   import jax
   print(jax.devices())  # Should show TpuDevice entries
   ```

### Build Cache

To speed up builds, the Dockerfile uses BuildKit cache mounts. Enable BuildKit:

```bash
export DOCKER_BUILDKIT=1
docker build --build-arg HARDWARE_TYPE=cuda -t easydel:cuda .
```

## Environment Variables

### Common

- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered
- `HARDWARE_TYPE`: Set to `cpu`, `cuda`, or `tpu`

### CUDA-Specific

- `CUDA_VISIBLE_DEVICES`: Control which CUDA devices are visible
- `JAX_CUDA_VERSION`: Override CUDA version for JAX

### TPU-Specific

- `TPU_NAME`: TPU instance name (auto-detected if not set)
- `TPU_ZONE`: GCP zone where TPU is located
- `JAX_PLATFORM_NAME`: Set to `tpu` for TPU execution

## Quick Start Examples

### Train a Model on CUDA

```bash
docker compose -f docker-compose.cuda.yml run --rm easydel-cuda \
    python train.py --config configs/training.yaml
```

### Run Inference on TPU

```bash
docker compose -f docker-compose.tpu.yml run --rm easydel-tpu \
    python inference.py --model-path /workspace/models/my_model
```

### Interactive Development

```bash
# CUDA development
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    easydel:dev-cuda \
    jupyter lab --ip=0.0.0.0 --allow-root

# TPU development
docker run -it --rm \
    --network host \
    --privileged \
    -v $(pwd):/workspace \
    easydel:dev-tpu \
    ipython
```

## Notes

- The CUDA image requires CUDA 12.8 compatible hardware
- TPU support requires Google Cloud credentials and TPU access
- Images default to Python 3.13 (override with `PYTHON_VERSION`, e.g. `PYTHON_VERSION=3.12 ./docker/build.sh cpu`)
- Ray is pinned to 2.53.0 (`ray[default,gcp]` installed in Docker images)
- Dependencies are installed with `uv` (`JAX_PLATFORMS=cpu` is set during CUDA/TPU builds to avoid backend detection at build time)
