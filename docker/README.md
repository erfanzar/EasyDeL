# EasyDeL Docker Setup

This directory contains Docker configurations for running EasyDeL with different hardware accelerators (CPU, GPU, TPU).

## Prerequisites

### For GPU Support

- NVIDIA GPU with CUDA 12.8 support
- NVIDIA drivers installed on host system
- Docker installed

### For TPU Support

- Access to Google Cloud TPUs
- TPU configuration credentials

## Building Docker Images

### Build for GPU (CUDA 12.8)

```bash
# Using the build script
./docker/build.sh gpu

# Or directly with docker
sudo docker build --build-arg HARDWARE_TYPE=gpu -t easydel:gpu-cuda12.8 .
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

### GPU Container

#### Using Docker Compose

```bash
# Start GPU container with all necessary mounts
docker compose -f docker-compose.gpu.yml up

# Run in detached mode
docker compose -f docker-compose.gpu.yml up -d

# Execute commands in running container
docker compose -f docker-compose.gpu.yml exec easydel-gpu python your_script.py
```

#### Using Docker Run

```bash
# Sanity Check

docker compose -f docker-compose.gpu.yml exec easydel-gpu python -c "import jax; print('Devices:', jax.devices())"

# Basic GPU container
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    easydel:gpu-cuda12.8

# With all GPU capabilities
docker run --gpus all --privileged \
    --network host \
    --ipc host \
    -v $(pwd):/workspace \
    -v /dev:/dev \
    -v /usr/local/cuda:/usr/local/cuda:ro \
    easydel:gpu-cuda12.8
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
docker build --target development --build-arg HARDWARE_TYPE=gpu -t easydel:dev-gpu .

# Run with development tools
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    easydel:dev-gpu bash
```

## Testing

Run tests in containerized environment:

```bash
# Build test image
docker build --target test --build-arg HARDWARE_TYPE=gpu -t easydel:test-gpu .

# Run tests
docker run --gpus all --rm easydel:test-gpu
```

## Multi-Stage Build Targets

The Dockerfile supports multiple build stages:

- **base**: Base image with Python and system dependencies
- **production**: Production-ready image with EasyDeL installed
- **development**: Includes development tools (vim, ipython, pytest, etc.)
- **test**: Configured for running tests

## Hardware-Specific Features

### GPU Features

- CUDA 12.8 with cuDNN support
- Automatic CUDA library detection
- JAX GPU backend configured
- PyTorch with CUDA support

### TPU Features

- TPU runtime libraries
- JAX TPU backend configured
- Automatic TPU detection when running on GCP
- PyTorch XLA support

## Troubleshooting

### GPU Not Detected

1. Ensure NVIDIA drivers are installed: `nvidia-smi`
2. Check Docker GPU support: `docker run --gpus all nvidia/cuda:12.8.0-base nvidia-smi`
3. Verify JAX can see GPU:

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
docker build --build-arg HARDWARE_TYPE=gpu -t easydel:gpu-cuda12.8 .
```

## Environment Variables

### Common

- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered
- `HARDWARE_TYPE`: Set to `cpu`, `gpu`, or `tpu`

### GPU-Specific

- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible
- `JAX_CUDA_VERSION`: Override CUDA version for JAX

### TPU-Specific

- `TPU_NAME`: TPU instance name (auto-detected if not set)
- `TPU_ZONE`: GCP zone where TPU is located
- `JAX_PLATFORM_NAME`: Set to `tpu` for TPU execution

## Quick Start Examples

### Train a Model on GPU

```bash
docker compose -f docker-compose.gpu.yml run --rm easydel-gpu \
    python train.py --config configs/gpu_training.yaml
```

### Run Inference on TPU

```bash
docker compose -f docker-compose.tpu.yml run --rm easydel-tpu \
    python inference.py --model-path /workspace/models/my_model
```

### Interactive Development

```bash
# GPU development
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    easydel:dev-gpu \
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

- The GPU image requires CUDA 12.8 compatible hardware
- TPU support requires Google Cloud credentials and TPU access
- All images are based on Python 3.11
- Dependencies are managed with `uv` package manager for fast installation
- The `--no-strict` flag is used for dependency resolution to handle JAX/PyTorch compatibility
