# Dockerfile for EasyDeL using uv package manager
# Use CUDA base image for GPU support, TPU base, or python:slim for CPU
ARG HARDWARE_TYPE=cpu
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS gpu-base
FROM ubuntu:22.04 AS tpu-base
FROM python:3.11-slim AS cpu-base

# ============= Base Build Stage =============
# Select base image based on hardware type
FROM ${HARDWARE_TYPE}-base AS base

# Re-declare ARG after FROM
ARG HARDWARE_TYPE=cpu

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 if using CUDA or TPU base image
RUN if [ -f /etc/apt/sources.list.d/cuda.list ] || [ ! -f /usr/bin/python ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            software-properties-common \
            tzdata && \
        ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
        echo $TZ > /etc/timezone && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.11 \
            python3.11-venv \
            python3.11-dev \
            python3-pip && \
        update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY easydel ./easydel

# Create and activate virtual environment
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the project with dependencies using uv
RUN if [ "$HARDWARE_TYPE" = "gpu" ]; then \
        echo "Installing GPU dependencies..." && \
        uv pip install --no-strict -e ".[gpu,torch,lm_eval,profile]"; \
    elif [ "$HARDWARE_TYPE" = "tpu" ]; then \
        echo "Installing TPU dependencies..." && \
        uv pip install --no-strict -e ".[tpu,torch,lm_eval,profile]"; \
    else \
        echo "Installing CPU dependencies..." && \
        uv pip install -e ".[torch,lm_eval,profile]"; \
    fi

# Set Python to use unbuffered output
ENV PYTHONUNBUFFERED=1

# Add NVIDIA libraries to LD_LIBRARY_PATH for GPU version
RUN if [ -f /etc/apt/sources.list.d/cuda.list ]; then \
        echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
        echo "/usr/local/cuda/extras/CUPTI/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
        ldconfig; \
    fi

# Set environment based on hardware type
ENV HARDWARE_TYPE=${HARDWARE_TYPE}

# Set CUDA environment for GPU
ENV PATH=/usr/local/cuda/bin:${PATH}
# Don't set LD_LIBRARY_PATH as it interferes with JAX's CUDA detection
# See: https://github.com/jax-ml/jax/issues/29843

# TPU configuration is handled at runtime, not build time
# TPUs will be configured through environment variables when running the container

# ============= Production Stage =============
FROM base AS production

# Verify installation
RUN python -c "import easydel; print('EasyDeL installed successfully')"

CMD ["python"]

# ============= Development Stage =============
FROM production AS development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN uv pip install --no-strict \
    ipython \
    ipdb \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy

# Set development environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEVELOPMENT=1

CMD ["bash"]

# ============= Test Stage =============
FROM production AS test

# Install test dependencies
RUN uv pip install --no-strict \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-timeout \
    hypothesis

# Copy test files if they exist
COPY --chown=root:root tests* ./tests/

# Run tests by default
CMD ["python", "-m", "pytest", "-v", "--tb=short"]

# ============= Final stage selector =============
# Default to production if not specified
FROM production AS final
