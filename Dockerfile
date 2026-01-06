# syntax=docker/dockerfile:1

# Bases
ARG HARDWARE_TYPE=cpu
ARG PYTHON_VERSION=3.13
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04 AS gpu-base
FROM ubuntu:22.04 AS tpu-base
FROM python:${PYTHON_VERSION}-slim AS cpu-base
FROM ghcr.io/astral-sh/uv:latest AS uv

# Final image chosen by HARDWARE_TYPE
FROM ${HARDWARE_TYPE}-base AS final
ARG PYTHON_VERSION=3.13
SHELL ["/bin/bash", "-lc"]

ARG HARDWARE_TYPE=cpu
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# OS deps + Python for Ubuntu-based images (GPU/TPU)
RUN set -eux; \
    if [[ "$HARDWARE_TYPE" != "cpu" ]]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            ca-certificates gnupg lsb-release software-properties-common tzdata && \
        ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev && \
        update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1; \
    fi; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git curl wget \
        libgomp1 rsync openssh-client sudo tmux screen netbase gnupg && \
    if [[ "$HARDWARE_TYPE" == "tpu" ]]; then \
        apt-get install -y --no-install-recommends docker.io; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# uv binary
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Virtualenv
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Project files
COPY pyproject.toml README.md ./
COPY easydel ./easydel

# Install project deps
RUN set -eux; \
    if [[ "$HARDWARE_TYPE" == "gpu" ]]; then \
        JAX_PLATFORMS=cpu uv pip install -e ".[gpu,torch,lm_eval,profile]"; \
    elif [[ "$HARDWARE_TYPE" == "tpu" ]]; then \
        JAX_PLATFORMS=cpu uv pip install -e ".[tpu,torch,lm_eval,profile]"; \
    else \
        uv pip install -e ".[torch,lm_eval,profile]"; \
    fi

# Ray with GCP extras
RUN uv pip install 'ray[default,gcp]==2.53.0'

# TPU-only: install gcloud
RUN set -eux; \
    if [[ "$HARDWARE_TYPE" == "tpu" ]]; then \
        curl -fsSL https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -o /tmp/google-cloud-sdk.tar.gz; \
        mkdir -p /usr/local/gcloud; \
        tar -C /usr/local/gcloud -xzf /tmp/google-cloud-sdk.tar.gz; \
        /usr/local/gcloud/google-cloud-sdk/install.sh --quiet; \
        rm -f /tmp/google-cloud-sdk.tar.gz; \
    fi

# Non-root user (Ray prefers non-root)
RUN useradd -m -s /bin/bash easydel && \
    echo "easydel ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R easydel:easydel /app && \
    if [[ "$HARDWARE_TYPE" == "tpu" ]]; then usermod -aG docker easydel; fi

# Shell quieting for rsync/ssh
ENV HOME=/home/easydel
RUN mkdir -p "$HOME" && \
    touch "$HOME/.bashrc" && \
    echo 'if [ -z "$PS1" ]; then return; fi' >> "$HOME/.bashrc" && \
    touch "$HOME/.hushlogin" && \
    chown -R easydel:easydel "$HOME"

# Helpful env
ENV PYTHONPATH=/app:/app/easydel:. \
    RAY_USAGE_STATS_ENABLED=0 \
    TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60 \
    TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES=1024 \
    PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin

USER easydel
WORKDIR /app

CMD ["bash"]
