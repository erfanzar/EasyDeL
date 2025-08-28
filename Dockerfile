# syntax=docker/dockerfile:1
ARG HARDWARE_TYPE=cpu
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS gpu-base
FROM ubuntu:22.04 AS tpu-base
FROM python:3.11-slim AS cpu-base

FROM ${HARDWARE_TYPE}-base AS final
SHELL ["/bin/bash", "-lc"]

ARG HARDWARE_TYPE=cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN set -eux; \
    if [[ "$HARDWARE_TYPE" != "cpu" ]]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            ca-certificates gnupg lsb-release software-properties-common tzdata && \
        ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.11 python3.11-venv python3.11-dev python3.11-distutils python3-pip && \
        update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1; \
    fi; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git curl wget \
        libgomp1 rsync openssh-client sudo tmux screen netbase gnupg && \
    # TPU needs Docker engine inside to manage TPU slices
    if [[ "$HARDWARE_TYPE" == "tpu" ]]; then \
        apt-get install -y --no-install-recommends docker.io; \
    fi; \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
RUN uv venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/cuda/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY easydel ./easydel

RUN set -eux; \
    if [[ "$HARDWARE_TYPE" == "gpu" ]]; then \
        echo "Installing GPU deps"; \
        JAX_PLATFORMS=cpu uv pip install --no-strict -e ".[gpu,torch,lm_eval,profile]"; \
    elif [[ "$HARDWARE_TYPE" == "tpu" ]]; then \
        echo "Installing TPU deps"; \
        JAX_PLATFORMS=cpu uv pip install --no-strict -e ".[tpu,torch,lm_eval,profile]"; \
    else \
        echo "Installing CPU deps"; \
        uv pip install -e ".[torch,lm_eval,profile]"; \
    fi

RUN uv pip install --no-strict 'ray[default,gcp]==2.34.0'

RUN set -eux; \
    if [[ "$HARDWARE_TYPE" == "tpu" ]]; then \
        git clone https://github.com/dlwh/ray.git /tmp/ray --branch tpu_docker_2.34 --depth 1 && \
        py_site=$(python - <<'PY'\nimport site; print(next(p for p in site.getsitepackages() if "site-packages" in p)) )\nPY) && \
        cp /tmp/ray/python/ray/autoscaler/_private/gcp/tpu_command_runner.py \
           "$py_site/ray/autoscaler/_private/gcp/tpu_command_runner.py" && \
        rm -rf /tmp/ray && \
        curl -fsSL https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -o /tmp/google-cloud-sdk.tar.gz && \
        mkdir -p /usr/local/gcloud && \
        tar -C /usr/local/gcloud -xzf /tmp/google-cloud-sdk.tar.gz && \
        /usr/local/gcloud/google-cloud-sdk/install.sh --quiet && \
        rm -f /tmp/google-cloud-sdk.tar.gz; \
    fi

RUN useradd -m -s /bin/bash easydel && \
    echo "easydel ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R easydel:easydel /app && \
    if [[ "$HARDWARE_TYPE" == "tpu" ]]; then usermod -aG docker easydel; fi

ENV HOME=/home/easydel
RUN mkdir -p "$HOME" && \
    touch "$HOME/.bashrc" && \
    echo 'if [ -z "$PS1" ]; then return; fi' >> "$HOME/.bashrc" && \
    touch "$HOME/.hushlogin" && \
    chown -R easydel:easydel "$HOME"

ENV PYTHONPATH=/app:/app/easydel:. \
    RAY_USAGE_STATS_ENABLED=0 \
    TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60 \
    TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES=1024 \
    PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin

USER easydel
WORKDIR /app

CMD ["bash"]