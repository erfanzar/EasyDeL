# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 \
	git \
	build-essential \
	curl && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Build argument for hardware type (cpu, gpu, tpu)
ARG HARDWARE_TYPE=cpu

# Create virtual environment and install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
	uv venv /app/.venv && \
	if [ "$HARDWARE_TYPE" = "gpu" ]; then \
		uv sync --frozen --no-install-project --no-dev --extra gpu; \
	elif [ "$HARDWARE_TYPE" = "tpu" ]; then \
		uv sync --frozen --no-install-project --no-dev --extra tpu; \
	else \
		uv sync --frozen --no-install-project --no-dev; \
	fi

# ----------------------------------
# Runtime Stage  
# ----------------------------------
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PATH="/app/.venv/bin:$PATH" \
	VIRTUAL_ENV="/app/.venv"

# Copy source code
COPY . .

# Install the project itself
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv pip install -e . --no-deps

# Verify installations based on hardware type
RUN python -c "import jax; print(f'JAX version: {jax.__version__}')" && \
	python -c "import easydel; print('EasyDeL imported successfully')" && \
	if [ "$HARDWARE_TYPE" = "gpu" ] || [ "$HARDWARE_TYPE" = "tpu" ]; then \
		python -c "import torch; print(f'PyTorch version: {torch.__version__}')"; \
	fi

ARG VERSION
ARG HARDWARE_TYPE=cpu
ENV VERSION=$VERSION \
	HARDWARE_TYPE=$HARDWARE_TYPE

LABEL org.opencontainers.image.version=$VERSION \
	org.opencontainers.image.description="EasyDeL: An open-source library to make training faster and more optimized in JAX" \
	org.opencontainers.image.source="https://github.com/dvruette/EasyDeL"

ENTRYPOINT ["/usr/bin/env"]
