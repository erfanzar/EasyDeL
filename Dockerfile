# syntax=docker/dockerfile:1
FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 \
	git \
	build-essential && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.2
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./ 

RUN poetry config virtualenvs.in-project true && poetry install --no-interaction --no-ansi --only main --no-root

# ----------------------------------
# Runtime Stage  
# ----------------------------------
FROM python:3.10-slim as runtime

WORKDIR /app

COPY --from=builder /app /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	PATH="/app/.venv/bin:$PATH" \
	PYTHONPATH="/app/.venv/lib/python3.10/site-packages:$PYTHONPATH"

RUN python -c "import jax; print(f'JAX version: {jax.__version__}')" && \
	python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

ARG VERSION
ENV VERSION=$VERSION

LABEL org.opencontainers.image.version=$VERSION \
	org.opencontainers.image.description="EasyDeL: An open-source library to make training faster and more optimized in JAX" \
	org.opencontainers.image.source="https://github.com/erfanzar/EasyDeL"
