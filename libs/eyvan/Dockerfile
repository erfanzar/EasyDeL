# Use libs/eyvan as the build context. Every image argument is copied
# verbatim from oci/release-lock.json after validate-policy succeeds.
# Standalone: docker buildx build -f Dockerfile [required build args] .
# Workspace:  docker buildx build -f libs/eyvan/Dockerfile [required build args] libs/eyvan
ARG TARGETPLATFORM
ARG EYVAN_NODE_IMAGE
ARG EYVAN_UV_IMAGE
ARG EYVAN_RUNTIME_IMAGE

FROM --platform=$TARGETPLATFORM ${EYVAN_NODE_IMAGE} AS dashboard
ARG TARGETOS
ARG TARGETARCH
RUN test "$TARGETOS" = linux && test "$TARGETARCH" = amd64
WORKDIR /dashboard
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
# OIDC issuer/client configuration is fetched from /api/v1/dashboard/config at
# runtime. Never bake a deployment identity into this reusable image.
RUN npm run build

FROM --platform=$TARGETPLATFORM ${EYVAN_UV_IMAGE} AS uv

FROM --platform=$TARGETPLATFORM ${EYVAN_RUNTIME_IMAGE} AS runtime
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG EYVAN_SOURCE
ARG EYVAN_REVISION
ARG EYVAN_VERSION
RUN test "$TARGETOS" = linux && test "$TARGETARCH" = amd64 && \
    test "$TARGETPLATFORM" = linux/amd64 && \
    printf '%s' "$EYVAN_REVISION" | grep -Eq '^[0-9a-f]{40}$' && \
    printf '%s' "$EYVAN_SOURCE" | grep -Eq '^https://github\.com/[^/]+/[^/]+$' && \
    printf '%s' "$EYVAN_VERSION" | grep -Eq '^[0-9]+([.][0-9A-Za-z]+)+([-+][0-9A-Za-z.-]+)?$'
LABEL org.opencontainers.image.source="${EYVAN_SOURCE}" \
      org.opencontainers.image.revision="${EYVAN_REVISION}" \
      org.opencontainers.image.version="${EYVAN_VERSION}" \
      org.opencontainers.image.title="Eyvan control plane" \
      org.opencontainers.image.description="Policy-aware global fleet control plane for CPU, GPU, and TPU capacity" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"
COPY --from=uv /uv /uvx /bin/
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    EYVAN_DASHBOARD_DIR=/app/dashboard/dist
WORKDIR /app
COPY requirements-docker.txt requirements-build.txt ./
# The interpreter is pinned, not resolved. `uv venv` alone takes the newest uv
# knows about -- which became 3.14, outside this package's own >=3.10,<3.14 --
# and the hashed exports carry wheels for supported versions only, so the build
# fell through to compiling httptools from source in an image with no compiler.
RUN uv venv --python 3.13 "$VIRTUAL_ENV" && \
    uv pip install --require-hashes -r requirements-docker.txt && \
    uv pip install --require-hashes -r requirements-build.txt
COPY pyproject.toml README.md LICENSE ./
COPY eyvan ./eyvan
RUN uv pip install --no-build-isolation --no-deps . && \
    uv pip uninstall uv-build
COPY --from=dashboard /dashboard/dist /app/dashboard/dist
EXPOSE 8080
ENTRYPOINT ["eyvan-server"]
# A network bind refuses to start without TLS material (EYVAN_TLS_CERT_FILE /
# EYVAN_TLS_KEY_FILE) or the explicit EYVAN_INSECURE_PLAINTEXT_BIND=1 opt-in;
# see docs/security.md "Exposure".
CMD ["--host", "0.0.0.0", "--port", "8080"]
