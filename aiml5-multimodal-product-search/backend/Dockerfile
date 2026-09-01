# syntax=docker/dockerfile:1.7
#
# Backend image: FastAPI + CLIP inference on CPU.
#
# Two stages. The builder installs dependencies into a virtualenv; the runtime
# copies only that venv and the application source. This keeps compilers and pip
# caches out of the final image and means a source-only change does not reinstall
# torch.
#
# Size note: torch is installed from PyTorch's CPU wheel index. The default
# (CUDA) build pulls ~2.5 GB of nvidia libraries that a CPU deployment never
# loads; the CPU wheel is ~200 MB. Override with TORCH_INDEX_URL to build a GPU
# image - see docs/architecture.md.

# ---------------------------------------------------------------- build stage
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed by a few wheels that lack manylinux builds.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch first, and in its own layer: it is the largest and least volatile
# dependency, so it stays cached across application dependency changes.
RUN pip install --upgrade pip setuptools wheel \
 && pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# -------------------------------------------------------------- runtime stage
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="Multimodal Product Search API" \
      org.opencontainers.image.description="CLIP-based multimodal product search over Qdrant." \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Keep model weights on a mounted volume so they survive container restarts
    # instead of being re-downloaded (~600 MB) on every boot.
    HF_HOME=/models \
    # Containers are usually CPU-limited; letting torch spawn one thread per host
    # core causes contention rather than throughput.
    OMP_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    APP_ENV=docker \
    LOG_FORMAT=json

# curl is required by the container healthcheck below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run unprivileged. Created before COPY so ownership can be set in one step.
RUN groupadd --gid 1001 app \
 && useradd --uid 1001 --gid app --create-home --shell /usr/sbin/nologin app \
 && mkdir -p /models /data \
 && chown -R app:app /models /data

WORKDIR /app
COPY --chown=app:app ./app ./app
COPY --chown=app:app pyproject.toml ./

USER app
EXPOSE 8000

# Liveness only: readiness (which requires the model to be loaded) is checked by
# compose's own dependency conditions and by /api/health/ready. Using liveness
# here means a slow first-time model download does not mark the container
# unhealthy and trigger a restart loop.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8000/api/health/live || exit 1

# Single worker by design: each worker would load its own copy of the model
# (~600 MB of weights). Scale by running more containers, or raise
# MAX_CONCURRENT_INFERENCES - see docs/architecture.md.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
