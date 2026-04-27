# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — builder
# Install all dependencies into a venv so only the venv is copied to the
# final image. This keeps the production image lean and free of build tools.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed to compile C extensions (e.g. greenlet)
# These stay in the builder only — not in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# so re-builds after only code changes skip pip entirely
COPY requirements.txt .

# Create a venv inside /build/venv and install there
RUN python -m venv /build/venv \
    && /build/venv/bin/pip install --upgrade pip \
    && /build/venv/bin/pip install --no-cache-dir -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — runtime
# Minimal image: no compiler, no build tools, just the app + venv
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home appuser

WORKDIR /app

# Copy only the installed venv from the builder (no compiler baggage)
COPY --from=builder /build/venv /app/venv

# Copy application code
COPY configs/   configs/
COPY src/       src/
COPY scripts/   scripts/

# Create outputs directory for the model artefact
# In production this is mounted from a volume or S3
RUN mkdir -p outputs && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Make the venv the default Python
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # These can be overridden at `docker run` time with -e
    CONFIG_PATH="configs/config.yaml" \
    MODEL_PATH="outputs/best_model.pkl" \
    THRESHOLD="0.5"

EXPOSE 8000

# Health check — Docker marks the container unhealthy if this fails
# Useful for docker-compose depends_on and ECS health checks
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

# Production server: 4 workers, preloaded model, access log to stdout
CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
