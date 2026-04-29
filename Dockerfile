# ══════════════════════════════════════════════════════════════════
# Stage 1 — builder: compile C extensions only
# ══════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build wheels for every package (compiled, portable, no source needed)
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ══════════════════════════════════════════════════════════════════
# Stage 2 — runtime: install from pre-built wheels (no compiler needed)
# ══════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home appuser

WORKDIR /app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /build/wheels /wheels

# Install from wheels into the FINAL location — shebangs point here correctly
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY configs/   configs/
COPY src/       src/
COPY scripts/   scripts/

RUN mkdir -p outputs && chown -R appuser:appgroup /app

USER appuser

ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH="configs/config.yaml" \
    MODEL_PATH="outputs/best_model.pkl" \
    THRESHOLD="0.5"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]