"""
src/api/app.py
---------------
FastAPI application factory.

Design choices:
- lifespan context manager loads the model ONCE at startup
  (not on every request — that would be catastrophically slow)
- config is stored on app.state so routes can read it via Request
- no global variables in routes — everything flows through dependency injection
"""

import logging
import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.inference import load_bundle
from src.api.routes import router

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan (replaces deprecated @app.on_event)
# ──────────────────────────────────────────────────────────────────────────────

def make_lifespan(config_path: str, model_path: str, threshold: float):
    """
    Returns an async context manager that:
    - loads config + model at startup
    - stores them on app.state for route access
    - logs readiness
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ── Startup ──────────────────────────────
        logger.info("Starting up Churn Prediction API...")

        cfg = _load_config(config_path)
        app.state.features_cfg = cfg["features"]
        app.state.threshold = threshold

        load_bundle(model_path)

        logger.info(
            "Ready — model: %s | threshold: %.2f", model_path, threshold
        )

        yield  # ← application runs here

        # ── Shutdown ─────────────────────────────
        logger.info("Shutting down Churn Prediction API")

    return lifespan


# ──────────────────────────────────────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────────────────────────────────────

def create_app(
    config_path: str = "configs/config.yaml",
    model_path: str = "outputs/best_model.pkl",
    threshold: float = 0.5,
) -> FastAPI:
    """
    Build and return the FastAPI application.

    Parameters are read from environment variables first,
    then fall back to the supplied defaults — making Docker/CI easy.

    Environment variables:
      CONFIG_PATH   — path to config.yaml
      MODEL_PATH    — path to best_model.pkl
      THRESHOLD     — float probability cutoff (default 0.5)
    """
    config_path = os.getenv("CONFIG_PATH", config_path)
    model_path  = os.getenv("MODEL_PATH",  model_path)
    threshold   = float(os.getenv("THRESHOLD", str(threshold)))

    app = FastAPI(
        title="Customer Churn Prediction API",
        description=(
            "Predicts the probability that a telco customer will churn. "
            "Powers targeted retention campaigns by surfacing high-risk customers "
            "before they leave.\n\n"
            "**Stage 2** of the ML deployment pipeline: "
            "Notebook → Scripts → **FastAPI** → Docker → CI/CD → AWS"
        ),
        version="1.0.0",
        lifespan=make_lifespan(config_path, model_path, threshold),
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Permissive for now; lock down origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── Root redirect ────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    def root():
        return JSONResponse({"message": "Churn API — see /docs"})

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Entry point for uvicorn
# ──────────────────────────────────────────────────────────────────────────────

# uvicorn src.api.app:app --reload
app = create_app()
