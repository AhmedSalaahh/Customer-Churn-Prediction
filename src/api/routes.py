"""
src/api/routes.py
------------------
All HTTP route handlers, grouped by concern.
Each handler is thin: validate (Pydantic) → call inference → return schema.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.inference import get_bundle, predict_batch, predict_single
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    SinglePredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Dependency — pull config from app state
# ──────────────────────────────────────────────────────────────────────────────

def get_features_cfg(request: Request) -> dict:
    return request.app.state.features_cfg


def get_threshold(request: Request) -> float:
    return request.app.state.threshold


# ──────────────────────────────────────────────────────────────────────────────
# Health & info
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Liveness check",
)
def health(request: Request):
    """Returns 200 when the API is running and the model is loaded."""
    try:
        bundle = get_bundle()
        loaded = True
        version = f"features={bundle.feature_count}"
    except RuntimeError:
        loaded = False
        version = "not loaded"

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=version,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Operations"],
    summary="Model metadata",
)
def model_info(request: Request):
    """Returns feature names, threshold, and risk tier definitions."""
    try:
        bundle = get_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    from src.api.inference import RISK_TIERS

    return ModelInfoResponse(
        feature_count=bundle.feature_count,
        feature_names=bundle.feature_names,
        threshold=request.app.state.threshold,
        risk_tiers={
            tier: {"min": lo, "max": hi}
            for tier, (lo, hi) in RISK_TIERS.items()
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prediction endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=SinglePredictResponse,
    tags=["Prediction"],
    summary="Score a single customer",
    status_code=status.HTTP_200_OK,
)
def predict(
    request: Request,
    body,           # CustomerFeatures — imported below to avoid circular dep
    features_cfg: dict = Depends(get_features_cfg),
    threshold: float = Depends(get_threshold),
):
    """
    Accepts one customer record and returns a churn probability,
    binary prediction, risk tier, and expected revenue loss.
    """
    try:
        result = predict_single(
            customer=body.model_dump(),
            features_cfg=features_cfg,
            threshold=threshold,
        )
        return SinglePredictResponse(prediction=PredictionResult(**result))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Score multiple customers",
    status_code=status.HTTP_200_OK,
)
def predict_batch_endpoint(
    request: Request,
    body: BatchPredictRequest,
    features_cfg: dict = Depends(get_features_cfg),
    threshold: float = Depends(get_threshold),
):
    """
    Accepts up to 1 000 customer records and returns predictions for all.
    Vectorised — much faster than calling /predict in a loop.
    """
    try:
        raw_results = predict_batch(
            customers=[c.model_dump() for c in body.customers],
            features_cfg=features_cfg,
            threshold=threshold,
        )
        predictions = [PredictionResult(**r) for r in raw_results]
        return BatchPredictResponse(total=len(predictions), predictions=predictions)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Fix the type annotation after import to avoid circular reference ──────────
from src.api.schemas import CustomerFeatures  # noqa: E402
predict.__annotations__["body"] = CustomerFeatures
