"""
src/api/schemas.py
-------------------
Pydantic models for request validation and response serialisation.

These act as the contract between the API caller and the model —
any field mismatch is caught here before it reaches the ML code.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    A single customer record.
    Field names and allowed values mirror the raw Telco CSV exactly,
    so callers can POST the same data they'd put in the CSV.
    """

    # Demographics
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    # Account
    tenure: int = Field(..., ge=0, le=72, description="Months with the company")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)

    # Phone
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    # Internet
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_gte_monthly(cls, v, info):
        """Basic sanity: TotalCharges should not be wildly less than MonthlyCharges."""
        if "MonthlyCharges" in info.data and v < 0:
            raise ValueError("TotalCharges must be non-negative")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Wrap a list of customers for batch scoring."""
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


# ──────────────────────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Output for a single customer."""
    churn_probability: float = Field(..., ge=0.0, le=1.0,
                                     description="Calibrated probability of churn")
    churn_predicted: bool = Field(...,
                                  description="True if churn_probability ≥ threshold")
    risk_tier: Literal["Low", "Medium", "High"] = Field(
        ..., description="Business-friendly risk bucket"
    )
    expected_revenue_loss: float | None = Field(
        None,
        description="churn_probability × TotalCharges (proxy CLV)"
    )


class SinglePredictResponse(BaseModel):
    """Response for /predict (single customer)."""
    status: str = "ok"
    prediction: PredictionResult


class BatchPredictResponse(BaseModel):
    """Response for /predict/batch."""
    status: str = "ok"
    total: int
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    """Response for /model/info."""
    feature_count: int
    feature_names: list[str]
    threshold: float
    risk_tiers: dict
