"""
evaluation/metrics.py
----------------------
Classification metrics, SHAP explainability, and business simulations.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test,
    model_name: str = "Model",
) -> dict:
    """
    Return a dict of key classification metrics.
    Logs a formatted report to the console.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n── {model_name} ──────────────────────────────")
    print(classification_report(y_test, y_pred))
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    return {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "classification_report": report,
        "y_proba": y_proba,
    }


def plot_pr_curves(models: dict, X_test: np.ndarray, y_test, save_path: Optional[str] = None):
    """Plot precision-recall curves for all models in the dict."""
    plt.figure()
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("PR curve saved to %s", save_path)
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# SHAP explainability
# ──────────────────────────────────────────────

def compute_shap_importance(
    model,                   # calibrated sklearn model (must have .estimator)
    X_test: np.ndarray,
    feature_names: list,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| values using TreeExplainer on the underlying
    XGBoost (or RF) estimator inside a CalibratedClassifierCV.

    Returns a DataFrame sorted by importance.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Skipping SHAP computation.")
        return pd.DataFrame()

    # Unwrap calibrated model → base estimator
    base = getattr(model, "estimator", model)

    explainer = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(X_test)

    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Summary plot
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, show=False
    )
    plt.title("SHAP Feature Importance")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("SHAP plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()

    return df


# ──────────────────────────────────────────────
# Business simulations
# ──────────────────────────────────────────────

def intervention_simulation(
    model,
    X_test: np.ndarray,
    y_test,
    threshold: float = 0.6,
    retention_effectiveness: float = 0.4,
    retention_cost: float = 50.0,
    revenue_per_customer: float = 500.0,
) -> dict:
    """
    Estimate P&L of a targeted retention campaign.

    Returns a results dict and prints a formatted summary.
    """
    proba = model.predict_proba(X_test)[:, 1]
    targeted = proba >= threshold

    n_targeted = int(targeted.sum())
    true_churn_targeted = int(y_test[targeted].sum())
    churn_prevented = true_churn_targeted * retention_effectiveness

    revenue_saved = churn_prevented * revenue_per_customer
    campaign_cost = n_targeted * retention_cost
    net_profit = revenue_saved - campaign_cost

    results = {
        "n_targeted": n_targeted,
        "churn_prevented": int(churn_prevented),
        "revenue_saved": revenue_saved,
        "campaign_cost": campaign_cost,
        "net_profit": net_profit,
    }

    print("\nIntervention Simulation Results")
    print("────────────────────────────────")
    print(f"  Customers targeted  : {n_targeted:,}")
    print(f"  Churn prevented     : {int(churn_prevented):,}")
    print(f"  Revenue saved       : ${revenue_saved:,.0f}")
    print(f"  Campaign cost       : ${campaign_cost:,.0f}")
    print(f"  Net profit          : ${net_profit:,.0f}")

    return results


def retention_strategy_engine(
    model,
    X_test: np.ndarray,
    X_test_raw: pd.DataFrame,    # original feature df (for TotalCharges)
    y_test,
    budget: float = 20_000.0,
    retention_cost: float = 50.0,
    retention_success_rate: float = 0.5,
) -> pd.DataFrame:
    """
    Greedy customer selection under a budget constraint,
    ranked by expected net value (expected_savings - intervention_cost).

    Returns the selected customer DataFrame.
    """
    proba = model.predict_proba(X_test)[:, 1]
    df = X_test_raw.copy()
    df["churn_probability"] = proba
    df["customer_value"] = df["TotalCharges"]
    df["expected_revenue_loss"] = df["churn_probability"] * df["customer_value"]
    df["expected_savings"] = df["expected_revenue_loss"] * retention_success_rate
    df["intervention_cost"] = retention_cost
    df["net_expected_value"] = df["expected_savings"] - df["intervention_cost"]
    df = df.sort_values("net_expected_value", ascending=False)

    selected, total_cost = [], 0.0
    for _, row in df.iterrows():
        if total_cost + retention_cost <= budget:
            selected.append(row)
            total_cost += retention_cost
        else:
            break

    selected_df = pd.DataFrame(selected)

    print("\nRetention Strategy Summary")
    print("───────────────────────────")
    print(f"  Customers selected   : {len(selected_df):,}")
    print(f"  Total campaign cost  : ${total_cost:,.0f}")
    if len(selected_df):
        print(f"  Expected savings     : ${selected_df['expected_savings'].sum():,.0f}")
        print(f"  Projected net profit : ${selected_df['net_expected_value'].sum():,.0f}")

    return selected_df
