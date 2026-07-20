"""Diagnostics for Generalized Estimating Equation regression results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult

GEE_DIAGNOSTIC_MODELS = {"gee_gaussian", "gee_logit", "gee_poisson"}


@dataclass(slots=True)
class GEEClusterDiagnostic:
    group: str
    observation_count: int
    observed_mean: float
    predicted_mean: float
    raw_residual_mean: float
    pearson_residual_mean: float
    pearson_residual_sd: float


@dataclass(slots=True)
class GEEDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    cluster_count: int
    group_variable: str
    covariance_structure: str
    cluster_diagnostics: list[GEEClusterDiagnostic]
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_gee_result(result: RegressionResult) -> None:
    if result.model_type not in GEE_DIAGNOSTIC_MODELS:
        raise ValueError(f"GEE diagnostics require a GEE result; got {result.model_type}.")
    diagnostics = result.metadata.get("diagnostics", {})
    required = {"endog", "predicted_mean", "group_labels"}
    missing = sorted(required - set(diagnostics))
    if missing:
        raise ValueError("GEE diagnostics metadata is missing: " + ", ".join(missing))


def _variance_function(model_type: str, predicted: np.ndarray, scale: float) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=float)
    if model_type == "gee_logit":
        clipped = np.clip(predicted, 1e-8, 1 - 1e-8)
        return clipped * (1 - clipped)
    if model_type == "gee_poisson":
        return np.clip(predicted, 1e-8, None)
    return np.full_like(predicted, max(float(scale), 1e-8), dtype=float)


def build_gee_diagnostics(result: RegressionResult) -> GEEDiagnosticsReport:
    """Build residual and cluster diagnostics for a GEE result."""
    _validate_gee_result(result)
    diagnostics = result.metadata["diagnostics"]
    observed = np.asarray(diagnostics["endog"], dtype=float)
    predicted = np.asarray(diagnostics["predicted_mean"], dtype=float)
    groups = pd.Series(diagnostics["group_labels"], dtype=str)
    if observed.shape != predicted.shape or len(groups) != len(observed):
        raise ValueError("GEE diagnostics arrays have inconsistent lengths.")

    raw_residual = observed - predicted
    variance = _variance_function(result.model_type, predicted, float(result.fit_statistics.get("scale", 1.0)))
    pearson_residual = raw_residual / np.sqrt(variance)
    residuals = pd.DataFrame(
        {
            "row_label": diagnostics.get("row_labels", list(range(len(observed)))),
            "group": groups,
            "observed": observed,
            "predicted": predicted,
            "raw_residual": raw_residual,
            "pearson_residual": pearson_residual,
        }
    )

    cluster_diagnostics: list[GEEClusterDiagnostic] = []
    for group, frame in residuals.groupby("group", sort=False):
        cluster_diagnostics.append(
            GEEClusterDiagnostic(
                group=str(group),
                observation_count=int(len(frame)),
                observed_mean=float(frame["observed"].mean()),
                predicted_mean=float(frame["predicted"].mean()),
                raw_residual_mean=float(frame["raw_residual"].mean()),
                pearson_residual_mean=float(frame["pearson_residual"].mean()),
                pearson_residual_sd=float(frame["pearson_residual"].std(ddof=1))
                if len(frame) > 1
                else 0.0,
            )
        )

    cluster_count = len(cluster_diagnostics)
    small_cluster_count = sum(item.observation_count < 2 for item in cluster_diagnostics)
    max_abs_cluster_mean = max(
        (abs(item.pearson_residual_mean) for item in cluster_diagnostics), default=0.0
    )
    warnings: list[str] = []
    if cluster_count < 10:
        warnings.append("GEE robust standard errors can be unstable with fewer than 10 clusters.")
    if small_cluster_count:
        warnings.append(f"GEE diagnostics found {small_cluster_count} clusters with fewer than 2 observations.")
    if max_abs_cluster_mean > 2.0:
        warnings.append("At least one cluster has a large mean Pearson residual.")

    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "cluster_count": cluster_count,
        "group_variable": result.metadata.get("group_variable"),
        "covariance_structure": result.metadata.get("covariance_structure"),
        "residual_mean": float(np.mean(raw_residual)),
        "pearson_residual_mean": float(np.mean(pearson_residual)),
        "pearson_residual_sd": float(np.std(pearson_residual, ddof=1))
        if len(pearson_residual) > 1
        else 0.0,
        "max_abs_cluster_mean_pearson_residual": float(max_abs_cluster_mean),
        "small_cluster_count": small_cluster_count,
        "warning_count": len(warnings),
    }

    return GEEDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        cluster_count=cluster_count,
        group_variable=str(result.metadata.get("group_variable")),
        covariance_structure=str(result.metadata.get("covariance_structure")),
        cluster_diagnostics=cluster_diagnostics,
        residuals=residuals,
        warnings=warnings,
        summary=summary,
    )


def gee_cluster_diagnostics_to_dataframe(report: GEEDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.cluster_diagnostics])


def gee_residuals_to_dataframe(report: GEEDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def gee_diagnostic_summary_to_dataframe(report: GEEDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame({"item": list(report.summary), "value": list(report.summary.values())})
