"""Diagnostics for robust linear regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class RobustWeightSummary:
    mean_weight: float
    median_weight: float
    minimum_weight: float
    downweighted_count: int
    heavily_downweighted_count: int


@dataclass(slots=True)
class RobustDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    multicollinearity: list[MulticollinearityResult]
    residuals: pd.DataFrame
    weight_summary: RobustWeightSummary
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_robust_result(result: RegressionResult) -> Any:
    if result.model_type != "robust_regression":
        raise ValueError("Robust regression diagnostics require model_type='robust_regression'.")
    if result.raw_result is None:
        raise ValueError("A fitted robust regression result is required for diagnostics.")
    return result.raw_result


def calculate_robust_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_robust_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in getattr(fitted.model, "exog_names", [])]
    output: list[MulticollinearityResult] = []
    for index, name in enumerate(names):
        if name.lower() in {"const", "intercept"}:
            continue
        try:
            vif = float(variance_inflation_factor(exog, index))
        except (ValueError, IndexError, np.linalg.LinAlgError, ZeroDivisionError):
            vif = np.inf
        tolerance = 0.0 if not np.isfinite(vif) or np.isclose(vif, 0.0) else 1.0 / vif
        if not np.isfinite(vif) or vif >= 10:
            status = "FAIL"
            interpretation = "Severe multicollinearity is suspected."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "Multicollinearity should be reviewed."
        else:
            status = "PASS"
            interpretation = "VIF is within the usual screening threshold."
        output.append(
            MulticollinearityResult(
                variable_name=name,
                vif=float(vif),
                tolerance=float(tolerance),
                status=status,
                interpretation=interpretation,
            )
        )
    return output


def build_robust_diagnostics(result: RegressionResult) -> RobustDiagnosticsReport:
    fitted = _validate_robust_result(result)
    observed = np.asarray(fitted.model.endog, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    weights = np.asarray(fitted.weights, dtype=float)
    row_labels = result.metadata.get("row_labels") or getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(observed)))
    residual_frame = pd.DataFrame(
        {
            "row_index": row_labels,
            "observed": observed,
            "fitted": fitted_values,
            "residual": residuals,
            "absolute_residual": np.abs(residuals),
            "robust_weight": weights,
            "downweighted": weights < 0.999,
            "heavily_downweighted": weights < 0.5,
        }
    )
    weight_summary = RobustWeightSummary(
        mean_weight=float(np.mean(weights)),
        median_weight=float(np.median(weights)),
        minimum_weight=float(np.min(weights)),
        downweighted_count=int(np.sum(weights < 0.999)),
        heavily_downweighted_count=int(np.sum(weights < 0.5)),
    )
    multicollinearity = calculate_robust_multicollinearity(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    warnings.extend(result.warnings)
    if weight_summary.heavily_downweighted_count:
        warnings.append(
            f"{weight_summary.heavily_downweighted_count} observations received robust weights below 0.5."
        )
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
        "scale": result.fit_statistics.get("scale"),
        "downweighted_count": weight_summary.downweighted_count,
        "heavily_downweighted_count": weight_summary.heavily_downweighted_count,
        "mean_weight": weight_summary.mean_weight,
        "minimum_weight": weight_summary.minimum_weight,
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return RobustDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        multicollinearity=multicollinearity,
        residuals=residual_frame,
        weight_summary=weight_summary,
        warnings=warnings,
        summary=summary,
    )


def robust_multicollinearity_to_dataframe(report: RobustDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def robust_residuals_to_dataframe(report: RobustDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def robust_weight_summary_to_dataframe(report: RobustDiagnosticsReport) -> pd.DataFrame:
    values = asdict(report.weight_summary)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def robust_diagnostic_summary_to_dataframe(report: RobustDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
