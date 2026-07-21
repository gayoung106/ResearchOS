"""Diagnostics for regularized linear regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class RegularizedPredictionMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    residual_mean: float
    residual_std: float


@dataclass(slots=True)
class RegularizedCoefficientStatus:
    term: str
    estimate: float
    standardized_estimate: float | None
    selected: bool


@dataclass(slots=True)
class RegularizedDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    multicollinearity: list[MulticollinearityResult]
    coefficient_status: list[RegularizedCoefficientStatus]
    prediction_metrics: RegularizedPredictionMetrics
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_regularized_result(result: RegressionResult) -> Any:
    if result.model_type != "regularized_regression":
        raise ValueError("Regularized diagnostics require model_type='regularized_regression'.")
    if result.raw_result is None:
        raise ValueError("A fitted regularized regression result is required for diagnostics.")
    return result.raw_result


def calculate_regularized_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_regularized_result(result)
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
            interpretation = "Severe multicollinearity is suspected; regularization may be influential."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "Multicollinearity should be reviewed alongside penalty choice."
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


def build_regularized_diagnostics(result: RegressionResult) -> RegularizedDiagnosticsReport:
    fitted = _validate_regularized_result(result)
    observed = np.asarray(fitted.model.endog, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residual_values = np.asarray(fitted.resid, dtype=float)
    row_labels = result.metadata.get("row_labels") or getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(observed)))
    residuals = pd.DataFrame(
        {
            "row_index": row_labels,
            "observed": observed,
            "fitted": fitted_values,
            "residual": residual_values,
            "absolute_residual": np.abs(residual_values),
        }
    )
    metrics = RegularizedPredictionMetrics(
        mean_absolute_error=float(np.mean(np.abs(residual_values))),
        root_mean_squared_error=float(np.sqrt(np.mean(residual_values**2))),
        residual_mean=float(np.mean(residual_values)),
        residual_std=float(np.std(residual_values, ddof=1)) if len(residual_values) > 1 else np.nan,
    )
    standardized = result.metadata.get("standardized_coefficients", {})
    coefficient_status = [
        RegularizedCoefficientStatus(
            term=coefficient.term,
            estimate=float(coefficient.estimate),
            standardized_estimate=(
                float(standardized[coefficient.term]) if coefficient.term in standardized else None
            ),
            selected=bool(abs(coefficient.estimate) > 1e-8) if coefficient.term != "const" else True,
        )
        for coefficient in result.coefficients
    ]
    multicollinearity = calculate_regularized_multicollinearity(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    warnings.extend(result.warnings)
    if result.fit_statistics.get("selected_coefficient_count", 0) == 0:
        warnings.append("No predictors were selected after penalization.")
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "penalty": result.fit_statistics.get("penalty"),
        "alpha": result.fit_statistics.get("alpha"),
        "l1_ratio": result.fit_statistics.get("l1_ratio"),
        "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
        "root_mean_squared_error": metrics.root_mean_squared_error,
        "mean_absolute_error": metrics.mean_absolute_error,
        "selected_coefficient_count": result.fit_statistics.get("selected_coefficient_count"),
        "zero_coefficient_count": result.fit_statistics.get("zero_coefficient_count"),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return RegularizedDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        multicollinearity=multicollinearity,
        coefficient_status=coefficient_status,
        prediction_metrics=metrics,
        residuals=residuals,
        warnings=warnings,
        summary=summary,
    )


def regularized_multicollinearity_to_dataframe(report: RegularizedDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def regularized_coefficients_to_dataframe(report: RegularizedDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.coefficient_status])


def regularized_prediction_metrics_to_dataframe(report: RegularizedDiagnosticsReport) -> pd.DataFrame:
    values = asdict(report.prediction_metrics)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def regularized_residuals_to_dataframe(report: RegularizedDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def regularized_diagnostic_summary_to_dataframe(report: RegularizedDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
