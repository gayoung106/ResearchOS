"""Diagnostics for Tobit censored regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class TobitPredictionMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    uncensored_root_mean_squared_error: float
    censoring_accuracy: float


@dataclass(slots=True)
class TobitDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    multicollinearity: list[MulticollinearityResult]
    prediction_metrics: TobitPredictionMetrics
    observations: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_tobit_result(result: RegressionResult) -> Any:
    if result.model_type != "tobit_regression":
        raise ValueError("Tobit diagnostics require model_type='tobit_regression'.")
    if result.raw_result is None:
        raise ValueError("A fitted Tobit result is required for diagnostics.")
    return result.raw_result


def calculate_tobit_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_tobit_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in getattr(fitted.model, "exog_names", [])]
    output: list[MulticollinearityResult] = []
    for index, name in enumerate(names):
        if name.lower() in {"const", "intercept"}:
            continue
        if exog.shape[1] == 1:
            vif = 1.0
        else:
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


def build_tobit_diagnostics(result: RegressionResult) -> TobitDiagnosticsReport:
    fitted = _validate_tobit_result(result)
    observed = np.asarray(fitted.model.endog, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    latent_fitted = np.asarray(result.metadata.get("latent_fitted_values", []), dtype=float)
    left = np.asarray(result.metadata.get("left_censored", []), dtype=bool)
    right = np.asarray(result.metadata.get("right_censored", []), dtype=bool)
    censored = left | right
    row_labels = getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(observed)))
    observations = pd.DataFrame(
        {
            "row_index": row_labels,
            "observed": observed,
            "expected_observed": fitted_values,
            "latent_fitted": latent_fitted,
            "residual": residuals,
            "absolute_residual": np.abs(residuals),
            "left_censored": left,
            "right_censored": right,
            "censored": censored,
        }
    )
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    if (~censored).any():
        uncensored_rmse = float(np.sqrt(np.mean(residuals[~censored] ** 2)))
    else:
        uncensored_rmse = np.nan
    lower_limit = result.metadata.get("lower_limit")
    upper_limit = result.metadata.get("upper_limit")
    predicted_left = np.zeros(len(observed), dtype=bool)
    predicted_right = np.zeros(len(observed), dtype=bool)
    if lower_limit is not None:
        predicted_left = fitted_values <= float(lower_limit) + 1e-8
    if upper_limit is not None:
        predicted_right = fitted_values >= float(upper_limit) - 1e-8
    censoring_accuracy = float(np.mean((predicted_left | predicted_right) == censored))
    metrics = TobitPredictionMetrics(
        mean_absolute_error=mae,
        root_mean_squared_error=rmse,
        uncensored_root_mean_squared_error=uncensored_rmse,
        censoring_accuracy=censoring_accuracy,
    )
    multicollinearity = calculate_tobit_multicollinearity(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    warnings.extend(result.warnings)
    if result.fit_statistics.get("censoring_rate", 0.0) > 0.5:
        warnings.append("More than half of observations are censored; review Tobit assumptions.")
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "left_censored_count": result.fit_statistics.get("left_censored_count"),
        "right_censored_count": result.fit_statistics.get("right_censored_count"),
        "uncensored_count": result.fit_statistics.get("uncensored_count"),
        "censoring_rate": result.fit_statistics.get("censoring_rate"),
        "sigma": result.fit_statistics.get("sigma"),
        "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
        "root_mean_squared_error": rmse,
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return TobitDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        multicollinearity=multicollinearity,
        prediction_metrics=metrics,
        observations=observations,
        warnings=warnings,
        summary=summary,
    )


def tobit_multicollinearity_to_dataframe(report: TobitDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def tobit_prediction_metrics_to_dataframe(report: TobitDiagnosticsReport) -> pd.DataFrame:
    values = asdict(report.prediction_metrics)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def tobit_observations_to_dataframe(report: TobitDiagnosticsReport) -> pd.DataFrame:
    return report.observations.copy()


def tobit_diagnostic_summary_to_dataframe(report: TobitDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
