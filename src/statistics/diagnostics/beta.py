"""Diagnostics for beta regression models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class BetaPredictionMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_prediction: float
    minimum_prediction: float
    maximum_prediction: float
    precision: float | None


@dataclass(slots=True)
class BetaDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    multicollinearity: list[MulticollinearityResult]
    prediction_metrics: BetaPredictionMetrics
    observations: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_beta_result(result: RegressionResult) -> Any:
    if result.model_type != "beta_regression":
        raise ValueError("Beta diagnostics require model_type='beta_regression'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for beta diagnostics.")
    return result.raw_result


def calculate_beta_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_beta_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in result.metadata.get("design_matrix_columns", [])]
    if len(names) != exog.shape[1]:
        names = [str(name) for name in fitted.model.exog_names[: exog.shape[1]]]
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


def build_beta_diagnostics(result: RegressionResult) -> BetaDiagnosticsReport:
    fitted = _validate_beta_result(result)
    observed = np.asarray(fitted.model.endog, dtype=float)
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    raw_residual = observed - predicted
    pearson_residual = np.asarray(fitted.resid_pearson, dtype=float)
    row_labels = getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(observed)))

    observations = pd.DataFrame(
        {
            "row_index": row_labels,
            "observed": observed,
            "predicted": predicted,
            "raw_residual": raw_residual,
            "pearson_residual": pearson_residual,
            "absolute_error": np.abs(raw_residual),
        }
    )
    precision = result.fit_statistics.get("precision")
    metrics = BetaPredictionMetrics(
        mean_absolute_error=float(np.mean(np.abs(raw_residual))),
        root_mean_squared_error=float(np.sqrt(np.mean(raw_residual**2))),
        mean_prediction=float(np.mean(predicted)),
        minimum_prediction=float(np.min(predicted)),
        maximum_prediction=float(np.max(predicted)),
        precision=float(precision) if precision is not None else None,
    )
    multicollinearity = calculate_beta_multicollinearity(result)
    parameter_count = int(result.fit_statistics.get("parameter_count", len(result.coefficients)))
    observations_per_parameter = result.sample_size / parameter_count if parameter_count else np.nan
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    if np.isfinite(observations_per_parameter) and observations_per_parameter < 10:
        warnings.append("The sample size may be small relative to the number of beta regression parameters.")
    warnings.extend(result.warnings)

    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "parameter_count": parameter_count,
        "observations_per_parameter": observations_per_parameter,
        "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
        "precision": precision,
        "mean_absolute_error": metrics.mean_absolute_error,
        "root_mean_squared_error": metrics.root_mean_squared_error,
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return BetaDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        multicollinearity=multicollinearity,
        prediction_metrics=metrics,
        observations=observations,
        warnings=warnings,
        summary=summary,
    )


def beta_multicollinearity_to_dataframe(report: BetaDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def beta_prediction_metrics_to_dataframe(report: BetaDiagnosticsReport) -> pd.DataFrame:
    values = asdict(report.prediction_metrics)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def beta_observations_to_dataframe(report: BetaDiagnosticsReport) -> pd.DataFrame:
    return report.observations.copy()


def beta_diagnostic_summary_to_dataframe(report: BetaDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
