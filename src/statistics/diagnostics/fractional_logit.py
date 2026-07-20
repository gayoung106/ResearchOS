"""Diagnostics for fractional logit models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class FractionalPredictionMetrics:
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_prediction: float
    minimum_prediction: float
    maximum_prediction: float
    boundary_count: int


@dataclass(slots=True)
class FractionalLogitDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    multicollinearity: list[MulticollinearityResult]
    prediction_metrics: FractionalPredictionMetrics
    observations: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_fractional_result(result: RegressionResult) -> Any:
    if result.model_type != "fractional_logit":
        raise ValueError("Fractional logit diagnostics require model_type='fractional_logit'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for fractional logit diagnostics.")
    return result.raw_result


def calculate_fractional_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_fractional_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
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


def build_fractional_logit_diagnostics(result: RegressionResult) -> FractionalLogitDiagnosticsReport:
    fitted = _validate_fractional_result(result)
    observed = np.asarray(fitted.model.endog, dtype=float)
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    raw_residual = observed - predicted
    pearson_residual = np.asarray(fitted.resid_pearson, dtype=float)
    deviance_residual = np.asarray(fitted.resid_deviance, dtype=float)
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
            "deviance_residual": deviance_residual,
            "absolute_error": np.abs(raw_residual),
        }
    )
    boundary_count = int(((observed == 0.0) | (observed == 1.0)).sum())
    metrics = FractionalPredictionMetrics(
        mean_absolute_error=float(np.mean(np.abs(raw_residual))),
        root_mean_squared_error=float(np.sqrt(np.mean(raw_residual**2))),
        mean_prediction=float(np.mean(predicted)),
        minimum_prediction=float(np.min(predicted)),
        maximum_prediction=float(np.max(predicted)),
        boundary_count=boundary_count,
    )
    multicollinearity = calculate_fractional_multicollinearity(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    if boundary_count == len(observed):
        warnings.append("All observed outcomes are boundary values; binary logit may be more appropriate.")
    if result.fit_statistics.get("dispersion_ratio") is not None and float(result.fit_statistics["dispersion_ratio"]) > 1.5:
        warnings.append("Pearson dispersion ratio is above 1.5; robust inference should be reviewed.")
    warnings.extend(result.warnings)

    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "pseudo_r_squared_deviance": result.fit_statistics.get("pseudo_r_squared_deviance"),
        "dispersion_ratio": result.fit_statistics.get("dispersion_ratio"),
        "mean_absolute_error": metrics.mean_absolute_error,
        "root_mean_squared_error": metrics.root_mean_squared_error,
        "boundary_count": boundary_count,
        "zero_count": result.fit_statistics.get("zero_count"),
        "one_count": result.fit_statistics.get("one_count"),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return FractionalLogitDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        multicollinearity=multicollinearity,
        prediction_metrics=metrics,
        observations=observations,
        warnings=warnings,
        summary=summary,
    )


def fractional_multicollinearity_to_dataframe(report: FractionalLogitDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def fractional_prediction_metrics_to_dataframe(report: FractionalLogitDiagnosticsReport) -> pd.DataFrame:
    values = asdict(report.prediction_metrics)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def fractional_observations_to_dataframe(report: FractionalLogitDiagnosticsReport) -> pd.DataFrame:
    return report.observations.copy()


def fractional_diagnostic_summary_to_dataframe(report: FractionalLogitDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
