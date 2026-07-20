"""Diagnostics for quantile regression models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class QuantileResidualSummary:
    residual_mean: float
    residual_median: float
    residual_mad: float
    residual_min: float
    residual_max: float
    pinball_loss: float


@dataclass(slots=True)
class QuantileDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    quantile: float
    parameter_count: int
    multicollinearity: list[MulticollinearityResult]
    residual_summary: QuantileResidualSummary
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_quantile_result(result: RegressionResult) -> Any:
    if result.model_type != "quantile_regression":
        raise ValueError("Quantile diagnostics require model_type='quantile_regression'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for quantile diagnostics.")
    return result.raw_result


def _pinball_loss(residuals: np.ndarray, quantile: float) -> float:
    return float(np.mean(np.maximum(quantile * residuals, (quantile - 1.0) * residuals)))


def calculate_quantile_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_quantile_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in getattr(fitted.model, "exog_names", [])]
    if len(names) != exog.shape[1]:
        names = [f"x{index + 1}" for index in range(exog.shape[1])]

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


def build_quantile_diagnostics(result: RegressionResult) -> QuantileDiagnosticsReport:
    fitted = _validate_quantile_result(result)
    quantile = float(result.fit_statistics.get("quantile", 0.5))
    residuals = np.asarray(fitted.resid, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    observed = np.asarray(fitted.model.endog, dtype=float)
    row_labels = getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(observed)))

    residual_frame = pd.DataFrame(
        {
            "row_index": row_labels,
            "observed": observed,
            "fitted": fitted_values,
            "residual": residuals,
            "absolute_residual": np.abs(residuals),
        }
    )
    pinball = _pinball_loss(residuals, quantile)
    residual_summary = QuantileResidualSummary(
        residual_mean=float(np.mean(residuals)),
        residual_median=float(np.median(residuals)),
        residual_mad=float(np.median(np.abs(residuals - np.median(residuals)))),
        residual_min=float(np.min(residuals)),
        residual_max=float(np.max(residuals)),
        pinball_loss=pinball,
    )
    multicollinearity = calculate_quantile_multicollinearity(result)
    parameter_count = len(result.coefficients)
    observations_per_parameter = result.sample_size / parameter_count if parameter_count else np.nan
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    if np.isfinite(observations_per_parameter) and observations_per_parameter < 10:
        warnings.append("The sample size may be small relative to the number of parameters.")
    if result.warnings:
        warnings.extend(result.warnings)

    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "quantile": quantile,
        "parameter_count": parameter_count,
        "observations_per_parameter": observations_per_parameter,
        "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
        "pinball_loss": pinball,
        "residual_median": residual_summary.residual_median,
        "residual_mad": residual_summary.residual_mad,
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return QuantileDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        quantile=quantile,
        parameter_count=parameter_count,
        multicollinearity=multicollinearity,
        residual_summary=residual_summary,
        residuals=residual_frame,
        warnings=warnings,
        summary=summary,
    )


def quantile_multicollinearity_to_dataframe(report: QuantileDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def quantile_residual_summary_to_dataframe(report: QuantileDiagnosticsReport) -> pd.DataFrame:
    values = asdict(report.residual_summary)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def quantile_residuals_to_dataframe(report: QuantileDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def quantile_diagnostic_summary_to_dataframe(report: QuantileDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
