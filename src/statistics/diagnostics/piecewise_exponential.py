"""Diagnostics for piecewise exponential survival regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class PiecewiseExponentialDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    event_count: int
    censored_count: int
    interval_count: int
    long_row_count: int
    interval_hazards: pd.DataFrame
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_piecewise_result(result: RegressionResult) -> Any:
    if result.model_type != "piecewise_exponential":
        raise ValueError("Piecewise exponential diagnostics require model_type='piecewise_exponential'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for piecewise exponential diagnostics.")
    return result.raw_result


def build_piecewise_exponential_diagnostics(result: RegressionResult) -> PiecewiseExponentialDiagnosticsReport:
    fitted = _validate_piecewise_result(result)
    long_data = pd.DataFrame(result.metadata.get("long_data", []))
    predicted_events = np.asarray(fitted.fittedvalues, dtype=float)
    observed_events = np.asarray(fitted.model.endog, dtype=float)
    residuals = long_data.copy()
    residuals["observed_event"] = observed_events
    residuals["predicted_event"] = predicted_events
    residuals["response_residual"] = observed_events - predicted_events
    residuals["deviance_residual"] = np.asarray(fitted.resid_deviance, dtype=float)
    interval_hazards = pd.DataFrame(result.metadata.get("baseline_interval_hazards", []))
    event_count = int(result.fit_statistics.get("event_count", int(observed_events.sum())))
    censored_count = int(result.fit_statistics.get("censored_count", result.sample_size - event_count))
    interval_count = int(result.fit_statistics.get("interval_count", len(interval_hazards)))
    long_row_count = int(result.fit_statistics.get("long_row_count", len(residuals)))
    warnings = list(result.warnings)
    if interval_count <= 1:
        warnings.append("Only one interval was used; piecewise baseline variation cannot be assessed.")
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "event_count": event_count,
        "censored_count": censored_count,
        "event_rate": result.fit_statistics.get("event_rate"),
        "interval_count": interval_count,
        "long_row_count": long_row_count,
        "total_exposure": result.fit_statistics.get("total_exposure"),
        "dispersion_ratio": result.fit_statistics.get("dispersion_ratio"),
        "events_per_parameter": result.fit_statistics.get("events_per_parameter"),
        "aic": result.fit_statistics.get("aic"),
        "bic": result.fit_statistics.get("bic"),
    }
    return PiecewiseExponentialDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        event_count=event_count,
        censored_count=censored_count,
        interval_count=interval_count,
        long_row_count=long_row_count,
        interval_hazards=interval_hazards,
        residuals=residuals,
        warnings=warnings,
        summary=summary,
    )


def piecewise_interval_hazards_to_dataframe(report: PiecewiseExponentialDiagnosticsReport) -> pd.DataFrame:
    return report.interval_hazards.copy()


def piecewise_residuals_to_dataframe(report: PiecewiseExponentialDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def piecewise_diagnostic_summary_to_dataframe(report: PiecewiseExponentialDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
