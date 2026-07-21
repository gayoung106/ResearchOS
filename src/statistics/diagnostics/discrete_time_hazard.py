"""Diagnostics for discrete-time hazard survival regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class DiscreteTimeHazardDiagnosticsReport:
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


def _validate_result(result: RegressionResult) -> Any:
    if result.model_type != "discrete_time_hazard":
        raise ValueError("Discrete-time hazard diagnostics require model_type='discrete_time_hazard'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for discrete-time hazard diagnostics.")
    return result.raw_result


def build_discrete_time_hazard_diagnostics(result: RegressionResult) -> DiscreteTimeHazardDiagnosticsReport:
    fitted = _validate_result(result)
    long_data = pd.DataFrame(result.metadata.get("long_data", []))
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    observed = np.asarray(fitted.model.endog, dtype=float)
    residuals = long_data.copy()
    residuals["observed_event"] = observed
    residuals["predicted_hazard"] = predicted
    residuals["response_residual"] = observed - predicted
    residuals["deviance_residual"] = np.asarray(fitted.resid_deviance, dtype=float)
    interval_hazards = pd.DataFrame(result.metadata.get("baseline_interval_hazards", []))
    event_count = int(result.fit_statistics.get("event_count", int(observed.sum())))
    censored_count = int(result.fit_statistics.get("censored_count", result.sample_size - event_count))
    interval_count = int(result.fit_statistics.get("interval_count", len(interval_hazards)))
    long_row_count = int(result.fit_statistics.get("long_row_count", len(residuals)))
    warnings = list(result.warnings)
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "event_count": event_count,
        "censored_count": censored_count,
        "event_rate": result.fit_statistics.get("event_rate"),
        "interval_count": interval_count,
        "long_row_count": long_row_count,
        "person_period_event_rate": result.fit_statistics.get("person_period_event_rate"),
        "brier_score": result.fit_statistics.get("brier_score"),
        "pseudo_r_squared_mcfadden": result.fit_statistics.get("pseudo_r_squared_mcfadden"),
        "events_per_parameter": result.fit_statistics.get("events_per_parameter"),
        "link": result.metadata.get("link"),
    }
    return DiscreteTimeHazardDiagnosticsReport(
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


def discrete_time_interval_hazards_to_dataframe(report: DiscreteTimeHazardDiagnosticsReport) -> pd.DataFrame:
    return report.interval_hazards.copy()


def discrete_time_residuals_to_dataframe(report: DiscreteTimeHazardDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def discrete_time_diagnostic_summary_to_dataframe(report: DiscreteTimeHazardDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
