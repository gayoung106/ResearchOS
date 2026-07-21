"""Diagnostics for Weibull proportional hazards models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class WeibullPHDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    event_count: int
    censored_count: int
    parameter_count: int
    multicollinearity: list[MulticollinearityResult]
    residuals: pd.DataFrame
    prediction_metrics: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_weibull_ph_result(result: RegressionResult) -> Any:
    if result.model_type != "weibull_ph":
        raise ValueError("Weibull PH diagnostics require model_type='weibull_ph'.")
    if result.raw_result is None:
        raise ValueError("A fitted Weibull result is required.")
    return result.raw_result


def calculate_weibull_ph_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_weibull_ph_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
    output: list[MulticollinearityResult] = []
    for index, name in enumerate(names):
        if name.lower() in {"const", "intercept"}:
            continue
        try:
            vif = 1.0 if exog.shape[1] == 1 else float(variance_inflation_factor(exog, index))
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


def _concordance_index(duration: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> float:
    comparable = 0
    concordant = 0.0
    for i in range(len(duration)):
        if event[i] != 1:
            continue
        for j in range(len(duration)):
            if duration[i] >= duration[j]:
                continue
            comparable += 1
            if risk_score[i] > risk_score[j]:
                concordant += 1.0
            elif np.isclose(risk_score[i], risk_score[j]):
                concordant += 0.5
    return float(concordant / comparable) if comparable else np.nan


def build_weibull_ph_diagnostics(result: RegressionResult) -> WeibullPHDiagnosticsReport:
    fitted = _validate_weibull_ph_result(result)
    duration = np.asarray(fitted.model.endog, dtype=float)
    event = np.asarray(fitted.model.status, dtype=int)
    shape = float(result.fit_statistics["shape"])
    scale = np.asarray(fitted.predict(kind="scale"), dtype=float)
    cox_snell = (duration / scale) ** shape
    risk_score = -np.asarray(fitted.predict(kind="linear"), dtype=float) * shape
    row_labels = list(fitted.model.row_labels)
    residuals = pd.DataFrame(
        {
            "row_index": row_labels,
            "duration": duration,
            "event": event,
            "risk_score": risk_score,
            "cox_snell_residual": cox_snell,
        }
    )
    event_mask = event == 1
    metrics = {
        "mean_cox_snell_residual_events": float(np.mean(cox_snell[event_mask])) if event_mask.any() else np.nan,
        "concordance_index": _concordance_index(duration, event, risk_score),
    }
    multicollinearity = calculate_weibull_ph_multicollinearity(result)
    event_count = int(result.fit_statistics.get("event_count", int(event.sum())))
    censored_count = int(result.fit_statistics.get("censored_count", len(event) - event.sum()))
    parameter_count = len(result.coefficients)
    events_per_parameter = event_count / max(parameter_count, 1)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    if events_per_parameter < 10:
        warnings.append("Weibull PH regression has fewer than 10 events per estimated coefficient.")
    warnings.extend(result.warnings)
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "event_count": event_count,
        "censored_count": censored_count,
        "event_rate": result.fit_statistics.get("event_rate"),
        "parameter_count": parameter_count,
        "events_per_parameter": float(events_per_parameter),
        "shape": result.fit_statistics.get("shape"),
        "baseline_rate": result.fit_statistics.get("baseline_rate"),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
        **metrics,
    }
    return WeibullPHDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        event_count=event_count,
        censored_count=censored_count,
        parameter_count=parameter_count,
        multicollinearity=multicollinearity,
        residuals=residuals,
        prediction_metrics=metrics,
        warnings=warnings,
        summary=summary,
    )


def weibull_ph_multicollinearity_to_dataframe(report: WeibullPHDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def weibull_ph_residuals_to_dataframe(report: WeibullPHDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def weibull_ph_prediction_metrics_to_dataframe(report: WeibullPHDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame({"item": list(report.prediction_metrics.keys()), "value": list(report.prediction_metrics.values())})


def weibull_ph_diagnostic_summary_to_dataframe(report: WeibullPHDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
