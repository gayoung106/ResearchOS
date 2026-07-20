"""Diagnostics for Cox proportional hazards models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class CoxPHAssumptionCheck:
    term: str
    spearman_correlation: float
    p_value: float
    status: str
    interpretation: str


@dataclass(slots=True)
class CoxDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    event_count: int
    censored_count: int
    parameter_count: int
    multicollinearity: list[MulticollinearityResult]
    proportional_hazards: list[CoxPHAssumptionCheck]
    residuals: pd.DataFrame
    baseline_survival: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_cox_result(result: RegressionResult) -> Any:
    if result.model_type != "cox_proportional_hazards":
        raise ValueError("Cox diagnostics require model_type='cox_proportional_hazards'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for Cox diagnostics.")
    return result.raw_result


def calculate_cox_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_cox_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
    output: list[MulticollinearityResult] = []
    for index, name in enumerate(names):
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


def calculate_cox_ph_checks(result: RegressionResult) -> list[CoxPHAssumptionCheck]:
    fitted = _validate_cox_result(result)
    durations = np.asarray(fitted.model.endog, dtype=float)
    events = np.asarray(fitted.model.status, dtype=int)
    schoenfeld = np.asarray(fitted.schoenfeld_residuals, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
    event_mask = events == 1
    event_times = durations[event_mask]
    checks: list[CoxPHAssumptionCheck] = []
    for index, name in enumerate(names):
        values = schoenfeld[event_mask, index]
        valid = np.isfinite(event_times) & np.isfinite(values)
        if int(valid.sum()) < 3 or np.isclose(np.std(values[valid]), 0.0):
            correlation = np.nan
            p_value = np.nan
        else:
            correlation, p_value = stats.spearmanr(event_times[valid], values[valid])
            correlation = float(correlation)
            p_value = float(p_value)
        if np.isfinite(p_value) and p_value < 0.05:
            status = "WARNING"
            interpretation = "Schoenfeld residuals are correlated with event time; PH assumption should be reviewed."
        else:
            status = "PASS"
            interpretation = "No time trend was detected by the Schoenfeld residual screen."
        checks.append(
            CoxPHAssumptionCheck(
                term=name,
                spearman_correlation=float(correlation) if np.isfinite(correlation) else np.nan,
                p_value=float(p_value) if np.isfinite(p_value) else np.nan,
                status=status,
                interpretation=interpretation,
            )
        )
    return checks


def _baseline_survival_to_dataframe(result: RegressionResult) -> pd.DataFrame:
    return pd.DataFrame(result.metadata.get("baseline_survival", []))


def build_cox_diagnostics(result: RegressionResult) -> CoxDiagnosticsReport:
    fitted = _validate_cox_result(result)
    durations = np.asarray(fitted.model.endog, dtype=float)
    events = np.asarray(fitted.model.status, dtype=int)
    martingale = np.asarray(fitted.martingale_residuals, dtype=float)
    score = np.asarray(fitted.score_residuals, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
    row_labels = result.metadata.get("row_labels") or list(range(len(durations)))

    residual_values: dict[str, Any] = {
        "row_index": row_labels,
        "duration": durations,
        "event": events,
        "martingale_residual": martingale,
    }
    for index, name in enumerate(names):
        residual_values[f"score_residual_{name}"] = score[:, index]
    residuals = pd.DataFrame(residual_values)

    multicollinearity = calculate_cox_multicollinearity(result)
    ph_checks = calculate_cox_ph_checks(result)
    baseline = _baseline_survival_to_dataframe(result)
    event_count = int(result.fit_statistics.get("event_count", int(events.sum())))
    censored_count = int(result.fit_statistics.get("censored_count", len(events) - events.sum()))
    parameter_count = len(result.coefficients)
    events_per_parameter = event_count / max(parameter_count, 1)

    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    warnings.extend(
        f"{item.term}: {item.interpretation}"
        for item in ph_checks
        if item.status == "WARNING"
    )
    if events_per_parameter < 10:
        warnings.append("Cox regression has fewer than 10 events per estimated parameter.")
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
        "ph_warning_count": sum(item.status == "WARNING" for item in ph_checks),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
        "baseline_survival_rows": len(baseline),
    }
    return CoxDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        event_count=event_count,
        censored_count=censored_count,
        parameter_count=parameter_count,
        multicollinearity=multicollinearity,
        proportional_hazards=ph_checks,
        residuals=residuals,
        baseline_survival=baseline,
        warnings=warnings,
        summary=summary,
    )


def cox_multicollinearity_to_dataframe(report: CoxDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def cox_ph_checks_to_dataframe(report: CoxDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.proportional_hazards])


def cox_residuals_to_dataframe(report: CoxDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def cox_baseline_survival_to_dataframe(report: CoxDiagnosticsReport) -> pd.DataFrame:
    return report.baseline_survival.copy()


def cox_diagnostic_summary_to_dataframe(report: CoxDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
