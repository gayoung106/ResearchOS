"""Diagnostics for IV two-stage least-squares regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class IVFirstStageDiagnostic:
    endogenous_variable: str
    excluded_instrument_f_statistic: float | None
    excluded_instrument_p_value: float | None
    r_squared: float
    restricted_r_squared: float
    status: str
    interpretation: str


@dataclass(slots=True)
class IV2SLSDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    multicollinearity: list[MulticollinearityResult]
    first_stage: list[IVFirstStageDiagnostic]
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_iv_result(result: RegressionResult) -> Any:
    if result.model_type != "iv_2sls_regression":
        raise ValueError("IV diagnostics require model_type='iv_2sls_regression'.")
    if result.raw_result is None:
        raise ValueError("A fitted IV result is required for diagnostics.")
    return result.raw_result


def calculate_iv_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_iv_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
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
            interpretation = "Severe second-stage multicollinearity is suspected."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "Second-stage multicollinearity should be reviewed."
        else:
            status = "PASS"
            interpretation = "Second-stage VIF is within the usual screening threshold."
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


def _first_stage_diagnostics(result: RegressionResult) -> list[IVFirstStageDiagnostic]:
    output: list[IVFirstStageDiagnostic] = []
    for variable, values in result.metadata.get("first_stage", {}).items():
        statistic = values.get("excluded_instrument_f_statistic")
        if statistic is None:
            status = "UNAVAILABLE"
            interpretation = "First-stage excluded-instrument F statistic is unavailable."
        elif float(statistic) < 10.0:
            status = "WARNING"
            interpretation = "Excluded instruments may be weak for this endogenous variable."
        else:
            status = "PASS"
            interpretation = "Excluded instruments pass the common first-stage F screening rule."
        output.append(
            IVFirstStageDiagnostic(
                endogenous_variable=str(variable),
                excluded_instrument_f_statistic=(float(statistic) if statistic is not None else None),
                excluded_instrument_p_value=(
                    float(values["excluded_instrument_p_value"])
                    if values.get("excluded_instrument_p_value") is not None
                    else None
                ),
                r_squared=float(values.get("r_squared", np.nan)),
                restricted_r_squared=float(values.get("restricted_r_squared", np.nan)),
                status=status,
                interpretation=interpretation,
            )
        )
    return output


def build_iv_2sls_diagnostics(result: RegressionResult) -> IV2SLSDiagnosticsReport:
    fitted = _validate_iv_result(result)
    observed = np.asarray(fitted.model.endog, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residual_values = np.asarray(fitted.resid, dtype=float)
    row_labels = getattr(fitted.model.data, "row_labels", None)
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
    multicollinearity = calculate_iv_multicollinearity(result)
    first_stage = _first_stage_diagnostics(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    warnings.extend(
        f"{item.endogenous_variable}: {item.interpretation}"
        for item in first_stage
        if item.status == "WARNING"
    )
    warnings.extend(result.warnings)
    min_f = result.fit_statistics.get("minimum_first_stage_f_statistic")
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "r_squared": result.fit_statistics.get("r_squared"),
        "root_mean_squared_error": result.fit_statistics.get("root_mean_squared_error"),
        "endogenous_variable_count": result.fit_statistics.get("endogenous_variable_count"),
        "instrument_count": result.fit_statistics.get("instrument_count"),
        "overidentified": result.fit_statistics.get("overidentified"),
        "minimum_first_stage_f_statistic": min_f,
        "weak_instrument_warning": bool(min_f is not None and float(min_f) < 10.0),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return IV2SLSDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        multicollinearity=multicollinearity,
        first_stage=first_stage,
        residuals=residuals,
        warnings=warnings,
        summary=summary,
    )


def iv_multicollinearity_to_dataframe(report: IV2SLSDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def iv_first_stage_to_dataframe(report: IV2SLSDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.first_stage])


def iv_residuals_to_dataframe(report: IV2SLSDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def iv_diagnostic_summary_to_dataframe(report: IV2SLSDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
