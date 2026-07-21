"""Diagnostics for Heckman two-step selection models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class HeckmanSelectionCoefficient:
    term: str
    estimate: float
    standard_error: float
    statistic: float
    p_value: float


@dataclass(slots=True)
class HeckmanDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    selection_sample_size: int
    selection_coefficients: list[HeckmanSelectionCoefficient]
    multicollinearity: list[MulticollinearityResult]
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_heckman_result(result: RegressionResult) -> Any:
    if result.model_type != "heckman_selection":
        raise ValueError("Heckman diagnostics require model_type='heckman_selection'.")
    if result.raw_result is None:
        raise ValueError("A fitted Heckman result is required for diagnostics.")
    return result.raw_result


def calculate_heckman_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    fitted = _validate_heckman_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in fitted.model.exog_names]
    output: list[MulticollinearityResult] = []
    for index, name in enumerate(names):
        if name.lower() in {"const", "intercept", "inverse_mills_ratio"}:
            continue
        try:
            vif = float(variance_inflation_factor(exog, index))
        except (ValueError, IndexError, np.linalg.LinAlgError, ZeroDivisionError):
            vif = np.inf
        tolerance = 0.0 if not np.isfinite(vif) or np.isclose(vif, 0.0) else 1.0 / vif
        if not np.isfinite(vif) or vif >= 10:
            status = "FAIL"
            interpretation = "Severe outcome-equation multicollinearity is suspected."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "Outcome-equation multicollinearity should be reviewed."
        else:
            status = "PASS"
            interpretation = "Outcome-equation VIF is within the usual screening threshold."
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


def build_heckman_diagnostics(result: RegressionResult) -> HeckmanDiagnosticsReport:
    fitted = _validate_heckman_result(result)
    outcome_result = fitted.outcome_result
    selection_result = fitted.selection_result
    observed = np.asarray(outcome_result.model.endog, dtype=float)
    fitted_values = np.asarray(outcome_result.fittedvalues, dtype=float)
    residual_values = np.asarray(outcome_result.resid, dtype=float)
    row_labels = result.metadata.get("selected_row_labels") or getattr(outcome_result.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(observed)))
    residuals = pd.DataFrame(
        {
            "row_index": row_labels,
            "observed": observed,
            "fitted": fitted_values,
            "residual": residual_values,
            "absolute_residual": np.abs(residual_values),
            "inverse_mills_ratio": np.asarray(fitted.inverse_mills_ratio, dtype=float),
        }
    )
    selection_coefficients = [
        HeckmanSelectionCoefficient(
            term=str(term),
            estimate=float(selection_result.params[term]),
            standard_error=float(selection_result.bse[term]),
            statistic=float(selection_result.tvalues[term]),
            p_value=float(selection_result.pvalues[term]),
        )
        for term in selection_result.params.index
    ]
    multicollinearity = calculate_heckman_multicollinearity(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    if result.fit_statistics.get("exclusion_restriction_count", 0) == 0:
        warnings.append("No exclusion restriction was supplied for the selection equation.")
    imr_p = result.fit_statistics.get("inverse_mills_p_value")
    if imr_p is not None and np.isfinite(float(imr_p)) and float(imr_p) < 0.05:
        warnings.append("The inverse Mills ratio is statistically significant.")
    warnings.extend(result.warnings)
    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "selected_sample_size": result.fit_statistics.get("selected_sample_size"),
        "selection_sample_size": result.fit_statistics.get("selection_sample_size"),
        "selection_rate": result.fit_statistics.get("selection_rate"),
        "outcome_r_squared": result.fit_statistics.get("outcome_r_squared"),
        "inverse_mills_coefficient": result.fit_statistics.get("inverse_mills_coefficient"),
        "inverse_mills_p_value": imr_p,
        "rho": result.fit_statistics.get("rho"),
        "exclusion_restriction_count": result.fit_statistics.get("exclusion_restriction_count"),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return HeckmanDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        selection_sample_size=int(result.fit_statistics.get("selection_sample_size", 0)),
        selection_coefficients=selection_coefficients,
        multicollinearity=multicollinearity,
        residuals=residuals,
        warnings=warnings,
        summary=summary,
    )


def heckman_selection_coefficients_to_dataframe(report: HeckmanDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.selection_coefficients])


def heckman_multicollinearity_to_dataframe(report: HeckmanDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def heckman_residuals_to_dataframe(report: HeckmanDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def heckman_diagnostic_summary_to_dataframe(report: HeckmanDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
