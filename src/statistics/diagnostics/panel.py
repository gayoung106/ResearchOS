"""Diagnostics for panel fixed-effects regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class PanelEntityResidual:
    entity: str
    observation_count: int
    residual_mean: float
    residual_std: float
    max_abs_residual: float


@dataclass(slots=True)
class PanelHausmanDiagnostic:
    fixed_model_id: str
    random_model_id: str
    shared_terms: list[str]
    statistic: float
    degrees_of_freedom: int
    p_value: float
    status: str
    recommendation: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PanelDiagnosticsReport:
    model_id: str
    model_type: str
    sample_size: int
    entity_count: int
    time_period_count: int | None
    multicollinearity: list[MulticollinearityResult]
    entity_residuals: list[PanelEntityResidual]
    residuals: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_panel_result(result: RegressionResult) -> None:
    if result.model_type not in {"panel_fixed_effects", "panel_random_effects", "panel_correlated_random_effects", "panel_between_effects", "panel_first_difference", "panel_pooled_ols"}:
        raise ValueError("Panel diagnostics require a panel regression result.")


def calculate_panel_multicollinearity(result: RegressionResult) -> list[MulticollinearityResult]:
    _validate_panel_result(result)
    exog = np.asarray(result.metadata.get("within_predictors", []), dtype=float)
    names = [str(name) for name in result.metadata.get("within_predictor_names", [])]
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
            interpretation = "Severe within-panel multicollinearity is suspected."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "Within-panel multicollinearity should be reviewed."
        else:
            status = "PASS"
            interpretation = "Within-panel VIF is within the usual screening threshold."
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


def _coefficient_map(result: RegressionResult) -> dict[str, Any]:
    return {coefficient.term: coefficient for coefficient in result.coefficients}


def build_panel_hausman_diagnostic(
    fixed_result: RegressionResult,
    random_result: RegressionResult,
) -> PanelHausmanDiagnostic:
    if fixed_result.model_type != "panel_fixed_effects":
        raise ValueError("Hausman diagnostics require a panel fixed-effects result as the first argument.")
    if random_result.model_type != "panel_random_effects":
        raise ValueError("Hausman diagnostics require a panel random-effects result as the second argument.")

    fixed_coefficients = _coefficient_map(fixed_result)
    random_coefficients = _coefficient_map(random_result)
    shared_terms = [
        term
        for term in fixed_result.independent_variables
        if term in fixed_coefficients and term in random_coefficients
    ]
    if not shared_terms:
        raise ValueError("Hausman diagnostics require at least one shared slope coefficient.")

    differences = np.asarray(
        [fixed_coefficients[term].estimate - random_coefficients[term].estimate for term in shared_terms],
        dtype=float,
    )
    variance_difference = np.asarray(
        [
            fixed_coefficients[term].standard_error**2
            - random_coefficients[term].standard_error**2
            for term in shared_terms
        ],
        dtype=float,
    )
    warnings: list[str] = []
    if np.any(~np.isfinite(variance_difference)) or np.any(np.isclose(variance_difference, 0.0)):
        warnings.append("Hausman variance difference required numerical stabilization.")
    if np.any(variance_difference <= 0):
        warnings.append("Hausman covariance difference was not positive; absolute diagonal approximation was used.")
    stabilized_variance = np.maximum(np.abs(variance_difference), 1e-12)
    statistic = float(differences @ np.diag(1.0 / stabilized_variance) @ differences)
    degrees_of_freedom = len(shared_terms)
    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    if p_value < 0.05:
        status = "WARNING"
        recommendation = "Prefer fixed effects when correlated unit effects are plausible; random-effects consistency is rejected."
    else:
        status = "PASS"
        recommendation = "Random-effects consistency was not rejected; report the Hausman comparison with FE/RE estimates."

    return PanelHausmanDiagnostic(
        fixed_model_id=fixed_result.model_id,
        random_model_id=random_result.model_id,
        shared_terms=shared_terms,
        statistic=statistic,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        status=status,
        recommendation=recommendation,
        warnings=warnings,
    )


def build_panel_diagnostics(result: RegressionResult) -> PanelDiagnosticsReport:
    _validate_panel_result(result)
    residuals = np.asarray(result.metadata.get("within_residuals", []), dtype=float)
    fitted = np.asarray(result.metadata.get("within_fitted_values", []), dtype=float)
    observed = np.asarray(result.metadata.get("within_outcome", []), dtype=float)
    row_labels = result.metadata.get("row_labels") or list(range(len(residuals)))
    entity_labels = [str(value) for value in result.metadata.get("entity_labels", [])]
    time_labels = result.metadata.get("time_labels")
    values: dict[str, Any] = {
        "row_index": row_labels,
        "entity": entity_labels,
        "within_observed": observed,
        "within_fitted": fitted,
        "within_residual": residuals,
        "absolute_residual": np.abs(residuals),
    }
    if time_labels is not None:
        values["time"] = time_labels
    residual_frame = pd.DataFrame(values)
    entity_residuals: list[PanelEntityResidual] = []
    for entity, group in residual_frame.groupby("entity", sort=True):
        entity_residuals.append(
            PanelEntityResidual(
                entity=str(entity),
                observation_count=int(len(group)),
                residual_mean=float(group["within_residual"].mean()),
                residual_std=float(group["within_residual"].std(ddof=1)) if len(group) > 1 else 0.0,
                max_abs_residual=float(group["absolute_residual"].max()),
            )
        )

    multicollinearity = calculate_panel_multicollinearity(result)
    warnings = [
        f"{item.variable_name}: {item.interpretation}"
        for item in multicollinearity
        if item.status in {"WARNING", "FAIL"}
    ]
    warnings.extend(result.warnings)
    singleton_count = int(result.fit_statistics.get("singleton_entity_count", 0) or 0)
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")

    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": result.sample_size,
        "entity_count": result.fit_statistics.get("entity_count"),
        "time_period_count": result.fit_statistics.get("time_period_count"),
        "within_r_squared": result.fit_statistics.get("within_r_squared"),
        "adjusted_within_r_squared": result.fit_statistics.get("adjusted_within_r_squared"),
        "marginal_r_squared": result.fit_statistics.get("marginal_r_squared"),
        "between_r_squared": result.fit_statistics.get("between_r_squared"),
        "first_difference_r_squared": result.fit_statistics.get("first_difference_r_squared"),
        "pooled_r_squared": result.fit_statistics.get("pooled_r_squared"),
        "adjusted_pooled_r_squared": result.fit_statistics.get("adjusted_pooled_r_squared"),
        "adjusted_first_difference_r_squared": result.fit_statistics.get("adjusted_first_difference_r_squared"),
        "adjusted_between_r_squared": result.fit_statistics.get("adjusted_between_r_squared"),
        "conditional_r_squared": result.fit_statistics.get("conditional_r_squared"),
        "entity_mean_term_count": result.fit_statistics.get("entity_mean_term_count"),
        "singleton_entity_count": singleton_count,
        "residual_mean": float(np.mean(residuals)) if residuals.size else np.nan,
        "residual_std": float(np.std(residuals, ddof=1)) if residuals.size > 1 else np.nan,
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
    }
    return PanelDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=result.sample_size,
        entity_count=int(result.fit_statistics.get("entity_count", 0)),
        time_period_count=result.fit_statistics.get("time_period_count"),
        multicollinearity=multicollinearity,
        entity_residuals=entity_residuals,
        residuals=residual_frame,
        warnings=warnings,
        summary=summary,
    )


def panel_multicollinearity_to_dataframe(report: PanelDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def panel_entity_residuals_to_dataframe(report: PanelDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.entity_residuals])


def panel_residuals_to_dataframe(report: PanelDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def panel_hausman_to_dataframe(report: PanelHausmanDiagnostic) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixed_model_id": report.fixed_model_id,
                "random_model_id": report.random_model_id,
                "shared_terms": ", ".join(report.shared_terms),
                "statistic": report.statistic,
                "degrees_of_freedom": report.degrees_of_freedom,
                "p_value": report.p_value,
                "status": report.status,
                "recommendation": report.recommendation,
                "warning_count": len(report.warnings),
            }
        ]
    )


def panel_diagnostic_summary_to_dataframe(report: PanelDiagnosticsReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
