"""Weibull proportional hazards regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.weibull_aft import fit_weibull_aft


def _transform_aft_to_ph_coefficient(coefficient: ModelCoefficient, shape: float) -> ModelCoefficient:
    estimate = -shape * coefficient.estimate
    standard_error = abs(shape) * coefficient.standard_error
    statistic = estimate / standard_error if standard_error > 0 and np.isfinite(standard_error) else np.nan
    p_value = float(2.0 * stats.norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    lower = estimate - 1.96 * standard_error if np.isfinite(standard_error) else np.nan
    upper = estimate + 1.96 * standard_error if np.isfinite(standard_error) else np.nan
    return ModelCoefficient(
        term=coefficient.term,
        estimate=float(estimate),
        standard_error=float(standard_error),
        statistic=float(statistic),
        p_value=p_value,
        confidence_interval_lower=float(lower),
        confidence_interval_upper=float(upper),
        exponentiated_estimate=float(np.exp(estimate)),
    )


def fit_weibull_ph(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "weibull_ph_1",
    add_intercept: bool = True,
    maximum_iterations: int = 500,
) -> RegressionResult:
    """Fit a Weibull proportional hazards model for right-censored durations."""
    aft = fit_weibull_aft(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_id=model_id,
        add_intercept=add_intercept,
        maximum_iterations=maximum_iterations,
    )
    shape = float(aft.fit_statistics["shape"])
    coefficients = [_transform_aft_to_ph_coefficient(coefficient, shape) for coefficient in aft.coefficients]
    baseline_log_rate = None
    baseline_rate = None
    for coefficient in coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            baseline_log_rate = coefficient.estimate
            baseline_rate = coefficient.exponentiated_estimate
            break

    fit_statistics = dict(aft.fit_statistics)
    fit_statistics.update(
        {
            "baseline_log_rate": baseline_log_rate,
            "baseline_rate": baseline_rate,
            "shape": shape,
        }
    )
    metadata = dict(aft.metadata)
    metadata.update(
        {
            "parameterization": "proportional_hazards",
            "source_parameterization": "accelerated_failure_time",
            "baseline_log_rate": baseline_log_rate,
            "baseline_rate": baseline_rate,
        }
    )
    warnings = [warning.replace("Weibull AFT", "Weibull PH") for warning in aft.warnings]

    return RegressionResult(
        model_id=model_id,
        model_type="weibull_ph",
        dependent_variable=duration_variable,
        independent_variables=aft.independent_variables,
        sample_size=aft.sample_size,
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=aft.converged,
        standard_error_type=aft.standard_error_type,
        warnings=warnings,
        metadata=metadata,
        raw_result=aft.raw_result,
    )
