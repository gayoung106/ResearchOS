"""Robust linear regression using M-estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix

_SUPPORTED_NORMS = {
    "huber": sm.robust.norms.HuberT,
    "huber_t": sm.robust.norms.HuberT,
    "tukey": sm.robust.norms.TukeyBiweight,
    "bisquare": sm.robust.norms.TukeyBiweight,
    "andrews": sm.robust.norms.AndrewWave,
}


def _norm_from_name(name: str) -> Any:
    key = name.strip().lower()
    if key not in _SUPPORTED_NORMS:
        raise ValueError("Robust regression norm must be huber, tukey, bisquare, or andrews.")
    return _SUPPORTED_NORMS[key]()


def _lookup_value(values: Any, term: str, index: int) -> float:
    if hasattr(values, "loc"):
        return float(values.loc[term])
    return float(np.asarray(values, dtype=float)[index])


def _confidence_interval_value(confidence_intervals: Any, term: str, index: int, column: int) -> float:
    if hasattr(confidence_intervals, "loc"):
        return float(confidence_intervals.loc[term, column])
    return float(np.asarray(confidence_intervals, dtype=float)[index, column])


def fit_robust_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "robust_regression_1",
    norm: str = "huber",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a robust linear model with M-estimation weights."""
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Robust regression",
    )
    outcome = design.outcome.astype(float)
    predictors = design.predictors.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.RLM(outcome, predictors, M=_norm_from_name(norm))
    fitted = model.fit(maxiter=maximum_iterations)
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for index, term in enumerate([str(name) for name in fitted.params.index]):
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=_lookup_value(fitted.params, term, index),
                standard_error=_lookup_value(fitted.bse, term, index),
                statistic=_lookup_value(fitted.tvalues, term, index),
                p_value=_lookup_value(fitted.pvalues, term, index),
                confidence_interval_lower=_confidence_interval_value(
                    confidence_intervals, term, index, 0
                ),
                confidence_interval_upper=_confidence_interval_value(
                    confidence_intervals, term, index, 1
                ),
            )
        )

    residuals = np.asarray(fitted.resid, dtype=float)
    weights = np.asarray(fitted.weights, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    observed = np.asarray(outcome, dtype=float)
    ss_resid = float(np.sum(residuals**2))
    ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
    pseudo_r_squared = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan
    downweighted = weights < 0.999
    heavily_downweighted = weights < 0.5
    iteration_count = len(getattr(fitted, "fit_history", {}).get("deviance", []))
    warnings: list[str] = []
    if iteration_count >= maximum_iterations:
        warnings.append("Robust regression reached the maximum iteration count.")
    if float(np.mean(heavily_downweighted)) > 0.2:
        warnings.append("More than 20% of observations received robust weights below 0.5.")

    return RegressionResult(
        model_id=model_id,
        model_type="robust_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics={
            "pseudo_r_squared": float(pseudo_r_squared),
            "scale": float(fitted.scale),
            "deviance": float(getattr(fitted, "deviance", np.nan)),
            "downweighted_count": int(downweighted.sum()),
            "heavily_downweighted_count": int(heavily_downweighted.sum()),
            "downweighted_rate": float(np.mean(downweighted)),
            "iteration_count": int(iteration_count),
            "residual_degrees_of_freedom": float(getattr(fitted, "df_resid", np.nan)),
        },
        converged=iteration_count < maximum_iterations,
        standard_error_type="robust_m_estimator",
        warnings=warnings,
        metadata={
            "norm": norm,
            "add_intercept": add_intercept,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "row_labels": [str(index) for index in outcome.index],
            "fitted_values": fitted_values.tolist(),
            "residuals": residuals.tolist(),
            "robust_weights": weights.tolist(),
            **design.metadata,
        },
        raw_result=fitted,
    )
