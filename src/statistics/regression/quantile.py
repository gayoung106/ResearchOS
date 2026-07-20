"""Quantile regression implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def _validate_quantile(quantile: float) -> float:
    value = float(quantile)
    if value <= 0.0 or value >= 1.0:
        raise ValueError("Quantile regression requires 0 < quantile < 1.")
    return value


def _pinball_loss(residuals: np.ndarray, quantile: float) -> float:
    return float(np.mean(np.maximum(quantile * residuals, (quantile - 1.0) * residuals)))


def fit_quantile_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "quantile_regression_1",
    quantile: float = 0.5,
    add_intercept: bool = True,
    maximum_iterations: int = 1000,
) -> RegressionResult:
    """Fit a linear quantile regression model."""
    quantile = _validate_quantile(quantile)
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="quantile regression",
    )
    outcome = design.outcome
    predictors = design.predictors
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    fitted = sm.QuantReg(outcome, predictors).fit(q=quantile, max_iter=maximum_iterations)
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.params[term]),
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    residuals = np.asarray(fitted.resid, dtype=float)
    warning_messages: list[str] = []
    iterations = getattr(fitted, "iterations", None)
    if iterations is not None and int(iterations) >= maximum_iterations:
        warning_messages.append("Quantile regression reached the maximum iteration limit.")
    if len(outcome) <= len(predictors.columns) + 1:
        warning_messages.append("The sample size is small relative to the number of estimated parameters.")

    return RegressionResult(
        model_id=model_id,
        model_type="quantile_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "quantile": quantile,
            "pseudo_r_squared": float(fitted.prsquared),
            "pinball_loss": _pinball_loss(residuals, quantile),
            "residual_degrees_of_freedom": float(fitted.df_resid),
            "iteration_count": int(iterations) if iterations is not None else None,
        },
        converged=not warning_messages or all("maximum iteration" not in item for item in warning_messages),
        standard_error_type="asymptotic_quantile",
        warnings=warning_messages,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
