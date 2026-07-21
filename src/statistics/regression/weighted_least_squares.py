"""Weighted least squares regression."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix
from src.statistics.regression.ols import SUPPORTED_COVARIANCE_TYPES


def fit_weighted_least_squares(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    weight_variable: str,
    fixed_effects: list[str] | None = None,
    model_id: str = "wls_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
) -> RegressionResult:
    """Fit weighted least squares using a positive analytic weight variable."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type for weighted least squares: {covariance_type}")
    if weight_variable not in dataframe.columns:
        raise KeyError(f"Weight variable is missing from dataframe: {weight_variable}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    requested = list(dict.fromkeys([dependent_variable, *independent_variables, *fixed_effects, weight_variable]))
    working = dataframe[requested].copy()
    working[weight_variable] = pd.to_numeric(working[weight_variable], errors="coerce")
    valid_weight = np.isfinite(working[weight_variable].astype(float)) & (working[weight_variable].astype(float) > 0)
    invalid_weight_count = int((~valid_weight).sum())
    working = working.loc[valid_weight].copy()
    if working.empty:
        raise ValueError("Weighted least squares requires at least one positive finite weight.")

    design = prepare_regression_design_matrix(
        working,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="weighted least squares",
    )
    outcome = design.outcome
    predictors = design.predictors
    weights = working.loc[outcome.index, weight_variable].astype(float)
    if weights.empty:
        raise ValueError("Weighted least squares has no complete cases after aligning weights.")

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.WLS(outcome, predictors, weights=weights)
    if covariance_type == "nonrobust":
        fitted = model.fit()
    else:
        fitted = model.fit(cov_type=covariance_type)

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

    warnings: list[str] = []
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("Sample size is very small relative to the number of estimated parameters.")
    weight_ratio = float(weights.max() / weights.min())
    if weight_ratio > 20:
        warnings.append("The largest WLS weight is more than 20 times the smallest weight.")

    fit_statistics: dict[str, Any] = {
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "f_statistic": (
            float(fitted.fvalue)
            if fitted.fvalue is not None and not math.isnan(float(fitted.fvalue))
            else None
        ),
        "f_p_value": (
            float(fitted.f_pvalue)
            if fitted.f_pvalue is not None and not math.isnan(float(fitted.f_pvalue))
            else None
        ),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "residual_degrees_of_freedom": float(fitted.df_resid),
        "weight_sum": float(weights.sum()),
        "weight_mean": float(weights.mean()),
        "weight_minimum": float(weights.min()),
        "weight_maximum": float(weights.max()),
        "weight_ratio": weight_ratio,
        "invalid_weight_count": invalid_weight_count,
    }

    metadata = {
        "add_intercept": add_intercept,
        "weight_variable": weight_variable,
        **design.metadata,
        "dropped_case_count": int(design.metadata.get("dropped_case_count", 0)) + invalid_weight_count,
        "design_matrix_columns": [str(column) for column in predictors.columns],
        "fixed_effect_column_count": len(design.fixed_effect_columns),
    }

    return RegressionResult(
        model_id=model_id,
        model_type="weighted_least_squares",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata=metadata,
        raw_result=fitted,
    )
