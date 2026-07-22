"""Box-Cox transformed OLS regression."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix
from src.statistics.regression.ols import SUPPORTED_COVARIANCE_TYPES


def _boxcox_transform(values: np.ndarray, lambda_value: float) -> np.ndarray:
    if np.isclose(lambda_value, 0.0):
        return np.log(values)
    return (np.power(values, lambda_value) - 1.0) / lambda_value


def _boxcox_inverse(values: np.ndarray, lambda_value: float) -> np.ndarray:
    if np.isclose(lambda_value, 0.0):
        return np.exp(values)
    base = lambda_value * values + 1.0
    return np.power(np.maximum(base, 1e-12), 1.0 / lambda_value)


def fit_boxcox_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "boxcox_regression_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    lambda_value: float | None = None,
) -> RegressionResult:
    """Fit OLS after estimating or applying a Box-Cox transformation to a positive outcome."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type for Box-Cox regression: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Box-Cox regression",
    )
    outcome = design.outcome.astype(float)
    if (outcome <= 0).any():
        raise ValueError("Box-Cox regression requires a strictly positive dependent variable.")

    predictors = design.predictors.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    y = outcome.to_numpy(dtype=float)
    estimated_lambda = float(stats.boxcox_normmax(y)) if lambda_value is None else float(lambda_value)
    transformed = pd.Series(_boxcox_transform(y, estimated_lambda), index=outcome.index, name=dependent_variable)
    model = sm.OLS(transformed, predictors)
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

    transformed_fitted = np.asarray(fitted.fittedvalues, dtype=float)
    transformed_residuals = np.asarray(fitted.resid, dtype=float)
    original_scale_fitted = _boxcox_inverse(transformed_fitted, estimated_lambda)
    original_scale_residuals = y - original_scale_fitted
    warnings: list[str] = []
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("The sample size is very small relative to the number of estimated parameters.")
    if not np.isfinite(estimated_lambda):
        warnings.append("Estimated Box-Cox lambda is not finite.")

    fit_statistics: dict[str, Any] = {
        "boxcox_lambda": estimated_lambda,
        "lambda_estimated": lambda_value is None,
        "transformed_r_squared": float(fitted.rsquared),
        "adjusted_transformed_r_squared": float(fitted.rsquared_adj),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "f_statistic": (
            float(fitted.fvalue)
            if (fitted.fvalue is not None and not math.isnan(float(fitted.fvalue)))
            else None
        ),
        "f_p_value": (
            float(fitted.f_pvalue)
            if (fitted.f_pvalue is not None and not math.isnan(float(fitted.f_pvalue)))
            else None
        ),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "residual_degrees_of_freedom": float(fitted.df_resid),
        "original_scale_root_mean_squared_error": float(np.sqrt(np.mean(original_scale_residuals**2))),
    }

    return RegressionResult(
        model_id=model_id,
        model_type="boxcox_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "boxcox_lambda": estimated_lambda,
            "lambda_estimated": lambda_value is None,
            "transformed_outcome": transformed.tolist(),
            "original_outcome": y.tolist(),
            "transformed_fitted_values": transformed_fitted.tolist(),
            "transformed_residuals": transformed_residuals.tolist(),
            "original_scale_fitted_values": original_scale_fitted.tolist(),
            "original_scale_residuals": original_scale_residuals.tolist(),
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
