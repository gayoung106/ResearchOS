"""Gamma GLM regression for positive continuous outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_gamma_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "gamma_regression_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a Gamma GLM with log link for strictly positive outcomes."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Gamma regression",
    )
    outcome = design.outcome.astype(float)
    if (outcome <= 0.0).any():
        raise ValueError("Gamma regression dependent variable must be strictly positive.")
    predictors = design.predictors.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    family = sm.families.Gamma(link=sm.families.links.Log())
    model = sm.GLM(outcome, predictors, family=family)
    if covariance_type == "nonrobust":
        fitted = model.fit(maxiter=maximum_iterations)
    else:
        fitted = model.fit(maxiter=maximum_iterations, cov_type=covariance_type)

    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        estimate = float(fitted.params[term])
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=estimate,
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )

    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(outcome, dtype=float) - predicted
    pseudo_r_squared = (
        1.0 - float(fitted.deviance / fitted.null_deviance)
        if fitted.null_deviance and fitted.null_deviance > 0
        else None
    )
    dispersion = float(fitted.pearson_chi2 / fitted.df_resid) if fitted.df_resid > 0 else None
    warnings: list[str] = []
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("The sample size is small relative to the number of estimated parameters.")
    if dispersion is not None and dispersion > 2.0:
        warnings.append("Gamma dispersion ratio is above 2.0; model fit should be reviewed.")

    return RegressionResult(
        model_id=model_id,
        model_type="gamma_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "deviance": float(fitted.deviance),
            "null_deviance": float(fitted.null_deviance),
            "pseudo_r_squared_deviance": pseudo_r_squared,
            "pearson_chi_square": float(fitted.pearson_chi2),
            "dispersion_ratio": dispersion,
            "mean_absolute_error": float(np.mean(np.abs(residuals))),
            "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
            "minimum_observed": float(np.min(outcome)),
            "maximum_observed": float(np.max(outcome)),
            "mean_prediction": float(np.mean(predicted)),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic_llf),
        },
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "link": "log",
            "family": "gamma",
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
