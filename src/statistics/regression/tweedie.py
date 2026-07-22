"""Tweedie GLM regression for non-negative outcomes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_tweedie_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "tweedie_regression_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
    variance_power: float = 1.5,
) -> RegressionResult:
    """Fit a Tweedie GLM with log link for non-negative outcomes."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")
    if not 1.0 < float(variance_power) < 2.0:
        raise ValueError("Tweedie variance_power must be between 1 and 2 for compound Poisson-Gamma outcomes.")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Tweedie regression",
    )
    outcome = design.outcome.astype(float)
    if (outcome < 0.0).any():
        raise ValueError("Tweedie regression dependent variable must be non-negative.")
    predictors = design.predictors.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    family = sm.families.Tweedie(var_power=float(variance_power), link=sm.families.links.Log())
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
    observed = np.asarray(outcome, dtype=float)
    residuals = observed - predicted
    pseudo_r_squared = (
        1.0 - float(fitted.deviance / fitted.null_deviance)
        if fitted.null_deviance and fitted.null_deviance > 0
        else None
    )
    dispersion = float(fitted.pearson_chi2 / fitted.df_resid) if fitted.df_resid > 0 else None
    zero_count = int(np.sum(np.isclose(observed, 0.0)))
    log_likelihood = float(fitted.llf) if math.isfinite(float(fitted.llf)) else None
    aic = float(fitted.aic) if math.isfinite(float(fitted.aic)) else None
    bic = getattr(fitted, "bic_llf", None)
    bic_value = float(bic) if bic is not None and math.isfinite(float(bic)) else None

    warnings: list[str] = []
    if dispersion is not None and dispersion > 2.0:
        warnings.append("Tweedie dispersion ratio is above 2.0; model fit should be reviewed.")
    if zero_count == 0:
        warnings.append("Tweedie regression was fitted without observed zero outcomes; Gamma regression may also be considered.")

    return RegressionResult(
        model_id=model_id,
        model_type="tweedie_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": log_likelihood,
            "deviance": float(fitted.deviance),
            "null_deviance": float(fitted.null_deviance),
            "pseudo_r_squared_deviance": pseudo_r_squared,
            "pearson_chi_square": float(fitted.pearson_chi2),
            "dispersion_ratio": dispersion,
            "mean_absolute_error": float(np.mean(np.abs(residuals))),
            "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
            "minimum_observed": float(np.min(observed)),
            "maximum_observed": float(np.max(observed)),
            "mean_prediction": float(np.mean(predicted)),
            "zero_count": zero_count,
            "zero_proportion": float(zero_count / len(observed)),
            "aic": aic,
            "bic": bic_value,
        },
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "link": "log",
            "family": "tweedie",
            "variance_power": float(variance_power),
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
