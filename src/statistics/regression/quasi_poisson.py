"""Quasi-Poisson regression for overdispersed count outcomes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix
from src.statistics.regression.poisson import SUPPORTED_COVARIANCE_TYPES


def fit_quasi_poisson(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "quasi_poisson_1",
    covariance_type: str = "nonrobust",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a Poisson mean model with Pearson-scale quasi-likelihood standard errors."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="quasi-Poisson",
    )
    outcome = design.outcome
    predictors = design.predictors
    if (outcome < 0).any():
        raise ValueError("Quasi-Poisson dependent variable must be non-negative.")
    rounded = np.round(outcome)
    if not np.allclose(outcome, rounded):
        raise ValueError("Quasi-Poisson dependent variable must contain integer counts.")
    outcome = rounded.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.GLM(outcome, predictors, family=sm.families.Poisson())
    fit_options: dict[str, Any] = {"maxiter": maximum_iterations, "scale": "X2"}
    if covariance_type != "nonrobust":
        fit_options["cov_type"] = covariance_type
    fitted = model.fit(**fit_options)

    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        estimate = float(fitted.params[term])
        lower = float(confidence_intervals.loc[term, 0])
        upper = float(confidence_intervals.loc[term, 1])
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=estimate,
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )

    residual_degrees_of_freedom = float(fitted.df_resid)
    dispersion_ratio = (
        float(fitted.pearson_chi2 / residual_degrees_of_freedom)
        if residual_degrees_of_freedom > 0
        else math.nan
    )
    zero_count = int((outcome == 0).sum())
    zero_proportion = float(zero_count / len(outcome))
    deviance = float(fitted.deviance)
    null_deviance = float(fitted.null_deviance)
    pseudo_r_squared_deviance = (
        float(1 - deviance / null_deviance) if not np.isclose(null_deviance, 0.0) else None
    )
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(outcome, dtype=float) - predicted
    converged = bool(getattr(fitted, "converged", True))
    warnings: list[str] = []
    if not converged:
        warnings.append("Quasi-Poisson model did not converge.")
    if math.isfinite(dispersion_ratio) and dispersion_ratio <= 1.0:
        warnings.append("Quasi-Poisson dispersion is not above 1; ordinary Poisson may be sufficient.")

    return RegressionResult(
        model_id=model_id,
        model_type="quasi_poisson",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "deviance": deviance,
            "null_deviance": null_deviance,
            "pearson_chi_square": float(fitted.pearson_chi2),
            "dispersion_ratio": dispersion_ratio,
            "scale": float(fitted.scale),
            "pseudo_r_squared_deviance": pseudo_r_squared_deviance,
            "mean_absolute_error": float(np.mean(np.abs(residuals))),
            "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
            "residual_degrees_of_freedom": residual_degrees_of_freedom,
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": zero_count,
            "zero_proportion": zero_proportion,
        },
        converged=converged,
        standard_error_type=f"{covariance_type}_pearson_scale",
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "family": "quasi_poisson",
            "link": "log",
            "quasi_likelihood": True,
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
