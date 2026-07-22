"""Generalized Poisson count regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import GeneralizedPoisson

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix

SUPPORTED_COVARIANCE_TYPES = {"nonrobust", "HC0", "HC1", "HC2", "HC3"}


def fit_generalized_poisson(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "generalized_poisson_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 200,
    parameterization: int = 1,
) -> RegressionResult:
    """Fit a generalized Poisson count model."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")
    if parameterization not in {1, 2}:
        raise ValueError("Generalized Poisson parameterization must be 1 or 2.")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="generalized Poisson",
    )
    outcome = design.outcome
    predictors = design.predictors
    if (outcome < 0).any():
        raise ValueError("Generalized Poisson dependent variable must be non-negative.")
    rounded = np.round(outcome)
    if not np.allclose(outcome, rounded):
        raise ValueError("Generalized Poisson dependent variable must contain integer counts.")
    outcome = rounded.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = GeneralizedPoisson(outcome, predictors, p=parameterization)
    fit_options: dict[str, Any] = {
        "disp": False,
        "maxiter": maximum_iterations,
        "cov_type": covariance_type,
    }
    if covariance_type == "nonrobust":
        fit_options["cov_type"] = "nonrobust"
    fitted = model.fit(**fit_options)
    confidence_intervals = fitted.conf_int()

    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        if str(term).lower() == "alpha":
            continue
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

    predicted = np.asarray(fitted.predict(), dtype=float)
    residual_df = max(len(outcome) - len(coefficients), 1)
    pearson = (outcome.to_numpy(dtype=float) - predicted) / np.sqrt(np.maximum(predicted, 1e-12))
    dispersion_ratio = float(np.sum(pearson**2) / residual_df)
    zero_count = int((outcome == 0).sum())
    alpha = float(fitted.params.get("alpha", np.nan))
    converged = bool(fitted.mle_retvals.get("converged", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("Generalized Poisson model did not converge.")

    return RegressionResult(
        model_id=model_id,
        model_type="generalized_poisson",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "alpha": alpha,
            "dispersion_ratio": dispersion_ratio,
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": zero_count,
            "zero_proportion": float(zero_count / len(outcome)),
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "generalized_poisson_parameterization": parameterization,
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
