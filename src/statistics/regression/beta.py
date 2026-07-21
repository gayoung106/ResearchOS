"""Beta regression for continuous proportion outcomes in (0, 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_beta_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "beta_regression_1",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit beta regression for outcomes strictly between 0 and 1."""
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="beta regression",
    )
    outcome = design.outcome.astype(float)
    if ((outcome <= 0.0) | (outcome >= 1.0)).any():
        raise ValueError(
            "Beta regression dependent variable must be strictly between 0 and 1. "
            "Use fractional_logit when boundary values are present."
        )

    predictors = design.predictors
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    fitted = BetaModel(outcome, predictors).fit(disp=False, maxiter=maximum_iterations)
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
    converged = bool(fitted.mle_retvals.get("converged", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("Beta regression did not converge.")
    if len(outcome) <= len(predictors.columns) + 2:
        warnings.append("The sample size is small relative to the number of beta regression parameters.")

    precision = float(np.exp(fitted.params["precision"])) if "precision" in fitted.params.index else None
    return RegressionResult(
        model_id=model_id,
        model_type="beta_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "pseudo_r_squared": float(fitted.prsquared),
            "mean_absolute_error": float(np.mean(np.abs(residuals))),
            "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
            "precision": precision,
            "parameter_count": len(coefficients),
        },
        converged=converged,
        standard_error_type="maximum_likelihood",
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
            "precision_parameter": "precision",
        },
        raw_result=fitted,
    )
