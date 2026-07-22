"""Linear probability model for binary outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix
from src.statistics.regression.ols import SUPPORTED_COVARIANCE_TYPES


def fit_linear_probability_model(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "linear_probability_model_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
) -> RegressionResult:
    """Fit an OLS linear probability model for a 0/1 dependent variable."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type for linear probability model: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="linear probability model",
    )
    outcome = design.outcome
    predictors = design.predictors
    unique_outcomes = sorted(outcome.unique().tolist())
    if unique_outcomes != [0.0, 1.0]:
        raise ValueError(
            "Linear probability model dependent variable must be coded 0 and 1. "
            f"Current values: {unique_outcomes}"
        )
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.OLS(outcome, predictors)
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

    observed = np.asarray(outcome, dtype=float)
    predicted = np.asarray(fitted.predict(), dtype=float)
    clipped = np.clip(predicted, 0.0, 1.0)
    out_of_bounds_count = int(((predicted < 0.0) | (predicted > 1.0)).sum())
    event_count = int(outcome.sum())
    non_event_count = int(len(outcome) - event_count)
    brier_score = float(np.mean((clipped - observed) ** 2))
    residuals = observed - predicted
    warnings: list[str] = []
    if out_of_bounds_count:
        warnings.append(
            f"Linear probability fitted values outside [0, 1] occurred for {out_of_bounds_count} observations."
        )
    if min(event_count, non_event_count) < 10:
        warnings.append("Fewer than 10 events or non-events were available; estimates may be unstable.")
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("Sample size is very small relative to the number of estimated parameters.")

    return RegressionResult(
        model_id=model_id,
        model_type="linear_probability_model",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "r_squared": float(fitted.rsquared),
            "adjusted_r_squared": float(fitted.rsquared_adj),
            "f_statistic": float(fitted.fvalue) if fitted.fvalue is not None else None,
            "f_p_value": float(fitted.f_pvalue) if fitted.f_pvalue is not None else None,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "event_count": event_count,
            "non_event_count": non_event_count,
            "brier_score": brier_score,
            "out_of_bounds_prediction_count": out_of_bounds_count,
            "mean_absolute_error": float(np.mean(np.abs(residuals))),
            "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
        },
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "link": "identity",
            "family": "linear_probability",
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
            "diagnostics": {
                "endog": observed.astype(int).tolist(),
                "predicted_probability": clipped.tolist(),
                "row_labels": list(design.outcome.index),
                "exog": np.asarray(predictors, dtype=float).tolist(),
                "exog_names": [str(column) for column in predictors.columns],
            },
        },
        raw_result=fitted,
    )
