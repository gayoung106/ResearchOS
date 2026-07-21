"""Ordered probit regression."""

from __future__ import annotations

import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_ordered_probit(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "ordered_probit_1",
    maximum_iterations: int = 200,
) -> RegressionResult:
    """Fit an ordered probit model for ordinal dependent variables."""
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="ordered probit",
    )
    outcome = design.outcome
    predictors = design.predictors
    unique_outcomes = sorted(outcome.unique().tolist())
    if len(unique_outcomes) < 3:
        raise ValueError("Ordered probit dependent variable must have at least three categories.")

    model = OrderedModel(outcome, predictors, distr="probit")
    fitted = model.fit(method="bfgs", disp=False, maxiter=maximum_iterations)

    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    threshold_terms: list[str] = []
    for term in fitted.params.index:
        estimate = float(fitted.params[term])
        is_threshold = "/" in str(term)
        if is_threshold:
            threshold_terms.append(str(term))
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=estimate,
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    converged = bool(fitted.mle_retvals.get("converged", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("Ordered probit model did not converge.")

    category_counts = {
        str(category): int(count) for category, count in outcome.value_counts().sort_index().items()
    }
    if min(category_counts.values()) < 10:
        warnings.append("At least one ordinal outcome category has fewer than 10 observations.")

    parameter_count = len(predictors.columns) + len(unique_outcomes) - 1
    if len(outcome) <= parameter_count + 1:
        warnings.append("Sample size is very small relative to the number of estimated parameters.")

    return RegressionResult(
        model_id=model_id,
        model_type="ordered_probit",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "category_count": len(unique_outcomes),
        },
        converged=converged,
        standard_error_type="maximum_likelihood",
        warnings=warnings,
        metadata={
            "threshold_terms": threshold_terms,
            "category_counts": category_counts,
            "maximum_iterations": maximum_iterations,
            "link": "probit",
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
