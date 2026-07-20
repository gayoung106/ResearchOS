"""Multinomial logistic regression implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import _encode_fixed_effects, _validate_fixed_effects


def _ordered_categories(series: pd.Series) -> list[Any]:
    categories = series.dropna().drop_duplicates().tolist()
    try:
        return sorted(categories)
    except TypeError:
        return sorted(categories, key=lambda value: str(value))


def _prepare_multinomial_design(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str],
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    _validate_fixed_effects(
        dataframe,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    selected = dataframe[[dependent_variable, *independent_variables, *fixed_effects]].copy()
    for variable in independent_variables:
        selected[variable] = pd.to_numeric(selected[variable], errors="coerce")

    complete = selected.dropna()
    if complete.empty:
        raise ValueError("Multinomial logit has no complete observations to estimate.")

    categories = _ordered_categories(complete[dependent_variable])
    if len(categories) < 3:
        raise ValueError("Multinomial logit requires at least three outcome categories.")

    outcome = pd.Series(
        pd.Categorical(complete[dependent_variable], categories=categories).codes,
        index=complete.index,
        name=dependent_variable,
        dtype=float,
    )
    predictors = complete[independent_variables].astype(float).copy()
    if predictors.empty:
        raise ValueError("Multinomial logit requires at least one predictor.")

    constant_predictors = [
        variable for variable in independent_variables if complete[variable].nunique() <= 1
    ]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    predictors, fixed_effect_columns, reference_categories = _encode_fixed_effects(
        complete,
        predictors=predictors,
        fixed_effects=fixed_effects,
    )

    category_labels = [str(category) for category in categories]
    category_counts = {
        str(category): int(count)
        for category, count in complete[dependent_variable].value_counts().sort_index().items()
    }

    return outcome, predictors, {
        "category_labels": category_labels,
        "reference_category": category_labels[0],
        "category_counts": category_counts,
        "fixed_effects": fixed_effects,
        "fixed_effect_reference_categories": reference_categories,
        "fixed_effect_columns": fixed_effect_columns,
        "dropped_case_count": len(dataframe) - len(complete),
        "row_labels": [str(index) for index in complete.index],
    }


def fit_multinomial_logit(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "multinomial_logit_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a nominal outcome multinomial logit model."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    outcome, predictors, metadata = _prepare_multinomial_design(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.MNLogit(outcome, predictors)
    try:
        fit_kwargs = {"disp": False, "maxiter": maximum_iterations}
        if covariance_type != "nonrobust":
            fit_kwargs["cov_type"] = covariance_type
        fitted = model.fit(**fit_kwargs)
    except PerfectSeparationError as error:
        raise ValueError("Multinomial logit could not be estimated due to perfect separation.") from error

    confidence_intervals = fitted.conf_int()
    category_labels = list(metadata["category_labels"])
    coefficients: list[ModelCoefficient] = []

    for raw_column in fitted.params.columns:
        category_index = int(raw_column) + 1
        category_label = category_labels[category_index]
        for term in fitted.params.index:
            estimate = float(fitted.params.loc[term, raw_column])
            interval = confidence_intervals.loc[(str(category_index), term)]
            coefficients.append(
                ModelCoefficient(
                    term=f"{category_label}::{term}",
                    estimate=estimate,
                    standard_error=float(fitted.bse.loc[term, raw_column]),
                    statistic=float(fitted.tvalues.loc[term, raw_column]),
                    p_value=float(fitted.pvalues.loc[term, raw_column]),
                    confidence_interval_lower=float(interval["lower"]),
                    confidence_interval_upper=float(interval["upper"]),
                    exponentiated_estimate=float(np.exp(estimate)),
                )
            )

    converged = bool(fitted.mle_retvals.get("converged", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("Multinomial logit did not converge.")
    if min(metadata["category_counts"].values()) < 10:
        warnings.append("At least one outcome category has fewer than 10 observations.")
    parameter_count = len(coefficients)
    if int(fitted.nobs) <= parameter_count + 1:
        warnings.append("The sample size is small relative to the number of multinomial parameters.")

    fit_statistics = {
        "log_likelihood": float(fitted.llf),
        "null_log_likelihood": float(fitted.llnull),
        "likelihood_ratio_statistic": float(fitted.llr),
        "likelihood_ratio_p_value": float(fitted.llr_pvalue),
        "pseudo_r_squared_mcfadden": float(fitted.prsquared),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "category_count": len(category_labels),
        "parameter_count": parameter_count,
    }

    return RegressionResult(
        model_id=model_id,
        model_type="multinomial_logit",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            **metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(metadata["fixed_effect_columns"]),
        },
        raw_result=fitted,
    )
