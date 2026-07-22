"""Hurdle Poisson count regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix

SUPPORTED_COVARIANCE_TYPES = {"nonrobust", "HC0", "HC1", "HC2", "HC3"}


@dataclass(slots=True)
class _HurdleModelData:
    endog: np.ndarray
    exog: np.ndarray
    exog_names: list[str]


@dataclass(slots=True)
class _HurdleRawResult:
    model: _HurdleModelData
    positive_model: Any
    count_model: Any
    predicted_mean: np.ndarray
    predicted_zero_probability: np.ndarray

    def predict(self, which: str | None = None) -> np.ndarray:
        if which == "prob-zero":
            return self.predicted_zero_probability
        return self.predicted_mean


def _validate_count_outcome(outcome: pd.Series) -> np.ndarray:
    if (outcome < 0).any():
        raise ValueError("Hurdle Poisson dependent variable must be non-negative.")
    rounded = np.round(outcome)
    if not np.allclose(outcome, rounded):
        raise ValueError("Hurdle Poisson dependent variable must contain integer counts.")
    if int(np.sum(rounded > 0)) == 0 or int(np.sum(rounded == 0)) == 0:
        raise ValueError("Hurdle Poisson requires both zero and positive count outcomes.")
    return rounded.astype(float).to_numpy()


def _poisson_truncated_mean(mu: np.ndarray) -> np.ndarray:
    zero_probability = np.exp(-np.clip(mu, 0.0, 700.0))
    denominator = np.maximum(1.0 - zero_probability, 1e-12)
    return mu / denominator


def _coefficient_rows(
    fitted: Any,
    *,
    prefix: str,
    exponentiate: bool,
) -> list[ModelCoefficient]:
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        estimate = float(fitted.params[term])
        lower = float(confidence_intervals.loc[term, 0])
        upper = float(confidence_intervals.loc[term, 1])
        coefficients.append(
            ModelCoefficient(
                term=f"{prefix}:{term}",
                estimate=estimate,
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=float(np.exp(estimate)) if exponentiate else None,
            )
        )
    return coefficients


def fit_hurdle_poisson(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "hurdle_poisson_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 200,
) -> RegressionResult:
    """Fit a two-part hurdle Poisson model.

    The first part models whether the count crosses the zero hurdle with a logit
    model. The second part models positive counts with a Poisson GLM and uses
    the zero-truncated Poisson mean for full-sample predictions.
    """
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="hurdle Poisson",
    )
    outcome = _validate_count_outcome(design.outcome)
    predictors = design.predictors
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    positive_indicator = (outcome > 0).astype(float)
    positive_model = sm.Logit(positive_indicator, predictors)
    fit_options: dict[str, Any] = {"disp": False, "maxiter": maximum_iterations}
    if covariance_type != "nonrobust":
        fit_options["cov_type"] = covariance_type
    try:
        positive_fitted = positive_model.fit(**fit_options)
    except PerfectSeparationError as error:
        raise ValueError("Hurdle Poisson positive-count hurdle has perfect separation.") from error

    positive_mask = outcome > 0
    positive_outcome = outcome[positive_mask]
    positive_predictors = predictors.loc[positive_mask]
    count_model = sm.GLM(
        positive_outcome,
        positive_predictors,
        family=sm.families.Poisson(),
    )
    count_fit_options: dict[str, Any] = {"maxiter": maximum_iterations, "disp": False}
    if covariance_type != "nonrobust":
        count_fit_options["cov_type"] = covariance_type
    count_fitted = count_model.fit(**count_fit_options)

    positive_probability = np.asarray(positive_fitted.predict(predictors), dtype=float)
    positive_mu = np.asarray(count_fitted.predict(predictors), dtype=float)
    predicted_positive_mean = _poisson_truncated_mean(positive_mu)
    predicted_mean = positive_probability * predicted_positive_mean
    predicted_zero_probability = 1.0 - positive_probability

    coefficients = [
        *_coefficient_rows(positive_fitted, prefix="hurdle", exponentiate=True),
        *_coefficient_rows(count_fitted, prefix="count", exponentiate=True),
    ]
    residual_df = max(len(outcome) - len(coefficients), 1)
    pearson = (outcome - predicted_mean) / np.sqrt(np.maximum(predicted_mean, 1e-12))
    dispersion_ratio = float(np.sum(pearson**2) / residual_df)
    zero_count = int(np.sum(outcome == 0))
    positive_count = int(np.sum(outcome > 0))
    count_bic = getattr(count_fitted, "bic_llf", None)
    if count_bic is None:
        count_bic = count_fitted.bic

    raw_result = _HurdleRawResult(
        model=_HurdleModelData(
            endog=outcome,
            exog=np.asarray(predictors, dtype=float),
            exog_names=[str(column) for column in predictors.columns],
        ),
        positive_model=positive_fitted,
        count_model=count_fitted,
        predicted_mean=predicted_mean,
        predicted_zero_probability=predicted_zero_probability,
    )

    warnings: list[str] = []
    converged = bool(positive_fitted.mle_retvals.get("converged", False)) and bool(
        getattr(count_fitted, "converged", True)
    )
    if not converged:
        warnings.append("Hurdle Poisson model did not fully converge.")

    return RegressionResult(
        model_id=model_id,
        model_type="hurdle_poisson",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics={
            "positive_log_likelihood": float(positive_fitted.llf),
            "count_log_likelihood": float(count_fitted.llf),
            "log_likelihood": float(positive_fitted.llf + count_fitted.llf),
            "aic": float(positive_fitted.aic + count_fitted.aic),
            "bic": float(positive_fitted.bic + count_bic),
            "dispersion_ratio": dispersion_ratio,
            "outcome_mean": float(np.mean(outcome)),
            "outcome_variance": float(np.var(outcome, ddof=1)),
            "zero_count": zero_count,
            "positive_count": positive_count,
            "zero_proportion": float(zero_count / len(outcome)),
            "predicted_zero_proportion": float(np.mean(predicted_zero_probability)),
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "hurdle_model": "logit",
            "count_model": "zero_truncated_poisson_mean",
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
            "diagnostics": {
                "endog": outcome.tolist(),
                "predicted_mean": predicted_mean.tolist(),
                "row_labels": list(design.outcome.index),
                "exog": np.asarray(predictors, dtype=float).tolist(),
                "exog_names": [str(column) for column in predictors.columns],
            },
        },
        raw_result=raw_result,
    )
