"""Hurdle negative binomial count regression."""

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
        raise ValueError("Hurdle negative binomial dependent variable must be non-negative.")
    rounded = np.round(outcome)
    if not np.allclose(outcome, rounded):
        raise ValueError("Hurdle negative binomial dependent variable must contain integer counts.")
    if int(np.sum(rounded > 0)) == 0 or int(np.sum(rounded == 0)) == 0:
        raise ValueError("Hurdle negative binomial requires both zero and positive count outcomes.")
    return rounded.astype(float).to_numpy()


def _nb2_zero_probability(mu: np.ndarray, alpha: float) -> np.ndarray:
    safe_alpha = max(float(alpha), np.finfo(float).eps)
    return np.power(1.0 + safe_alpha * np.maximum(mu, 0.0), -1.0 / safe_alpha)


def _nb2_truncated_mean(mu: np.ndarray, alpha: float) -> np.ndarray:
    zero_probability = _nb2_zero_probability(mu, alpha)
    return mu / np.maximum(1.0 - zero_probability, 1e-12)


def _coefficient_rows(
    fitted: Any,
    *,
    prefix: str,
    skip_alpha: bool = False,
) -> list[ModelCoefficient]:
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        if skip_alpha and str(term).lower() == "alpha":
            continue
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
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )
    return coefficients


def fit_hurdle_negative_binomial(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "hurdle_negative_binomial_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 300,
) -> RegressionResult:
    """Fit a two-part hurdle negative binomial model."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="hurdle negative binomial",
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
        raise ValueError("Hurdle negative binomial positive-count hurdle has perfect separation.") from error

    positive_mask = outcome > 0
    positive_outcome = outcome[positive_mask]
    positive_predictors = predictors.loc[positive_mask]
    count_model = sm.NegativeBinomial(positive_outcome, positive_predictors, loglike_method="nb2")
    count_fit_options: dict[str, Any] = {"disp": False, "maxiter": maximum_iterations}
    if covariance_type != "nonrobust":
        count_fit_options["cov_type"] = covariance_type
    count_fitted = count_model.fit(**count_fit_options)

    alpha = float(count_fitted.params.get("alpha", np.nan))
    positive_probability = np.asarray(positive_fitted.predict(predictors), dtype=float)
    positive_mu = np.asarray(count_fitted.predict(predictors), dtype=float)
    predicted_positive_mean = _nb2_truncated_mean(positive_mu, alpha)
    predicted_mean = positive_probability * predicted_positive_mean
    predicted_zero_probability = 1.0 - positive_probability

    coefficients = [
        *_coefficient_rows(positive_fitted, prefix="hurdle"),
        *_coefficient_rows(count_fitted, prefix="count", skip_alpha=True),
    ]
    residual_df = max(len(outcome) - len(coefficients), 1)
    safe_alpha = max(alpha if np.isfinite(alpha) else 0.0, 0.0)
    variance = predicted_mean + safe_alpha * predicted_mean**2
    pearson = (outcome - predicted_mean) / np.sqrt(np.maximum(variance, 1e-12))
    dispersion_ratio = float(np.sum(pearson**2) / residual_df)
    zero_count = int(np.sum(outcome == 0))
    positive_count = int(np.sum(outcome > 0))

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
        count_fitted.mle_retvals.get("converged", False)
    )
    if not converged:
        warnings.append("Hurdle negative binomial model did not fully converge.")
    if np.isfinite(alpha) and alpha <= 0:
        warnings.append("Estimated hurdle negative binomial alpha is not positive.")

    return RegressionResult(
        model_id=model_id,
        model_type="hurdle_negative_binomial",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics={
            "positive_log_likelihood": float(positive_fitted.llf),
            "count_log_likelihood": float(count_fitted.llf),
            "log_likelihood": float(positive_fitted.llf + count_fitted.llf),
            "aic": float(positive_fitted.aic + count_fitted.aic),
            "bic": float(positive_fitted.bic + count_fitted.bic),
            "alpha": alpha,
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
            "count_model": "zero_truncated_negative_binomial_mean",
            "negative_binomial_parameterization": "NB2",
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
