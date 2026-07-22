"""Truncated normal regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


@dataclass(slots=True)
class TruncatedRegressionRawResult:
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    tvalues: pd.Series
    fittedvalues: pd.Series
    resid: pd.Series
    model: Any
    covariance: np.ndarray
    log_likelihood: float
    converged: bool
    message: str

    def conf_int(self) -> pd.DataFrame:
        lower = self.params - 1.96 * self.bse
        upper = self.params + 1.96 * self.bse
        return pd.DataFrame({0: lower, 1: upper})


def _validate_limits(lower_limit: float | None, upper_limit: float | None) -> None:
    if lower_limit is None and upper_limit is None:
        raise ValueError("Truncated regression requires lower_limit, upper_limit, or both.")
    if lower_limit is not None and upper_limit is not None and lower_limit >= upper_limit:
        raise ValueError("Truncated regression lower_limit must be smaller than upper_limit.")


def _log_likelihood(
    parameters: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    *,
    lower_limit: float | None,
    upper_limit: float | None,
) -> float:
    beta = parameters[:-1]
    sigma = float(np.exp(parameters[-1]))
    mu = x @ beta
    z = (y - mu) / sigma
    log_density = stats.norm.logpdf(z) - np.log(sigma)

    if lower_limit is None:
        lower_probability = np.zeros_like(mu)
    else:
        lower_probability = stats.norm.cdf((lower_limit - mu) / sigma)
    if upper_limit is None:
        upper_probability = np.ones_like(mu)
    else:
        upper_probability = stats.norm.cdf((upper_limit - mu) / sigma)
    interval_probability = upper_probability - lower_probability
    if np.any(interval_probability <= 0) or np.any(~np.isfinite(interval_probability)):
        return -np.inf
    total = log_density - np.log(interval_probability)
    if not np.all(np.isfinite(total)):
        return -np.inf
    return float(np.sum(total))


def _expected_observed(
    mu: np.ndarray,
    sigma: float,
    *,
    lower_limit: float | None,
    upper_limit: float | None,
) -> np.ndarray:
    if lower_limit is None:
        a = np.full_like(mu, -np.inf)
        lower_probability = np.zeros_like(mu)
    else:
        a = (lower_limit - mu) / sigma
        lower_probability = stats.norm.cdf(a)
    if upper_limit is None:
        b = np.full_like(mu, np.inf)
        upper_probability = np.ones_like(mu)
    else:
        b = (upper_limit - mu) / sigma
        upper_probability = stats.norm.cdf(b)
    interval_probability = np.maximum(upper_probability - lower_probability, 1e-12)
    return mu + sigma * (stats.norm.pdf(a) - stats.norm.pdf(b)) / interval_probability


def fit_truncated_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    fixed_effects: list[str] | None = None,
    model_id: str = "truncated_regression_1",
    add_intercept: bool = True,
    maximum_iterations: int = 300,
) -> RegressionResult:
    """Fit a truncated normal regression model by maximum likelihood."""
    _validate_limits(lower_limit, upper_limit)
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Truncated regression",
    )
    y_series = design.outcome.astype(float)
    predictors = design.predictors.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")
    y = y_series.to_numpy(dtype=float)
    x = predictors.to_numpy(dtype=float)

    if lower_limit is not None and np.any(y <= lower_limit):
        raise ValueError("Truncated regression observations must be greater than lower_limit.")
    if upper_limit is not None and np.any(y >= upper_limit):
        raise ValueError("Truncated regression observations must be smaller than upper_limit.")

    ols_start = np.linalg.lstsq(x, y, rcond=None)[0]
    residual_sd = float(np.std(y - x @ ols_start, ddof=1))
    if not np.isfinite(residual_sd) or residual_sd <= 1e-8:
        residual_sd = float(np.std(y, ddof=1))
    if not np.isfinite(residual_sd) or residual_sd <= 1e-8:
        residual_sd = 1.0
    start = np.r_[ols_start, np.log(residual_sd)]

    def objective(parameters: np.ndarray) -> float:
        value = _log_likelihood(
            parameters,
            y,
            x,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        if not np.isfinite(value):
            return np.inf
        return -value

    fitted = optimize.minimize(
        objective,
        start,
        method="BFGS",
        options={"maxiter": maximum_iterations},
    )
    parameters = np.asarray(fitted.x, dtype=float)
    beta = parameters[:-1]
    sigma = float(np.exp(parameters[-1]))
    log_likelihood = -float(fitted.fun)
    covariance = np.asarray(getattr(fitted, "hess_inv", np.full((len(parameters), len(parameters)), np.nan)), dtype=float)
    if covariance.shape != (len(parameters), len(parameters)):
        covariance = np.full((len(parameters), len(parameters)), np.nan)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    statistic = np.divide(parameters, standard_errors, out=np.full_like(parameters, np.nan), where=standard_errors > 0)
    p_values = 2.0 * stats.norm.sf(np.abs(statistic))
    names = [str(column) for column in predictors.columns] + ["log_sigma"]
    param_series = pd.Series(parameters, index=names)
    bse_series = pd.Series(standard_errors, index=names)
    pvalue_series = pd.Series(p_values, index=names)
    statistic_series = pd.Series(statistic, index=names)
    confidence_intervals = pd.DataFrame(
        {0: param_series - 1.96 * bse_series, 1: param_series + 1.96 * bse_series}
    )

    mu = x @ beta
    expected = _expected_observed(
        mu,
        sigma,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    residuals = y - expected
    raw_result = TruncatedRegressionRawResult(
        params=param_series,
        bse=bse_series,
        pvalues=pvalue_series,
        tvalues=statistic_series,
        fittedvalues=pd.Series(expected, index=y_series.index),
        resid=pd.Series(residuals, index=y_series.index),
        model=type(
            "TruncatedRegressionModelData",
            (),
            {
                "endog": y,
                "exog": x,
                "exog_names": [str(column) for column in predictors.columns],
                "data": type("TruncatedRegressionModelRows", (), {"row_labels": y_series.index.tolist()})(),
            },
        )(),
        covariance=covariance,
        log_likelihood=log_likelihood,
        converged=bool(fitted.success),
        message=str(fitted.message),
    )

    coefficients: list[ModelCoefficient] = []
    for term in names:
        estimate = float(np.exp(param_series[term])) if term == "log_sigma" else float(param_series[term])
        lower = float(np.exp(confidence_intervals.loc[term, 0])) if term == "log_sigma" else float(confidence_intervals.loc[term, 0])
        upper = float(np.exp(confidence_intervals.loc[term, 1])) if term == "log_sigma" else float(confidence_intervals.loc[term, 1])
        coefficients.append(
            ModelCoefficient(
                term="sigma" if term == "log_sigma" else term,
                estimate=estimate,
                standard_error=float(bse_series[term]),
                statistic=float(statistic_series[term]),
                p_value=float(pvalue_series[term]),
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
            )
        )

    parameter_count = len(parameters)
    aic = 2 * parameter_count - 2 * log_likelihood
    bic = np.log(len(y)) * parameter_count - 2 * log_likelihood
    observed_variance = float(np.var(y, ddof=1)) if len(y) > 1 else np.nan
    residual_variance = float(np.var(residuals, ddof=1)) if len(y) > 1 else np.nan
    pseudo_r_squared = 1.0 - residual_variance / observed_variance if observed_variance > 0 else np.nan
    warnings: list[str] = []
    if not fitted.success:
        warnings.append("Truncated regression maximum likelihood optimization did not fully converge.")
    return RegressionResult(
        model_id=model_id,
        model_type="truncated_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(y)),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "bic": float(bic),
            "sigma": sigma,
            "pseudo_r_squared": float(pseudo_r_squared),
            "left_truncation_limit": lower_limit,
            "right_truncation_limit": upper_limit,
            "truncated_sample_count": int(len(y)),
            "parameter_count": parameter_count,
        },
        converged=bool(fitted.success),
        standard_error_type="maximum_likelihood_hessian",
        warnings=warnings,
        metadata={
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "add_intercept": add_intercept,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "left_truncated": bool(lower_limit is not None),
            "right_truncated": bool(upper_limit is not None),
            "latent_fitted_values": mu.tolist(),
            "expected_observed_values": expected.tolist(),
            "residuals": residuals.tolist(),
            **design.metadata,
        },
        raw_result=raw_result,
    )
