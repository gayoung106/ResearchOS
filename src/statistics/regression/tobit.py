"""Tobit censored normal regression."""

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
class TobitRawResult:
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
        raise ValueError("Tobit regression requires lower_limit, upper_limit, or both.")
    if lower_limit is not None and upper_limit is not None and lower_limit >= upper_limit:
        raise ValueError("Tobit lower_limit must be smaller than upper_limit.")


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
    uncensored = np.ones(y.shape, dtype=bool)
    total = np.zeros(y.shape, dtype=float)

    if lower_limit is not None:
        left = y <= lower_limit + 1e-10
        z_left = (lower_limit - mu[left]) / sigma
        total[left] = stats.norm.logcdf(z_left)
        uncensored &= ~left
    if upper_limit is not None:
        right = y >= upper_limit - 1e-10
        z_right = (upper_limit - mu[right]) / sigma
        total[right] = stats.norm.logsf(z_right)
        uncensored &= ~right

    z = (y[uncensored] - mu[uncensored]) / sigma
    total[uncensored] = stats.norm.logpdf(z) - np.log(sigma)
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
    if lower_limit is None and upper_limit is None:
        return mu
    if lower_limit is None:
        b = (upper_limit - mu) / sigma  # type: ignore[operator]
        return mu * stats.norm.cdf(b) - sigma * stats.norm.pdf(b) + upper_limit * stats.norm.sf(b)  # type: ignore[operator]
    if upper_limit is None:
        a = (lower_limit - mu) / sigma
        return lower_limit * stats.norm.cdf(a) + mu * stats.norm.sf(a) + sigma * stats.norm.pdf(a)
    a = (lower_limit - mu) / sigma
    b = (upper_limit - mu) / sigma
    middle = mu * (stats.norm.cdf(b) - stats.norm.cdf(a))
    middle += sigma * (stats.norm.pdf(a) - stats.norm.pdf(b))
    return lower_limit * stats.norm.cdf(a) + middle + upper_limit * stats.norm.sf(b)


def fit_tobit_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    fixed_effects: list[str] | None = None,
    model_id: str = "tobit_regression_1",
    add_intercept: bool = True,
    maximum_iterations: int = 300,
) -> RegressionResult:
    """Fit a censored normal Tobit model by maximum likelihood."""
    _validate_limits(lower_limit, upper_limit)
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Tobit",
    )
    y_series = design.outcome.astype(float)
    predictors = design.predictors.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")
    y = y_series.to_numpy(dtype=float)
    x = predictors.to_numpy(dtype=float)

    left_censored = np.zeros(len(y), dtype=bool)
    right_censored = np.zeros(len(y), dtype=bool)
    if lower_limit is not None:
        left_censored = y <= lower_limit + 1e-10
    if upper_limit is not None:
        right_censored = y >= upper_limit - 1e-10
    uncensored = ~(left_censored | right_censored)
    if not uncensored.any():
        raise ValueError("Tobit regression requires at least one uncensored observation.")
    if not (left_censored | right_censored).any():
        raise ValueError("Tobit regression requires at least one censored observation.")

    ols_start = np.linalg.lstsq(x[uncensored], y[uncensored], rcond=None)[0]
    residual_sd = float(np.std(y[uncensored] - x[uncensored] @ ols_start, ddof=1))
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
    raw_result = TobitRawResult(
        params=param_series,
        bse=bse_series,
        pvalues=pvalue_series,
        tvalues=statistic_series,
        fittedvalues=pd.Series(expected, index=y_series.index),
        resid=pd.Series(residuals, index=y_series.index),
        model=type(
            "TobitModelData",
            (),
            {
                "endog": y,
                "exog": x,
                "exog_names": [str(column) for column in predictors.columns],
                "data": type("TobitModelRows", (), {"row_labels": y_series.index.tolist()})(),
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
    censoring_rate = float(np.mean(left_censored | right_censored))
    warnings: list[str] = []
    if not fitted.success:
        warnings.append("Tobit maximum likelihood optimization did not fully converge.")
    if censoring_rate > 0.8:
        warnings.append("More than 80% of observations are censored; estimates may be unstable.")

    return RegressionResult(
        model_id=model_id,
        model_type="tobit_regression",
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
            "left_censored_count": int(left_censored.sum()),
            "right_censored_count": int(right_censored.sum()),
            "uncensored_count": int(uncensored.sum()),
            "censoring_rate": censoring_rate,
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
            "left_censored": left_censored.tolist(),
            "right_censored": right_censored.tolist(),
            "latent_fitted_values": mu.tolist(),
            "expected_observed_values": expected.tolist(),
            "residuals": residuals.tolist(),
            **design.metadata,
        },
        raw_result=raw_result,
    )
