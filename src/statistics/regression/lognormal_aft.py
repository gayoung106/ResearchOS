"""Log-normal accelerated failure time regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.weibull_aft import (
    _add_intercept,
    _covariance_matrix,
    _prepare_survival_design,
)


@dataclass(slots=True)
class LogNormalAFTModelData:
    endog: np.ndarray
    status: np.ndarray
    exog: np.ndarray
    exog_names: list[str]
    row_labels: list[str]


@dataclass(slots=True)
class LogNormalAFTFittedResult:
    model: LogNormalAFTModelData
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    confidence_intervals: pd.DataFrame
    linear_predictor: np.ndarray
    fittedvalues: np.ndarray
    resid: np.ndarray
    sigma: float
    log_sigma: float
    llf: float
    nobs: int
    df_model: int
    converged: bool
    mle_retvals: dict[str, Any]

    def conf_int(self) -> pd.DataFrame:
        return self.confidence_intervals.copy()

    def predict(self, exog: np.ndarray | pd.DataFrame | None = None, *, kind: str = "median") -> np.ndarray:
        matrix = self.model.exog if exog is None else np.asarray(exog, dtype=float)
        eta = np.asarray(matrix, dtype=float) @ self.params.to_numpy(dtype=float)
        median = np.exp(eta)
        if kind == "median":
            return median
        if kind == "mean":
            return np.exp(eta + 0.5 * self.sigma**2)
        if kind == "linear":
            return eta
        if kind == "scale":
            return median
        raise ValueError("Log-normal AFT prediction kind must be 'median', 'mean', 'scale', or 'linear'.")

    def survival(self, time: np.ndarray, exog: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
        matrix = self.model.exog if exog is None else np.asarray(exog, dtype=float)
        eta = np.asarray(matrix, dtype=float) @ self.params.to_numpy(dtype=float)
        values = np.asarray(time, dtype=float)
        z = (np.log(values) - eta) / self.sigma
        return stats.norm.sf(z)


def _negative_log_likelihood(parameters: np.ndarray, duration: np.ndarray, event: np.ndarray, exog: np.ndarray) -> float:
    beta = parameters[:-1]
    log_sigma = float(parameters[-1])
    if not np.isfinite(log_sigma) or abs(log_sigma) > 20:
        return np.inf
    sigma = float(np.exp(log_sigma))
    log_duration = np.log(duration)
    eta = exog @ beta
    z = (log_duration - eta) / sigma
    log_density = stats.norm.logpdf(z) - np.log(duration) - log_sigma
    log_survival = stats.norm.logsf(z)
    loglike = event * log_density + (1 - event) * log_survival
    value = -float(np.sum(loglike))
    return value if np.isfinite(value) else np.inf


def _initial_parameters(duration: np.ndarray, exog: np.ndarray) -> np.ndarray:
    log_duration = np.log(duration)
    beta, *_ = np.linalg.lstsq(exog, log_duration, rcond=None)
    residual = log_duration - exog @ beta
    sigma = max(float(np.std(residual, ddof=min(1, max(len(residual) - 1, 0)))), 0.25)
    return np.append(beta, np.log(sigma))


def _fit_lognormal_parameters(
    duration: np.ndarray,
    event: np.ndarray,
    exog: np.ndarray,
    *,
    maximum_iterations: int,
) -> optimize.OptimizeResult:
    result = optimize.minimize(
        _negative_log_likelihood,
        _initial_parameters(duration, exog),
        args=(duration, event, exog),
        method="BFGS",
        options={"maxiter": maximum_iterations},
    )
    if not result.success:
        fallback = optimize.minimize(
            _negative_log_likelihood,
            result.x,
            args=(duration, event, exog),
            method="L-BFGS-B",
            bounds=[(None, None)] * exog.shape[1] + [(-6.0, 6.0)],
            options={"maxiter": maximum_iterations * 2},
        )
        if np.isfinite(fallback.fun) and fallback.fun <= result.fun:
            return fallback
    return result


def fit_lognormal_aft(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "lognormal_aft_1",
    add_intercept: bool = True,
    maximum_iterations: int = 500,
) -> RegressionResult:
    """Fit a log-normal accelerated failure time model for right-censored durations."""
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    duration, event, predictors, metadata = _prepare_survival_design(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Log-normal AFT regression",
    )
    if add_intercept:
        predictors = _add_intercept(predictors)

    duration_values = duration.to_numpy(dtype=float)
    event_values = event.to_numpy(dtype=int)
    exog = predictors.to_numpy(dtype=float)
    names = [str(column) for column in predictors.columns]
    fit = _fit_lognormal_parameters(
        duration_values,
        event_values,
        exog,
        maximum_iterations=maximum_iterations,
    )
    parameter_count = exog.shape[1] + 1
    covariance = _covariance_matrix(fit, parameter_count)
    beta = np.asarray(fit.x[:-1], dtype=float)
    log_sigma = float(fit.x[-1])
    sigma = float(np.exp(log_sigma))
    beta_covariance = covariance[: len(beta), : len(beta)]
    standard_errors = np.sqrt(np.where(np.diag(beta_covariance) >= 0, np.diag(beta_covariance), np.nan))
    z_values = np.divide(beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0)
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))
    lower = beta - 1.96 * standard_errors
    upper = beta + 1.96 * standard_errors
    coefficients = [
        ModelCoefficient(
            term=name,
            estimate=float(estimate),
            standard_error=float(se),
            statistic=float(z_value),
            p_value=float(p_value),
            confidence_interval_lower=float(ci_low),
            confidence_interval_upper=float(ci_high),
            exponentiated_estimate=float(np.exp(estimate)),
        )
        for name, estimate, se, z_value, p_value, ci_low, ci_high in zip(
            names,
            beta,
            standard_errors,
            z_values,
            p_values,
            lower,
            upper,
            strict=False,
        )
    ]
    linear_predictor = exog @ beta
    median_time = np.exp(linear_predictor)
    mean_time = np.exp(linear_predictor + 0.5 * sigma**2)
    residuals = np.log(duration_values) - linear_predictor
    llf = -float(fit.fun)
    null_exog = np.ones((len(duration_values), 1))
    null_fit = _fit_lognormal_parameters(
        duration_values,
        event_values,
        null_exog,
        maximum_iterations=maximum_iterations,
    )
    null_llf = -float(null_fit.fun)
    lr_stat = 2.0 * (llf - null_llf)
    df_model = max(len(beta) - int(add_intercept), 0)
    lr_p = float(stats.chi2.sf(lr_stat, df_model)) if df_model > 0 else np.nan
    event_count = int(event_values.sum())
    censored_count = int(len(event_values) - event_count)
    events_per_parameter = event_count / max(len(coefficients), 1)
    warnings: list[str] = []
    if events_per_parameter < 10:
        warnings.append("Log-normal AFT regression has fewer than 10 events per estimated coefficient.")
    if not fit.success:
        warnings.append("Log-normal AFT optimizer did not report formal convergence.")

    fitted = LogNormalAFTFittedResult(
        model=LogNormalAFTModelData(
            endog=duration_values,
            status=event_values,
            exog=exog,
            exog_names=names,
            row_labels=list(metadata["row_labels"]),
        ),
        params=pd.Series(beta, index=names),
        bse=pd.Series(standard_errors, index=names),
        tvalues=pd.Series(z_values, index=names),
        pvalues=pd.Series(p_values, index=names),
        confidence_intervals=pd.DataFrame({0: lower, 1: upper}, index=names),
        linear_predictor=linear_predictor,
        fittedvalues=median_time,
        resid=residuals,
        sigma=sigma,
        log_sigma=log_sigma,
        llf=llf,
        nobs=len(duration_values),
        df_model=df_model,
        converged=bool(fit.success),
        mle_retvals={"success": bool(fit.success), "message": str(fit.message)},
    )

    return RegressionResult(
        model_id=model_id,
        model_type="lognormal_aft",
        dependent_variable=duration_variable,
        independent_variables=independent_variables,
        sample_size=len(duration_values),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": llf,
            "null_log_likelihood": null_llf,
            "likelihood_ratio_statistic": float(lr_stat),
            "likelihood_ratio_p_value": lr_p,
            "aic": float(2 * parameter_count - 2 * llf),
            "bic": float(np.log(len(duration_values)) * parameter_count - 2 * llf),
            "event_count": event_count,
            "censored_count": censored_count,
            "event_rate": float(event_count / len(event_values)),
            "parameter_count": len(coefficients),
            "events_per_parameter": float(events_per_parameter),
            "sigma": sigma,
            "log_sigma": log_sigma,
            "median_predicted_time": float(np.median(median_time)),
            "mean_predicted_time": float(np.mean(mean_time)),
        },
        converged=bool(fit.success),
        standard_error_type="maximum_likelihood_hessian",
        warnings=warnings,
        metadata={
            **metadata,
            "duration_variable": duration_variable,
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "distribution": "lognormal",
            "parameterization": "accelerated_failure_time",
            "design_matrix_columns": names,
            "fixed_effect_column_count": len(metadata["fixed_effect_columns"]),
        },
        raw_result=fitted,
    )
