"""Weibull accelerated failure time regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)
from src.statistics.regression.design_matrix import _encode_fixed_effects, _validate_fixed_effects


@dataclass(slots=True)
class WeibullAFTModelData:
    endog: np.ndarray
    status: np.ndarray
    exog: np.ndarray
    exog_names: list[str]
    row_labels: list[str]


@dataclass(slots=True)
class WeibullAFTFittedResult:
    model: WeibullAFTModelData
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    confidence_intervals: pd.DataFrame
    linear_predictor: np.ndarray
    fittedvalues: np.ndarray
    resid: np.ndarray
    shape: float
    log_shape: float
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
        scale = np.exp(eta)
        if kind == "scale":
            return scale
        if kind == "mean":
            return scale * float(special.gamma(1.0 + 1.0 / self.shape))
        if kind == "linear":
            return eta
        if kind != "median":
            raise ValueError("Weibull AFT prediction kind must be 'median', 'mean', 'scale', or 'linear'.")
        return scale * float(np.log(2.0) ** (1.0 / self.shape))


def _prepare_survival_design(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str],
    model_label: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, Any]]:
    validate_model_variables(dataframe, duration_variable, independent_variables)
    if event_variable not in dataframe.columns:
        raise KeyError("Event variable is missing from dataframe: " + event_variable)
    if event_variable == duration_variable or event_variable in independent_variables:
        raise ValueError("Event variable cannot duplicate duration or predictor variables.")
    _validate_fixed_effects(
        dataframe,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    selected = dataframe[[duration_variable, event_variable, *independent_variables, *fixed_effects]].copy()
    selected[duration_variable] = pd.to_numeric(selected[duration_variable], errors="coerce")
    selected[event_variable] = pd.to_numeric(selected[event_variable], errors="coerce")
    for variable in independent_variables:
        selected[variable] = pd.to_numeric(selected[variable], errors="coerce")
    complete = selected.dropna()
    if complete.empty:
        raise ValueError(f"{model_label} has no complete observations to estimate.")
    if (complete[duration_variable] <= 0).any():
        raise ValueError(f"{model_label} duration values must be positive.")

    event_values = sorted(complete[event_variable].unique().tolist())
    if event_values != [0.0, 1.0]:
        raise ValueError(f"{model_label} event variable must be coded 0/1. Current values: {event_values}")
    if int(complete[event_variable].sum()) == 0:
        raise ValueError(f"{model_label} requires at least one observed event.")

    constant_predictors = [
        variable for variable in independent_variables if complete[variable].nunique() <= 1
    ]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    predictors = complete[independent_variables].astype(float).copy()
    predictors, fixed_effect_columns, reference_categories = _encode_fixed_effects(
        complete,
        predictors=predictors,
        fixed_effects=fixed_effects,
    )
    if predictors.empty:
        raise ValueError(f"{model_label} requires at least one predictor.")

    return (
        complete[duration_variable].astype(float),
        complete[event_variable].astype(int),
        predictors,
        {
            "event_variable": event_variable,
            "fixed_effects": fixed_effects,
            "fixed_effect_reference_categories": reference_categories,
            "fixed_effect_columns": fixed_effect_columns,
            "dropped_case_count": len(dataframe) - len(complete),
            "row_labels": [str(index) for index in complete.index],
        },
    )


def _add_intercept(predictors: pd.DataFrame) -> pd.DataFrame:
    output = predictors.copy()
    output.insert(0, "const", 1.0)
    return output


def _negative_log_likelihood(parameters: np.ndarray, duration: np.ndarray, event: np.ndarray, exog: np.ndarray) -> float:
    beta = parameters[:-1]
    log_shape = float(parameters[-1])
    if not np.isfinite(log_shape) or abs(log_shape) > 20:
        return np.inf
    shape = float(np.exp(log_shape))
    eta = exog @ beta
    log_duration = np.log(duration)
    hazard_power = np.exp(np.clip(shape * (log_duration - eta), -745.0, 700.0))
    loglike = event * (log_shape - shape * eta + (shape - 1.0) * log_duration) - hazard_power
    value = -float(np.sum(loglike))
    return value if np.isfinite(value) else np.inf


def _initial_parameters(duration: np.ndarray, exog: np.ndarray) -> np.ndarray:
    log_duration = np.log(duration)
    beta, *_ = np.linalg.lstsq(exog, log_duration, rcond=None)
    residual = log_duration - exog @ beta
    sigma = max(float(np.std(residual, ddof=min(1, max(len(residual) - 1, 0)))), 0.25)
    shape = 1.0 / sigma
    return np.append(beta, np.log(shape))


def _fit_weibull_parameters(
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


def _covariance_matrix(result: optimize.OptimizeResult, parameter_count: int) -> np.ndarray:
    inverse = getattr(result, "hess_inv", None)
    if inverse is None:
        return np.full((parameter_count, parameter_count), np.nan)
    if hasattr(inverse, "todense"):
        inverse = inverse.todense()
    covariance = np.asarray(inverse, dtype=float)
    if covariance.shape != (parameter_count, parameter_count):
        return np.full((parameter_count, parameter_count), np.nan)
    return covariance


def fit_weibull_aft(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "weibull_aft_1",
    add_intercept: bool = True,
    maximum_iterations: int = 500,
) -> RegressionResult:
    """Fit a Weibull accelerated failure time model for right-censored durations."""
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    duration, event, predictors, metadata = _prepare_survival_design(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Weibull AFT regression",
    )
    if add_intercept:
        predictors = _add_intercept(predictors)

    duration_values = duration.to_numpy(dtype=float)
    event_values = event.to_numpy(dtype=int)
    exog = predictors.to_numpy(dtype=float)
    names = [str(column) for column in predictors.columns]
    fit = _fit_weibull_parameters(
        duration_values,
        event_values,
        exog,
        maximum_iterations=maximum_iterations,
    )
    parameter_count = exog.shape[1] + 1
    covariance = _covariance_matrix(fit, parameter_count)
    beta = np.asarray(fit.x[:-1], dtype=float)
    log_shape = float(fit.x[-1])
    shape = float(np.exp(log_shape))
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
    median_time = np.exp(linear_predictor) * float(np.log(2.0) ** (1.0 / shape))
    residuals = np.log(duration_values) - linear_predictor
    llf = -float(fit.fun)
    null_exog = np.ones((len(duration_values), 1))
    null_fit = _fit_weibull_parameters(
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
    warnings: list[str] = []
    events_per_parameter = event_count / max(len(coefficients), 1)
    if events_per_parameter < 10:
        warnings.append("Weibull AFT regression has fewer than 10 events per estimated coefficient.")
    if not fit.success:
        warnings.append("Weibull AFT optimizer did not report formal convergence.")

    fitted = WeibullAFTFittedResult(
        model=WeibullAFTModelData(
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
        shape=shape,
        log_shape=log_shape,
        llf=llf,
        nobs=len(duration_values),
        df_model=df_model,
        converged=bool(fit.success),
        mle_retvals={"success": bool(fit.success), "message": str(fit.message)},
    )

    return RegressionResult(
        model_id=model_id,
        model_type="weibull_aft",
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
            "shape": shape,
            "log_shape": log_shape,
            "median_predicted_time": float(np.median(median_time)),
            "mean_predicted_time": float(np.mean(fitted.predict(kind="mean"))),
        },
        converged=bool(fit.success),
        standard_error_type="maximum_likelihood_hessian",
        warnings=warnings,
        metadata={
            **metadata,
            "duration_variable": duration_variable,
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "distribution": "weibull",
            "parameterization": "accelerated_failure_time",
            "design_matrix_columns": names,
            "fixed_effect_column_count": len(metadata["fixed_effect_columns"]),
        },
        raw_result=fitted,
    )
