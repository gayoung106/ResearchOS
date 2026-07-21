"""Log-binomial regression for binary risk ratios."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def _fit_model(
    model: sm.GLM,
    *,
    covariance_type: str,
    maximum_iterations: int,
) -> object:
    if covariance_type == "nonrobust":
        return model.fit(maxiter=maximum_iterations)
    return model.fit(maxiter=maximum_iterations, cov_type=covariance_type)


def _fit_model_with_optimizer_fallback(
    model: sm.GLM,
    *,
    covariance_type: str,
    maximum_iterations: int,
) -> tuple[object, bool]:
    try:
        return (
            _fit_model(
                model,
                covariance_type=covariance_type,
                maximum_iterations=maximum_iterations,
            ),
            False,
        )
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if covariance_type == "nonrobust":
                fitted = model.fit(method="lbfgs", maxiter=maximum_iterations * 2, disp=0)
            else:
                fitted = model.fit(
                    method="lbfgs",
                    maxiter=maximum_iterations * 2,
                    disp=0,
                    cov_type=covariance_type,
                )
        return fitted, True


def _model_converged(fitted: object) -> bool:
    if hasattr(fitted, "converged"):
        return bool(fitted.converged)
    mle_retvals = getattr(fitted, "mle_retvals", {})
    if isinstance(mle_retvals, dict) and "converged" in mle_retvals:
        return bool(mle_retvals["converged"])
    return True


def fit_log_binomial(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "log_binomial_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a binary GLM with binomial family and log link."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type for log-binomial regression: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="log-binomial",
    )
    outcome = design.outcome
    predictors = design.predictors
    unique_outcomes = sorted(outcome.unique().tolist())
    if unique_outcomes != [0.0, 1.0]:
        raise ValueError(
            "Log-binomial dependent variable must be coded 0 and 1. "
            f"Current values: {unique_outcomes}"
        )

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.GLM(
        outcome,
        predictors,
        family=sm.families.Binomial(link=sm.families.links.Log()),
    )
    try:
        fitted, used_optimizer_fallback = _fit_model_with_optimizer_fallback(
            model,
            covariance_type=covariance_type,
            maximum_iterations=maximum_iterations,
        )
    except PerfectSeparationError as error:
        raise ValueError("Log-binomial model could not be estimated because of perfect separation.") from error

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

    converged = _model_converged(fitted)
    warnings: list[str] = []
    if used_optimizer_fallback:
        warnings.append("Log-binomial IRLS estimation failed; L-BFGS optimizer fallback was used.")
    if not converged:
        warnings.append("Log-binomial model did not converge.")

    event_count = int(outcome.sum())
    non_event_count = int(len(outcome) - event_count)
    if min(event_count, non_event_count) < 10:
        warnings.append("Fewer than 10 events or non-events were available; estimates may be unstable.")
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("Sample size is very small relative to the number of estimated parameters.")

    fitted_probability = np.asarray(fitted.predict(), dtype=float)
    out_of_bounds_count = int(((fitted_probability < 0.0) | (fitted_probability > 1.0)).sum())
    if out_of_bounds_count:
        warnings.append(
            f"Log-binomial fitted probabilities outside [0, 1] occurred for {out_of_bounds_count} observations."
        )
    bounded_probability = np.clip(fitted_probability, 0.0, 1.0)
    brier_score = float(np.mean((bounded_probability - np.asarray(outcome, dtype=float)) ** 2))
    null_ll = float(getattr(fitted, "llnull", np.nan))
    llf = float(fitted.llf)
    lr_stat = float(2.0 * (llf - null_ll)) if np.isfinite(null_ll) else np.nan
    df_model = float(getattr(fitted, "df_model", np.nan))
    lr_p = float(stats.chi2.sf(lr_stat, df_model)) if np.isfinite(lr_stat) and df_model > 0 else np.nan
    pseudo_r_squared = 1.0 - llf / null_ll if np.isfinite(null_ll) and not np.isclose(null_ll, 0.0) else np.nan

    return RegressionResult(
        model_id=model_id,
        model_type="log_binomial",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": llf,
            "null_log_likelihood": null_ll,
            "likelihood_ratio_statistic": lr_stat,
            "likelihood_ratio_p_value": lr_p,
            "pseudo_r_squared_mcfadden": float(pseudo_r_squared),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic_llf),
            "event_count": event_count,
            "non_event_count": non_event_count,
            "brier_score": brier_score,
            "out_of_bounds_prediction_count": out_of_bounds_count,
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "link": "log",
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
