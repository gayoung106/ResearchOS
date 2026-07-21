"""Binary complementary log-log regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_binary_cloglog(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "cloglog_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a binary complementary log-log GLM."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type for binary cloglog: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="binary cloglog",
    )
    outcome = design.outcome
    predictors = design.predictors
    unique_outcomes = sorted(outcome.unique().tolist())
    if unique_outcomes != [0.0, 1.0]:
        raise ValueError(
            "Binary cloglog dependent variable must be coded 0 and 1. "
            f"Current values: {unique_outcomes}"
        )

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.GLM(
        outcome,
        predictors,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    )
    try:
        if covariance_type == "nonrobust":
            fitted = model.fit(maxiter=maximum_iterations)
        else:
            fitted = model.fit(maxiter=maximum_iterations, cov_type=covariance_type)
    except PerfectSeparationError as error:
        raise ValueError("Binary cloglog could not be estimated because of perfect separation.") from error

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

    converged = bool(getattr(fitted, "converged", True))
    warnings: list[str] = []
    if not converged:
        warnings.append("Binary cloglog model did not converge.")

    event_count = int(outcome.sum())
    non_event_count = int(len(outcome) - event_count)
    if min(event_count, non_event_count) < 10:
        warnings.append("Fewer than 10 events or non-events were available; estimates may be unstable.")
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("Sample size is very small relative to the number of estimated parameters.")

    fitted_probability = np.asarray(fitted.predict(), dtype=float)
    brier_score = float(np.mean((fitted_probability - np.asarray(outcome, dtype=float)) ** 2))
    null_ll = float(getattr(fitted, "llnull", np.nan))
    llf = float(fitted.llf)
    lr_stat = float(2.0 * (llf - null_ll)) if np.isfinite(null_ll) else np.nan
    df_model = float(getattr(fitted, "df_model", np.nan))
    lr_p = float(stats.chi2.sf(lr_stat, df_model)) if np.isfinite(lr_stat) and df_model > 0 else np.nan
    pseudo_r_squared = 1.0 - llf / null_ll if np.isfinite(null_ll) and not np.isclose(null_ll, 0.0) else np.nan

    return RegressionResult(
        model_id=model_id,
        model_type="binary_cloglog",
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
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "link": "cloglog",
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
