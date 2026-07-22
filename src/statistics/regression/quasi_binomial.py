"""Quasi-binomial regression for overdispersed binary outcomes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_quasi_binomial(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "quasi_binomial_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a binomial-logit GLM with Pearson scale for quasi-binomial inference."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance_type for quasi-binomial regression: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="quasi-binomial",
    )
    outcome = design.outcome
    predictors = design.predictors
    unique_outcomes = sorted(outcome.unique().tolist())
    if unique_outcomes != [0.0, 1.0]:
        raise ValueError(
            "Quasi-binomial dependent variable must be coded 0 and 1. "
            f"Current values: {unique_outcomes}"
        )

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    model = sm.GLM(
        outcome,
        predictors,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
    )
    fit_options: dict[str, object] = {"maxiter": maximum_iterations, "scale": "X2"}
    if covariance_type != "nonrobust":
        fit_options["cov_type"] = covariance_type
    try:
        fitted = model.fit(**fit_options)
    except PerfectSeparationError as error:
        raise ValueError("Quasi-binomial model could not be estimated because of perfect separation.") from error

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
        warnings.append("Quasi-binomial model did not converge.")

    event_count = int(outcome.sum())
    non_event_count = int(len(outcome) - event_count)
    if min(event_count, non_event_count) < 10:
        warnings.append("Fewer than 10 events or non-events were available; estimates may be unstable.")
    if len(outcome) <= len(predictors.columns) + 1:
        warnings.append("Sample size is very small relative to the number of estimated parameters.")

    observed = np.asarray(outcome, dtype=float)
    predicted_probability = np.clip(np.asarray(fitted.predict(), dtype=float), 0.0, 1.0)
    brier_score = float(np.mean((predicted_probability - observed) ** 2))
    pearson_chi_square = float(getattr(fitted, "pearson_chi2", np.nan))
    residual_df = float(getattr(fitted, "df_resid", np.nan))
    dispersion_scale = float(getattr(fitted, "scale", np.nan))
    null_ll = float(getattr(fitted, "llnull", np.nan))
    llf = float(fitted.llf)
    lr_stat = float(2.0 * (llf - null_ll)) if np.isfinite(null_ll) else np.nan
    df_model = float(getattr(fitted, "df_model", np.nan))
    lr_p = float(stats.chi2.sf(lr_stat, df_model)) if np.isfinite(lr_stat) and df_model > 0 else np.nan
    pseudo_r_squared = 1.0 - llf / null_ll if np.isfinite(null_ll) and not np.isclose(null_ll, 0.0) else np.nan

    return RegressionResult(
        model_id=model_id,
        model_type="quasi_binomial",
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
            "dispersion_scale": dispersion_scale,
            "pearson_chi_square": pearson_chi_square,
            "residual_degrees_of_freedom": residual_df,
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "link": "logit",
            "family": "quasi_binomial",
            "scale": "pearson_chi_square",
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
            "diagnostics": {
                "endog": observed.astype(int).tolist(),
                "predicted_probability": predicted_probability.tolist(),
                "row_labels": list(design.outcome.index),
                "exog": np.asarray(predictors, dtype=float).tolist(),
                "exog_names": [str(column) for column in predictors.columns],
            },
        },
        raw_result=fitted,
    )
