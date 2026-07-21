"""Discrete-time hazard survival regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.piecewise_exponential import _prepare_piecewise_data

SUPPORTED_DISCRETE_HAZARD_LINKS = {"logit", "cloglog"}


def _binomial_family(link: str) -> sm.families.Binomial:
    if link == "logit":
        return sm.families.Binomial(link=sm.families.links.Logit())
    if link == "cloglog":
        return sm.families.Binomial(link=sm.families.links.CLogLog())
    raise ValueError("Discrete-time hazard link must be 'logit' or 'cloglog'.")


def _inverse_link(values: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-values))
    return 1.0 - np.exp(-np.exp(values))


def _baseline_interval_hazards(fitted: Any, metadata: dict[str, Any], link: str) -> list[dict[str, float | str]]:
    params = fitted.params
    rows: list[dict[str, float | str]] = []
    survival = 1.0
    for index, label in enumerate(metadata["interval_labels"]):
        term = "baseline_interval_" + str(label)
        linear_predictor = float(params.get(term, np.nan))
        hazard = float(_inverse_link(np.asarray([linear_predictor], dtype=float), link)[0]) if np.isfinite(linear_predictor) else np.nan
        if np.isfinite(hazard):
            survival *= max(1.0 - hazard, 0.0)
        rows.append(
            {
                "interval_index": float(index),
                "interval": str(label),
                "start": float(metadata["interval_starts"][index]),
                "stop": float(metadata["interval_stops"][index]),
                "linear_predictor": linear_predictor,
                "baseline_hazard_probability": hazard,
                "baseline_survival": float(survival),
            }
        )
    return rows


def fit_discrete_time_hazard_model(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    breakpoints: list[float] | None = None,
    link: str = "logit",
    model_id: str = "discrete_time_hazard_1",
    covariance_type: str = "HC3",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a discrete-time hazard model on person-period data."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")
    link = str(link).lower().strip()
    if link not in SUPPORTED_DISCRETE_HAZARD_LINKS:
        raise ValueError("Discrete-time hazard link must be 'logit' or 'cloglog'.")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    long_data, predictors, outcome, metadata = _prepare_piecewise_data(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        breakpoints=breakpoints,
    )
    model = sm.GLM(outcome, predictors, family=_binomial_family(link))
    try:
        if covariance_type == "nonrobust":
            fitted = model.fit(maxiter=maximum_iterations)
        else:
            fitted = model.fit(maxiter=maximum_iterations, cov_type=covariance_type)
    except PerfectSeparationError as error:
        raise ValueError("Discrete-time hazard model could not be estimated because of perfect separation.") from error

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

    event_count = int(outcome.sum())
    censored_count = int(len(metadata["row_labels"]) - event_count)
    covariate_parameter_count = sum(not coefficient.term.startswith("baseline_interval_") for coefficient in coefficients)
    predicted = np.asarray(fitted.predict(), dtype=float)
    brier_score = float(np.mean((predicted - np.asarray(outcome, dtype=float)) ** 2))
    null_ll = float(getattr(fitted, "llnull", np.nan))
    llf = float(fitted.llf)
    lr_stat = float(2.0 * (llf - null_ll)) if np.isfinite(null_ll) else np.nan
    df_model = float(getattr(fitted, "df_model", np.nan))
    lr_p = float(stats.chi2.sf(lr_stat, df_model)) if np.isfinite(lr_stat) and df_model > 0 else np.nan
    pseudo_r_squared = 1.0 - llf / null_ll if np.isfinite(null_ll) and not np.isclose(null_ll, 0.0) else np.nan
    interval_hazards = _baseline_interval_hazards(fitted, metadata, link)

    warnings: list[str] = []
    if event_count / max(covariate_parameter_count, 1) < 10:
        warnings.append("Discrete-time hazard model has fewer than 10 events per covariate parameter.")
    if event_count < 10 or censored_count < 10:
        warnings.append("Discrete-time hazard model has fewer than 10 events or censored observations.")

    return RegressionResult(
        model_id=model_id,
        model_type="discrete_time_hazard",
        dependent_variable=duration_variable,
        independent_variables=independent_variables,
        sample_size=int(len(metadata["row_labels"])),
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
            "censored_count": censored_count,
            "event_rate": float(event_count / len(metadata["row_labels"])),
            "interval_count": metadata["interval_count"],
            "long_row_count": metadata["long_row_count"],
            "person_period_event_rate": float(event_count / len(outcome)),
            "brier_score": brier_score,
            "parameter_count": len(coefficients),
            "covariate_parameter_count": int(covariate_parameter_count),
            "events_per_parameter": float(event_count / max(covariate_parameter_count, 1)),
        },
        converged=bool(getattr(fitted, "converged", True)),
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            **metadata,
            "family": "binomial",
            "link": link,
            "maximum_iterations": maximum_iterations,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(metadata["fixed_effect_columns"]),
            "baseline_interval_hazards": interval_hazards,
            "long_data": long_data[["_row_index", "_interval_index", "_interval", "_interval_start", "_interval_stop", "_exposure", "_event"]].to_dict(orient="records"),
        },
        raw_result=fitted,
    )
