"""Piecewise exponential survival regression."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)
from src.statistics.regression.binary_logit import SUPPORTED_COVARIANCE_TYPES
from src.statistics.regression.design_matrix import _encode_fixed_effects, _validate_fixed_effects


def _validate_breakpoints(breakpoints: list[float] | None, durations: pd.Series, events: pd.Series) -> list[float]:
    maximum_duration = float(durations.max())
    if breakpoints is None:
        event_times = durations[events == 1]
        if event_times.shape[0] >= 4:
            candidates = np.quantile(event_times, [0.25, 0.5, 0.75]).tolist()
        else:
            candidates = np.quantile(durations, [0.25, 0.5, 0.75]).tolist()
    else:
        candidates = [float(value) for value in breakpoints]
    output = sorted({float(value) for value in candidates if 0.0 < float(value) < maximum_duration})
    if not output:
        midpoint = maximum_duration / 2.0
        output = [midpoint] if midpoint > 0.0 else []
    return output


def _interval_label(start: float, stop: float) -> str:
    return f"({start:.6g}, {stop:.6g}]"


def _prepare_piecewise_data(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str],
    breakpoints: list[float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    validate_model_variables(dataframe, duration_variable, independent_variables)
    if event_variable not in dataframe.columns:
        raise KeyError("Event variable is missing from dataframe: " + event_variable)
    if event_variable == duration_variable or event_variable in independent_variables:
        raise ValueError("Event variable cannot duplicate the duration or predictor variables.")
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
        raise ValueError("Piecewise exponential regression has no complete observations to estimate.")
    if (complete[duration_variable] <= 0).any():
        raise ValueError("Piecewise exponential duration values must be positive.")
    event_values = sorted(complete[event_variable].unique().tolist())
    if event_values != [0.0, 1.0]:
        raise ValueError(f"Piecewise exponential event variable must be coded 0/1. Current values: {event_values}")
    if int(complete[event_variable].sum()) == 0:
        raise ValueError("Piecewise exponential regression requires at least one observed event.")

    interval_breaks = _validate_breakpoints(breakpoints, complete[duration_variable], complete[event_variable].astype(int))
    cutpoints = [0.0, *interval_breaks, float(complete[duration_variable].max())]
    rows: list[dict[str, Any]] = []
    for row_index, row in complete.iterrows():
        duration = float(row[duration_variable])
        event = int(row[event_variable])
        for interval_index, start in enumerate(cutpoints[:-1]):
            stop = cutpoints[interval_index + 1]
            if duration <= start:
                break
            observed_stop = min(duration, stop)
            exposure = observed_stop - start
            if exposure <= 0.0:
                continue
            interval_event = int(event == 1 and duration <= stop)
            record = {
                "_row_index": str(row_index),
                "_interval_index": interval_index,
                "_interval": _interval_label(start, stop),
                "_interval_start": start,
                "_interval_stop": stop,
                "_exposure": exposure,
                "_event": interval_event,
            }
            for variable in independent_variables:
                record[variable] = float(row[variable])
            for variable in fixed_effects:
                record[variable] = row[variable]
            rows.append(record)
            if interval_event == 1:
                break
    long_data = pd.DataFrame(rows)
    if long_data.empty:
        raise ValueError("Piecewise exponential regression produced no person-period rows.")

    predictors = long_data[independent_variables].astype(float).copy()
    predictors, fixed_effect_columns, reference_categories = _encode_fixed_effects(
        long_data,
        predictors=predictors,
        fixed_effects=fixed_effects,
    )
    interval_dummies = pd.get_dummies(long_data["_interval"], prefix="baseline_interval", dtype=float)
    predictors = pd.concat([predictors, interval_dummies], axis=1).astype(float)
    outcome = long_data["_event"].astype(float)

    metadata = {
        "duration_variable": duration_variable,
        "event_variable": event_variable,
        "fixed_effects": fixed_effects,
        "fixed_effect_reference_categories": reference_categories,
        "fixed_effect_columns": fixed_effect_columns,
        "dropped_case_count": len(dataframe) - len(complete),
        "row_labels": [str(index) for index in complete.index],
        "long_row_count": int(len(long_data)),
        "interval_breakpoints": interval_breaks,
        "interval_count": int(len(cutpoints) - 1),
        "interval_labels": [_interval_label(cutpoints[index], cutpoints[index + 1]) for index in range(len(cutpoints) - 1)],
        "interval_starts": [float(value) for value in cutpoints[:-1]],
        "interval_stops": [float(value) for value in cutpoints[1:]],
    }
    return long_data, predictors, outcome, metadata


def _baseline_interval_hazards(fitted: Any, metadata: dict[str, Any]) -> list[dict[str, float | str]]:
    params = fitted.params
    rows: list[dict[str, float | str]] = []
    cumulative_hazard = 0.0
    for index, label in enumerate(metadata["interval_labels"]):
        term = "baseline_interval_" + str(label)
        start = float(metadata["interval_starts"][index])
        stop = float(metadata["interval_stops"][index])
        width = stop - start
        log_hazard = float(params.get(term, np.nan))
        hazard = float(np.exp(log_hazard)) if np.isfinite(log_hazard) else np.nan
        if np.isfinite(hazard):
            cumulative_hazard += hazard * width
        rows.append(
            {
                "interval_index": float(index),
                "interval": str(label),
                "start": start,
                "stop": stop,
                "width": width,
                "log_baseline_hazard": log_hazard,
                "baseline_hazard": hazard,
                "baseline_cumulative_hazard": float(cumulative_hazard),
                "baseline_survival": float(np.exp(-cumulative_hazard)),
            }
        )
    return rows


def fit_piecewise_exponential_regression(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    breakpoints: list[float] | None = None,
    model_id: str = "piecewise_exponential_1",
    covariance_type: str = "HC3",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a piecewise exponential survival model via Poisson GLM with exposure offsets."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")
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
    offset = np.log(long_data["_exposure"].astype(float))
    model = sm.GLM(outcome, predictors, family=sm.families.Poisson(), offset=offset)
    if covariance_type == "nonrobust":
        fitted = model.fit(maxiter=maximum_iterations)
    else:
        fitted = model.fit(maxiter=maximum_iterations, cov_type=covariance_type)

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
    censored_count = int(metadata["row_labels"].__len__() - event_count)
    covariate_parameter_count = sum(not coefficient.term.startswith("baseline_interval_") for coefficient in coefficients)
    residual_df = float(fitted.df_resid)
    dispersion_ratio = float(fitted.pearson_chi2 / residual_df) if residual_df > 0 else math.nan
    interval_hazards = _baseline_interval_hazards(fitted, metadata)
    warnings: list[str] = []
    if event_count / max(covariate_parameter_count, 1) < 10:
        warnings.append("Piecewise exponential regression has fewer than 10 events per covariate parameter.")
    if np.isfinite(dispersion_ratio) and dispersion_ratio > 1.5:
        warnings.append("Piecewise exponential Poisson dispersion ratio is above 1.5; model fit should be reviewed.")

    return RegressionResult(
        model_id=model_id,
        model_type="piecewise_exponential",
        dependent_variable=duration_variable,
        independent_variables=independent_variables,
        sample_size=int(len(metadata["row_labels"])),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "deviance": float(fitted.deviance),
            "null_deviance": float(fitted.null_deviance),
            "pearson_chi_square": float(fitted.pearson_chi2),
            "dispersion_ratio": dispersion_ratio,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic_llf),
            "event_count": event_count,
            "censored_count": censored_count,
            "event_rate": float(event_count / len(metadata["row_labels"])),
            "interval_count": metadata["interval_count"],
            "long_row_count": metadata["long_row_count"],
            "total_exposure": float(long_data["_exposure"].sum()),
            "parameter_count": len(coefficients),
            "covariate_parameter_count": int(covariate_parameter_count),
            "events_per_parameter": float(event_count / max(covariate_parameter_count, 1)),
        },
        converged=bool(getattr(fitted, "converged", True)),
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            **metadata,
            "family": "poisson",
            "link": "log",
            "offset": "log_exposure",
            "maximum_iterations": maximum_iterations,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(metadata["fixed_effect_columns"]),
            "baseline_interval_hazards": interval_hazards,
            "long_data": long_data[["_row_index", "_interval_index", "_interval", "_interval_start", "_interval_stop", "_exposure", "_event"]].to_dict(orient="records"),
        },
        raw_result=fitted,
    )
