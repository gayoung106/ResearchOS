"""Cox proportional hazards regression implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)
from src.statistics.regression.design_matrix import _encode_fixed_effects, _validate_fixed_effects


def _ordered_categories(series: pd.Series) -> list[Any]:
    categories = series.dropna().drop_duplicates().tolist()
    try:
        return sorted(categories)
    except TypeError:
        return sorted(categories, key=lambda value: str(value))


def _prepare_cox_design(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str],
    strata_variable: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, Any]]:
    validate_model_variables(dataframe, duration_variable, independent_variables)
    if event_variable not in dataframe.columns:
        raise KeyError("Event variable is missing from dataframe: " + event_variable)
    if event_variable == duration_variable or event_variable in independent_variables:
        raise ValueError("Event variable cannot duplicate the duration or predictor variables.")
    if strata_variable is not None:
        if strata_variable not in dataframe.columns:
            raise KeyError("Strata variable is missing from dataframe: " + strata_variable)
        if strata_variable == duration_variable or strata_variable == event_variable or strata_variable in independent_variables:
            raise ValueError("Strata variable cannot duplicate the duration, event, or predictor variables.")
    _validate_fixed_effects(
        dataframe,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    selected_columns = [duration_variable, event_variable, *independent_variables, *fixed_effects]
    if strata_variable is not None:
        selected_columns.append(strata_variable)
    selected = dataframe[selected_columns].copy()
    selected[duration_variable] = pd.to_numeric(selected[duration_variable], errors="coerce")
    selected[event_variable] = pd.to_numeric(selected[event_variable], errors="coerce")
    for variable in independent_variables:
        selected[variable] = pd.to_numeric(selected[variable], errors="coerce")
    complete = selected.dropna()
    if complete.empty:
        raise ValueError("Cox regression has no complete observations to estimate.")
    if (complete[duration_variable] <= 0).any():
        raise ValueError("Cox regression duration values must be positive.")

    event_values = sorted(complete[event_variable].unique().tolist())
    if event_values != [0.0, 1.0]:
        raise ValueError(f"Cox regression event variable must be coded 0/1. Current values: {event_values}")
    if int(complete[event_variable].sum()) == 0:
        raise ValueError("Cox regression requires at least one observed event.")

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
        raise ValueError("Cox regression requires at least one predictor.")

    metadata: dict[str, Any] = {
        "event_variable": event_variable,
        "fixed_effects": fixed_effects,
        "fixed_effect_reference_categories": reference_categories,
        "fixed_effect_columns": fixed_effect_columns,
        "dropped_case_count": len(dataframe) - len(complete),
        "row_labels": [str(index) for index in complete.index],
    }
    if strata_variable is not None:
        strata_series = complete[strata_variable].astype(str)
        strata_labels = _ordered_categories(strata_series)
        strata_codes = pd.Categorical(strata_series, categories=strata_labels, ordered=False).codes
        strata_counts = strata_series.value_counts().reindex(strata_labels, fill_value=0)
        strata_events = complete.groupby(strata_series, sort=False)[event_variable].sum().reindex(strata_labels, fill_value=0)
        if (strata_events <= 0).any():
            empty_strata = ", ".join(str(index) for index in strata_events[strata_events <= 0].index)
            raise ValueError("Cox strata must each contain at least one observed event: " + empty_strata)
        metadata.update(
            {
                "strata_variable": strata_variable,
                "strata_values": strata_codes.astype(int),
                "strata_labels": [str(index) for index in strata_labels],
                "strata_counts": {str(index): int(value) for index, value in strata_counts.items()},
                "strata_event_counts": {str(index): int(value) for index, value in strata_events.items()},
                "strata_count": int(len(strata_labels)),
            }
        )

    return (
        complete[duration_variable].astype(float),
        complete[event_variable].astype(int),
        predictors,
        metadata,
    )


def _baseline_survival_frame(fitted: Any) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for stratum_index, stratum in enumerate(fitted.baseline_cumulative_hazard):
        times = np.asarray(stratum[0], dtype=float)
        cumulative_hazard = np.asarray(stratum[1], dtype=float)
        survival = np.asarray(stratum[2], dtype=float)
        for time, hazard, survival_value in zip(times, cumulative_hazard, survival, strict=False):
            rows.append(
                {
                    "stratum": float(stratum_index),
                    "time": float(time),
                    "baseline_cumulative_hazard": float(hazard),
                    "baseline_survival": float(survival_value),
                }
            )
    return pd.DataFrame(rows)



def _cox_coefficients(fitted: Any) -> list[ModelCoefficient]:
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    names = [str(name) for name in fitted.model.exog_names]
    for index, term in enumerate(names):
        estimate = float(fitted.params[index])
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=estimate,
                standard_error=float(fitted.bse[index]),
                statistic=float(fitted.tvalues[index]),
                p_value=float(fitted.pvalues[index]),
                confidence_interval_lower=float(confidence_intervals[index, 0]),
                confidence_interval_upper=float(confidence_intervals[index, 1]),
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )
    return coefficients


def _cox_fit_warnings(
    *,
    event_count: int,
    censored_count: int,
    parameter_count: int,
    metadata: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if event_count / max(parameter_count, 1) < 10:
        warnings.append("Cox regression has fewer than 10 events per estimated parameter.")
    if censored_count == 0:
        warnings.append("No censored observations were observed; censoring assumptions should be reviewed.")
    strata_events = metadata.get("strata_event_counts") or {}
    low_event_strata = [str(name) for name, count in strata_events.items() if int(count) < 5]
    if low_event_strata:
        warnings.append("Some Cox strata have fewer than 5 observed events: " + ", ".join(low_event_strata))
    return warnings


def _finalize_cox_result(
    *,
    fitted: Any,
    model_id: str,
    model_type: str,
    duration_variable: str,
    independent_variables: list[str],
    duration: pd.Series,
    event: pd.Series,
    metadata: dict[str, Any],
    ties: str,
    maximum_iterations: int,
) -> RegressionResult:
    coefficients = _cox_coefficients(fitted)
    names = [str(name) for name in fitted.model.exog_names]
    event_count = int(event.sum())
    censored_count = int(len(event) - event_count)
    parameter_count = len(coefficients)
    baseline = _baseline_survival_frame(fitted)
    result_metadata = dict(metadata)
    result_metadata.pop("strata_values", None)
    result_metadata.update(
        {
            "duration_variable": duration_variable,
            "ties": ties,
            "maximum_iterations": maximum_iterations,
            "design_matrix_columns": names,
            "fixed_effect_column_count": len(metadata["fixed_effect_columns"]),
            "baseline_survival": baseline.to_dict(orient="records"),
        }
    )
    fit_statistics = {
        "log_likelihood": float(fitted.llf),
        "event_count": event_count,
        "censored_count": censored_count,
        "event_rate": float(event_count / len(event)),
        "parameter_count": parameter_count,
        "events_per_parameter": float(event_count / max(parameter_count, 1)),
    }
    if "strata_count" in metadata:
        fit_statistics["strata_count"] = metadata["strata_count"]

    return RegressionResult(
        model_id=model_id,
        model_type=model_type,
        dependent_variable=duration_variable,
        independent_variables=independent_variables,
        sample_size=int(len(duration)),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=True,
        standard_error_type="partial_likelihood",
        warnings=_cox_fit_warnings(
            event_count=event_count,
            censored_count=censored_count,
            parameter_count=parameter_count,
            metadata=metadata,
        ),
        metadata=result_metadata,
        raw_result=fitted,
    )


def fit_cox_proportional_hazards(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "cox_ph_1",
    ties: str = "breslow",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a Cox proportional hazards model."""
    if ties not in {"breslow", "efron"}:
        raise ValueError("Cox regression ties must be 'breslow' or 'efron'.")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    duration, event, predictors, metadata = _prepare_cox_design(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    fitted = PHReg(duration, predictors, status=event, ties=ties).fit(maxiter=maximum_iterations)
    return _finalize_cox_result(
        fitted=fitted,
        model_id=model_id,
        model_type="cox_proportional_hazards",
        duration_variable=duration_variable,
        independent_variables=independent_variables,
        duration=duration,
        event=event,
        metadata=metadata,
        ties=ties,
        maximum_iterations=maximum_iterations,
    )


def fit_stratified_cox(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    strata_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "stratified_cox_1",
    ties: str = "breslow",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a Cox proportional hazards model with strata-specific baseline hazards."""
    if ties not in {"breslow", "efron"}:
        raise ValueError("Cox regression ties must be 'breslow' or 'efron'.")
    strata_variable = str(strata_variable).strip()
    if not strata_variable:
        raise ValueError("Stratified Cox regression requires strata_variable.")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    duration, event, predictors, metadata = _prepare_cox_design(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        strata_variable=strata_variable,
    )
    strata_values = metadata["strata_values"]
    fitted = PHReg(duration, predictors, status=event, ties=ties, strata=strata_values).fit(
        maxiter=maximum_iterations
    )
    return _finalize_cox_result(
        fitted=fitted,
        model_id=model_id,
        model_type="stratified_cox",
        duration_variable=duration_variable,
        independent_variables=independent_variables,
        duration=duration,
        event=event,
        metadata=metadata,
        ties=ties,
        maximum_iterations=maximum_iterations,
    )
