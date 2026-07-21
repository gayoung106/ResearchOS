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
    entry_variable: str | None = None,
    cluster_variable: str | None = None,
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
    if entry_variable is not None:
        if entry_variable not in dataframe.columns:
            raise KeyError("Entry variable is missing from dataframe: " + entry_variable)
        if entry_variable == duration_variable or entry_variable == event_variable or entry_variable in independent_variables:
            raise ValueError("Entry variable cannot duplicate the duration, event, or predictor variables.")
        if entry_variable == strata_variable:
            raise ValueError("Entry variable cannot duplicate the strata variable.")
    if cluster_variable is not None:
        if cluster_variable not in dataframe.columns:
            raise KeyError("Cluster variable is missing from dataframe: " + cluster_variable)
        if cluster_variable == duration_variable or cluster_variable == event_variable or cluster_variable in independent_variables:
            raise ValueError("Cluster variable cannot duplicate the duration, event, or predictor variables.")
        if cluster_variable in {strata_variable, entry_variable}:
            raise ValueError("Cluster variable cannot duplicate the strata or entry variable.")
    _validate_fixed_effects(
        dataframe,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    selected_columns = [duration_variable, event_variable, *independent_variables, *fixed_effects]
    if strata_variable is not None:
        selected_columns.append(strata_variable)
    if entry_variable is not None:
        selected_columns.append(entry_variable)
    if cluster_variable is not None:
        selected_columns.append(cluster_variable)
    selected = dataframe[selected_columns].copy()
    selected[duration_variable] = pd.to_numeric(selected[duration_variable], errors="coerce")
    selected[event_variable] = pd.to_numeric(selected[event_variable], errors="coerce")
    if entry_variable is not None:
        selected[entry_variable] = pd.to_numeric(selected[entry_variable], errors="coerce")
    for variable in independent_variables:
        selected[variable] = pd.to_numeric(selected[variable], errors="coerce")
    complete = selected.dropna()
    if complete.empty:
        raise ValueError("Cox regression has no complete observations to estimate.")
    if (complete[duration_variable] <= 0).any():
        raise ValueError("Cox regression duration values must be positive.")
    if entry_variable is not None:
        if (complete[entry_variable] < 0).any():
            raise ValueError("Cox regression entry values must be non-negative.")
        if (complete[entry_variable] >= complete[duration_variable]).any():
            raise ValueError("Cox regression entry values must be less than duration values.")

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
    if entry_variable is not None:
        entry_values = complete[entry_variable].astype(float)
        metadata.update(
            {
                "entry_variable": entry_variable,
                "entry_values": entry_values.to_numpy(),
                "left_truncated_count": int((entry_values > 0).sum()),
                "minimum_entry_time": float(entry_values.min()),
                "maximum_entry_time": float(entry_values.max()),
            }
        )
    if cluster_variable is not None:
        cluster_series = complete[cluster_variable].astype(str)
        cluster_counts = cluster_series.value_counts().sort_index()
        cluster_events = complete.groupby(cluster_series, sort=True)[event_variable].sum()
        metadata.update(
            {
                "cluster_variable": cluster_variable,
                "cluster_values": cluster_series.to_numpy(),
                "cluster_labels": [str(index) for index in cluster_counts.index],
                "cluster_counts": {str(index): int(value) for index, value in cluster_counts.items()},
                "cluster_event_counts": {str(index): int(value) for index, value in cluster_events.items()},
                "cluster_count": int(cluster_counts.shape[0]),
            }
        )
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
    result_metadata.pop("entry_values", None)
    result_metadata.pop("cluster_values", None)
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
    if "left_truncated_count" in metadata:
        fit_statistics["left_truncated_count"] = metadata["left_truncated_count"]
        fit_statistics["minimum_entry_time"] = metadata["minimum_entry_time"]
        fit_statistics["maximum_entry_time"] = metadata["maximum_entry_time"]
    if "cluster_count" in metadata:
        fit_statistics["cluster_count"] = metadata["cluster_count"]

    standard_error_type = "cluster_robust_partial_likelihood" if "cluster_count" in metadata else "partial_likelihood"

    return RegressionResult(
        model_id=model_id,
        model_type=model_type,
        dependent_variable=duration_variable,
        independent_variables=independent_variables,
        sample_size=int(len(duration)),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=True,
        standard_error_type=standard_error_type,
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


def fit_left_truncated_cox(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    entry_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "left_truncated_cox_1",
    ties: str = "breslow",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a Cox proportional hazards model with delayed entry/left truncation."""
    if ties not in {"breslow", "efron"}:
        raise ValueError("Cox regression ties must be 'breslow' or 'efron'.")
    entry_variable = str(entry_variable).strip()
    if not entry_variable:
        raise ValueError("Left-truncated Cox regression requires entry_variable.")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    duration, event, predictors, metadata = _prepare_cox_design(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        entry_variable=entry_variable,
    )
    fitted = PHReg(
        duration,
        predictors,
        status=event,
        entry=metadata["entry_values"],
        ties=ties,
    ).fit(maxiter=maximum_iterations)
    return _finalize_cox_result(
        fitted=fitted,
        model_id=model_id,
        model_type="left_truncated_cox",
        duration_variable=duration_variable,
        independent_variables=independent_variables,
        duration=duration,
        event=event,
        metadata=metadata,
        ties=ties,
        maximum_iterations=maximum_iterations,
    )


def _as_comparable_labels(series: pd.Series) -> pd.Series:
    return series.astype(str)


def fit_cause_specific_cox(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    cause_variable: str,
    target_event_code: str | int | float,
    independent_variables: list[str],
    censor_codes: list[str | int | float] | None = None,
    fixed_effects: list[str] | None = None,
    model_id: str = "cause_specific_cox_1",
    ties: str = "breslow",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a cause-specific Cox model for competing-risks event data."""
    if ties not in {"breslow", "efron"}:
        raise ValueError("Cox regression ties must be 'breslow' or 'efron'.")
    cause_variable = str(cause_variable).strip()
    if not cause_variable:
        raise ValueError("Cause-specific Cox regression requires cause_variable.")
    if cause_variable not in dataframe.columns:
        raise KeyError("Cause variable is missing from dataframe: " + cause_variable)
    if cause_variable == duration_variable or cause_variable in independent_variables:
        raise ValueError("Cause variable cannot duplicate the duration or predictor variables.")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    censor_codes = [0] if censor_codes is None else list(censor_codes)
    target_label = str(target_event_code)
    censor_labels = {str(code) for code in censor_codes}

    event_indicator_variable = "__cause_specific_event__"
    while event_indicator_variable in dataframe.columns:
        event_indicator_variable = "_" + event_indicator_variable

    working = dataframe.copy()
    cause_labels = _as_comparable_labels(working[cause_variable])
    working[event_indicator_variable] = np.where(
        cause_labels == target_label,
        1.0,
        0.0,
    )
    working.loc[working[cause_variable].isna(), event_indicator_variable] = np.nan

    duration, event, predictors, metadata = _prepare_cox_design(
        working,
        duration_variable=duration_variable,
        event_variable=event_indicator_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )
    cause_complete = _as_comparable_labels(working.loc[duration.index, cause_variable])
    competing_mask = (event == 0) & ~cause_complete.isin(censor_labels)
    censor_mask = (event == 0) & cause_complete.isin(censor_labels)
    metadata.update(
        {
            "event_variable": cause_variable,
            "event_indicator_variable": event_indicator_variable,
            "cause_variable": cause_variable,
            "target_event_code": target_event_code,
            "censor_codes": censor_codes,
            "competing_event_count": int(competing_mask.sum()),
            "cause_specific_event_count": int(event.sum()),
            "original_censored_count": int(censor_mask.sum()),
        }
    )

    fitted = PHReg(duration, predictors, status=event, ties=ties).fit(maxiter=maximum_iterations)
    result = _finalize_cox_result(
        fitted=fitted,
        model_id=model_id,
        model_type="cause_specific_cox",
        duration_variable=duration_variable,
        independent_variables=independent_variables,
        duration=duration,
        event=event,
        metadata=metadata,
        ties=ties,
        maximum_iterations=maximum_iterations,
    )
    result.fit_statistics.update(
        {
            "cause_specific_event_count": int(event.sum()),
            "competing_event_count": int(competing_mask.sum()),
            "original_censored_count": int(censor_mask.sum()),
            "target_event_rate": float(event.sum() / len(event)),
        }
    )
    if int(competing_mask.sum()) == 0:
        result.warnings.append("No competing events were observed for the cause-specific Cox model.")
    return result


def fit_clustered_cox(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    cluster_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "clustered_cox_1",
    ties: str = "breslow",
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a Cox model with cluster-robust standard errors."""
    if ties not in {"breslow", "efron"}:
        raise ValueError("Cox regression ties must be 'breslow' or 'efron'.")
    cluster_variable = str(cluster_variable).strip()
    if not cluster_variable:
        raise ValueError("Clustered Cox regression requires cluster_variable.")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    duration, event, predictors, metadata = _prepare_cox_design(
        dataframe,
        duration_variable=duration_variable,
        event_variable=event_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        cluster_variable=cluster_variable,
    )
    if int(metadata["cluster_count"]) < 2:
        raise ValueError("Clustered Cox regression requires at least two clusters.")
    fitted = PHReg(duration, predictors, status=event, ties=ties).fit(
        groups=metadata["cluster_values"],
        maxiter=maximum_iterations,
    )
    return _finalize_cox_result(
        fitted=fitted,
        model_id=model_id,
        model_type="clustered_cox",
        duration_variable=duration_variable,
        independent_variables=independent_variables,
        duration=duration,
        event=event,
        metadata=metadata,
        ties=ties,
        maximum_iterations=maximum_iterations,
    )
