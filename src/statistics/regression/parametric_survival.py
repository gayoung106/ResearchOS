"""Automatic selection among parametric AFT survival models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult
from src.statistics.regression.exponential_aft import fit_exponential_aft
from src.statistics.regression.loglogistic_aft import fit_loglogistic_aft
from src.statistics.regression.lognormal_aft import fit_lognormal_aft
from src.statistics.regression.weibull_aft import fit_weibull_aft

_AFT_FITTERS: dict[str, Callable[..., RegressionResult]] = {
    "exponential_aft": fit_exponential_aft,
    "weibull_aft": fit_weibull_aft,
    "lognormal_aft": fit_lognormal_aft,
    "loglogistic_aft": fit_loglogistic_aft,
}


def _candidate_record(result: RegressionResult, *, selected: bool = False) -> dict[str, Any]:
    return {
        "model_type": result.model_type,
        "status": "selected" if selected else "fit",
        "converged": result.converged,
        "aic": result.fit_statistics.get("aic"),
        "bic": result.fit_statistics.get("bic"),
        "log_likelihood": result.fit_statistics.get("log_likelihood"),
        "event_count": result.fit_statistics.get("event_count"),
        "censored_count": result.fit_statistics.get("censored_count"),
        "warning_count": len(result.warnings),
    }


def _failed_candidate_record(model_type: str, error: Exception) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "status": "failed",
        "converged": False,
        "aic": np.nan,
        "bic": np.nan,
        "log_likelihood": np.nan,
        "event_count": np.nan,
        "censored_count": np.nan,
        "warning_count": 1,
        "error": str(error),
    }


def fit_parametric_survival_regression(
    dataframe: pd.DataFrame,
    *,
    duration_variable: str,
    event_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "parametric_survival_1",
    candidate_models: list[str] | tuple[str, ...] | None = None,
    selection_criterion: str = "aic",
    add_intercept: bool = True,
    maximum_iterations: int = 500,
) -> RegressionResult:
    """Fit candidate AFT survival models and return the best model by AIC or BIC."""
    criterion = selection_criterion.strip().lower()
    if criterion not in {"aic", "bic"}:
        raise ValueError("Parametric survival selection criterion must be 'aic' or 'bic'.")

    requested = list(candidate_models or _AFT_FITTERS.keys())
    requested = [str(model_type).strip().lower() for model_type in requested if str(model_type).strip()]
    if not requested:
        raise ValueError("At least one parametric survival candidate model is required.")

    unknown = [model_type for model_type in requested if model_type not in _AFT_FITTERS]
    if unknown:
        raise ValueError("Unsupported parametric survival candidate model(s): " + ", ".join(unknown))

    fitted_results: list[RegressionResult] = []
    candidate_records: list[dict[str, Any]] = []
    for model_type in dict.fromkeys(requested):
        fitter = _AFT_FITTERS[model_type]
        try:
            result = fitter(
                dataframe,
                duration_variable=duration_variable,
                event_variable=event_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                model_id=model_id,
                add_intercept=add_intercept,
                maximum_iterations=maximum_iterations,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            candidate_records.append(_failed_candidate_record(model_type, error))
            continue

        candidate_records.append(_candidate_record(result))
        if result.converged and np.isfinite(float(result.fit_statistics.get(criterion, np.nan))):
            fitted_results.append(result)

    if not fitted_results:
        failures = [str(record.get("error", record["model_type"])) for record in candidate_records]
        raise ValueError("No parametric survival candidate model could be selected: " + "; ".join(failures))

    selected = min(fitted_results, key=lambda result: float(result.fit_statistics[criterion]))
    selected_records = []
    for record in candidate_records:
        is_selected = record["model_type"] == selected.model_type and record["status"] == "fit"
        selected_records.append({**record, "status": "selected" if is_selected else record["status"]})

    selected.metadata.update(
        {
            "selected_survival_model": selected.model_type,
            "survival_selection_criterion": criterion,
            "candidate_survival_models": selected_records,
            "candidate_survival_model_count": len(selected_records),
        }
    )
    selected.fit_statistics["selected_survival_model_aic"] = selected.fit_statistics.get("aic")
    selected.fit_statistics["selected_survival_model_bic"] = selected.fit_statistics.get("bic")
    return selected
