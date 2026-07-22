"""Generalized Estimating Equation regression models."""

from __future__ import annotations

import warnings as python_warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix

GEE_MODEL_TYPES = {"gee_gaussian", "gee_logit", "gee_poisson", "gee_negative_binomial"}


def _family_for_model(model_type: str) -> Any:
    if model_type == "gee_gaussian":
        return sm.families.Gaussian()
    if model_type == "gee_logit":
        return sm.families.Binomial()
    if model_type == "gee_poisson":
        return sm.families.Poisson()
    if model_type == "gee_negative_binomial":
        return sm.families.NegativeBinomial()
    raise ValueError(f"Unsupported GEE model type: {model_type}")


def _covariance_structure(name: str) -> Any:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"independence", "independent"}:
        return sm.cov_struct.Independence()
    if normalized in {"exchangeable", "compound_symmetry"}:
        return sm.cov_struct.Exchangeable()
    if normalized in {"autoregressive", "ar1", "ar_1"}:
        return sm.cov_struct.Autoregressive()
    raise ValueError(
        "GEE covariance_structure must be independence, exchangeable, or autoregressive."
    )


def _validate_outcome(outcome: pd.Series, model_type: str) -> pd.Series:
    if model_type == "gee_logit":
        unique = sorted(outcome.dropna().unique().tolist())
        if unique != [0.0, 1.0]:
            raise ValueError(f"GEE logit requires a 0/1 outcome; found {unique}.")
    if model_type in {"gee_poisson", "gee_negative_binomial"}:
        if (outcome < 0).any():
            raise ValueError("GEE count models require a non-negative count outcome.")
        rounded = np.round(outcome)
        if not np.allclose(outcome, rounded):
            raise ValueError("GEE count models require integer count outcomes.")
        return pd.Series(rounded, index=outcome.index, name=outcome.name, dtype=float)
    return outcome


def _safe_qic(fitted: Any) -> tuple[float | None, float | None, list[dict[str, str]]]:
    try:
        with python_warnings.catch_warnings(record=True) as captured:
            python_warnings.simplefilter("always")
            qic, qicu = fitted.qic()
    except (AttributeError, ImportError, ValueError, TypeError):
        return None, None, []
    qic_warnings = [
        {"category": item.category.__name__, "message": str(item.message)} for item in captured
    ]
    return float(qic), float(qicu), qic_warnings


def _diagnostic_metadata(
    fitted: Any,
    outcome: pd.Series,
    predictors: pd.DataFrame,
    groups: pd.Series,
) -> dict[str, Any]:
    predicted = np.asarray(fitted.predict(), dtype=float)
    return {
        "diagnostics": {
            "endog": outcome.to_numpy(dtype=float),
            "predicted_mean": predicted,
            "exog": predictors.to_numpy(dtype=float),
            "exog_names": [str(column) for column in predictors.columns],
            "row_labels": list(outcome.index),
            "group_labels": groups.loc[outcome.index].astype(str).tolist(),
        }
    }


def fit_gee(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    fixed_effects: list[str] | None = None,
    model_id: str = "gee_1",
    model_type: str = "gee_gaussian",
    covariance_structure: str = "exchangeable",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit a population-averaged GEE model and return the common regression result."""
    if model_type not in GEE_MODEL_TYPES:
        raise ValueError(f"Unsupported GEE model type: {model_type}")
    if not group_variable.strip():
        raise ValueError("GEE requires a group_variable.")
    if group_variable not in dataframe.columns:
        raise KeyError(f"GEE group_variable is not in the dataframe: {group_variable}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    requested = [dependent_variable, *independent_variables, *fixed_effects, group_variable]
    complete = dataframe[requested].copy()
    for column in [dependent_variable, *independent_variables]:
        complete[column] = pd.to_numeric(complete[column], errors="coerce")
    complete = complete.dropna()
    if complete[group_variable].nunique() < 2:
        raise ValueError("GEE requires at least two clusters.")

    design = prepare_regression_design_matrix(
        complete,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="GEE",
    )
    outcome = _validate_outcome(design.outcome, model_type)
    predictors = design.predictors
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    groups = complete.loc[outcome.index, group_variable]
    cluster_sizes = groups.value_counts(sort=False)
    model = sm.GEE(
        outcome,
        predictors,
        groups=groups,
        family=_family_for_model(model_type),
        cov_struct=_covariance_structure(covariance_structure),
    )
    fitted = model.fit(maxiter=maximum_iterations)
    confidence_intervals = fitted.conf_int()

    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        estimate = float(fitted.params[term])
        lower = float(confidence_intervals.loc[term, 0])
        upper = float(confidence_intervals.loc[term, 1])
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=estimate,
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=(
                    float(np.exp(estimate)) if model_type in {"gee_logit", "gee_poisson", "gee_negative_binomial"} else None
                ),
            )
        )

    qic, qicu, qic_warnings = _safe_qic(fitted)
    converged = bool(getattr(fitted, "converged", True))
    warnings: list[str] = []
    if not converged:
        warnings.append("GEE model did not converge.")
    if cluster_sizes.min() < 2:
        warnings.append("At least one GEE cluster has fewer than two observations.")

    fit_statistics: dict[str, Any] = {
        "cluster_count": int(cluster_sizes.size),
        "minimum_cluster_size": int(cluster_sizes.min()),
        "maximum_cluster_size": int(cluster_sizes.max()),
        "mean_cluster_size": float(cluster_sizes.mean()),
        "scale": float(fitted.scale),
        "qic": qic,
        "qicu": qicu,
        "alpha": 1.0 if model_type == "gee_negative_binomial" else None,
        "negative_binomial_alpha": 1.0 if model_type == "gee_negative_binomial" else None,
    }

    return RegressionResult(
        model_id=model_id,
        model_type=model_type,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=converged,
        standard_error_type="gee_robust_sandwich",
        warnings=warnings,
        metadata={
            "group_variable": group_variable,
            "covariance_structure": covariance_structure,
            "family": model.family.__class__.__name__,
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "qic_warnings": qic_warnings,
            "qic_warning_count": len(qic_warnings),
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
            **_diagnostic_metadata(fitted, outcome, predictors, groups),
        },
        raw_result=fitted,
    )
