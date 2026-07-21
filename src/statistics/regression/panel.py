"""Panel fixed-effects regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)


def _ordered_categories(series: pd.Series) -> list[Any]:
    values = series.dropna().drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def _residualize_against_fixed_effects(values: np.ndarray, fixed_effect_matrix: np.ndarray) -> np.ndarray:
    if fixed_effect_matrix.size == 0:
        return values.astype(float)
    fitted = fixed_effect_matrix @ np.linalg.lstsq(fixed_effect_matrix, values, rcond=None)[0]
    return values - fitted


def _fixed_effect_matrix(work: pd.DataFrame, entity_variable: str, time_variable: str | None) -> np.ndarray:
    parts = [np.ones((len(work), 1), dtype=float)]
    for variable in [entity_variable, time_variable]:
        if variable is None:
            continue
        categories = _ordered_categories(work[variable])
        if len(categories) <= 1:
            continue
        dummies = pd.get_dummies(
            pd.Categorical(work[variable], categories=categories),
            drop_first=True,
            dtype=float,
        )
        parts.append(dummies.to_numpy(dtype=float))
    return np.column_stack(parts)


def fit_panel_fixed_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str | None = None,
    model_id: str = "panel_fixed_effects_1",
    covariance_type: str = "cluster_entity",
) -> RegressionResult:
    """Fit a within-transformed fixed-effects panel regression."""
    if covariance_type not in {"nonrobust", "HC3", "cluster_entity"}:
        raise ValueError("Panel fixed effects covariance_type must be nonrobust, HC3, or cluster_entity.")
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable is not None:
        if time_variable not in dataframe.columns:
            raise KeyError("Panel time variable is missing from dataframe: " + time_variable)
        if time_variable == dependent_variable or time_variable in independent_variables:
            raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable]
    if time_variable is not None:
        requested.append(time_variable)
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ValueError("Panel fixed effects has no complete observations to estimate.")
    if work[dependent_variable].nunique() <= 1:
        raise ValueError("Panel dependent variable has no variation.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel fixed effects requires at least two entities.")
    constant_predictors = [variable for variable in independent_variables if work[variable].nunique() <= 1]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    fe_matrix = _fixed_effect_matrix(work, entity_variable, time_variable)
    y = work[dependent_variable].to_numpy(dtype=float)
    x = work[independent_variables].to_numpy(dtype=float)
    y_within = _residualize_against_fixed_effects(y, fe_matrix)
    x_within = np.column_stack(
        [_residualize_against_fixed_effects(x[:, index], fe_matrix) for index in range(x.shape[1])]
    )
    kept = np.asarray([not np.isclose(np.var(x_within[:, index]), 0.0) for index in range(x_within.shape[1])])
    if not kept.all():
        dropped = [name for name, keep in zip(independent_variables, kept, strict=True) if not keep]
        raise ValueError("Predictors with no within-panel variation cannot be estimated: " + ", ".join(dropped))

    model = sm.OLS(y_within, pd.DataFrame(x_within, columns=independent_variables, index=work.index))
    if covariance_type == "cluster_entity":
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": work[entity_variable].to_numpy()})
        standard_error_type = "cluster_entity"
    elif covariance_type == "HC3":
        fitted = model.fit(cov_type="HC3")
        standard_error_type = "HC3"
    else:
        fitted = model.fit()
        standard_error_type = "nonrobust"

    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.params[term]),
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    time_count = int(work[time_variable].nunique()) if time_variable is not None else None
    singleton_count = int((entity_counts == 1).sum())
    warnings: list[str] = []
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")
    if len(work) <= len(independent_variables) + len(entity_counts):
        warnings.append("The panel sample size is small relative to absorbed effects and predictors.")

    residuals = np.asarray(fitted.resid, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    return RegressionResult(
        model_id=model_id,
        model_type="panel_fixed_effects",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(work)),
        coefficients=coefficients,
        fit_statistics={
            "within_r_squared": float(fitted.rsquared),
            "adjusted_within_r_squared": float(fitted.rsquared_adj),
            "entity_count": int(len(entity_counts)),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_observations_per_entity": float(entity_counts.mean()),
            "residual_degrees_of_freedom": float(fitted.df_resid),
        },
        converged=True,
        standard_error_type=standard_error_type,
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "absorbed_effects": [entity_variable, *([time_variable] if time_variable else [])],
            "row_labels": [str(index) for index in work.index],
            "entity_labels": work[entity_variable].astype(str).tolist(),
            "time_labels": work[time_variable].astype(str).tolist() if time_variable is not None else None,
            "within_outcome": y_within.tolist(),
            "within_predictors": x_within.tolist(),
            "within_predictor_names": independent_variables,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "dropped_case_count": len(dataframe) - len(work),
        },
        raw_result=fitted,
    )
