"""Heckman two-step sample-selection regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.statistics.regression.base import ModelCoefficient, RegressionResult


@dataclass(slots=True)
class HeckmanRawResult:
    outcome_result: Any
    selection_result: Any
    fittedvalues: pd.Series
    resid: pd.Series
    model: Any
    inverse_mills_ratio: pd.Series

    @property
    def params(self) -> Any:
        return self.outcome_result.params

    def conf_int(self) -> pd.DataFrame:
        return self.outcome_result.conf_int()


def _unique(values: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(value) for value in dict.fromkeys(values or []) if str(value).strip()]


def _ordered_categories(series: pd.Series) -> list[Any]:
    values = series.dropna().drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def _build_design_matrix(
    dataframe: pd.DataFrame,
    *,
    variables: list[str],
    fixed_effects: list[str],
    add_intercept: bool,
) -> pd.DataFrame:
    columns: list[pd.DataFrame] = []
    if variables:
        numeric = dataframe[variables].apply(pd.to_numeric, errors="coerce").astype(float)
        columns.append(numeric)
    for fixed_effect in fixed_effects:
        categories = _ordered_categories(dataframe[fixed_effect])
        if len(categories) <= 1:
            continue
        dummies = pd.get_dummies(
            pd.Categorical(dataframe[fixed_effect], categories=categories),
            prefix=fixed_effect,
            prefix_sep="_",
            drop_first=True,
            dtype=float,
        )
        dummies.index = dataframe.index
        columns.append(dummies)
    if columns:
        design = pd.concat(columns, axis=1)
    else:
        design = pd.DataFrame(index=dataframe.index)
    if add_intercept:
        design = sm.add_constant(design, has_constant="add")
    if design.empty:
        raise ValueError("Heckman selection requires at least one design column.")
    return design.astype(float)


def _selection_status(selection: pd.Series) -> pd.Series:
    values = pd.to_numeric(selection, errors="coerce")
    if values.dropna().nunique() != 2:
        raise ValueError("Heckman selection variable must be binary.")
    unique_values = sorted(values.dropna().unique().tolist())
    high = unique_values[-1]
    return (values == high).astype(int)


def fit_heckman_selection(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    selection_variable: str,
    selection_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "heckman_selection_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """Fit Heckman's two-step sample-selection model."""
    if covariance_type not in {"nonrobust", "HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("Heckman covariance_type must be nonrobust, HC0, HC1, HC2, or HC3.")
    independent_variables = _unique(independent_variables)
    selection_variables = _unique(selection_variables)
    fixed_effects = _unique(fixed_effects)
    if not independent_variables:
        raise ValueError("Heckman outcome equation requires independent_variables.")
    if not selection_variables:
        raise ValueError("Heckman selection equation requires selection_variables.")
    requested = [dependent_variable, selection_variable, *independent_variables, *selection_variables, *fixed_effects]
    missing = [variable for variable in dict.fromkeys(requested) if variable not in dataframe.columns]
    if missing:
        raise KeyError("Heckman variables are missing from dataframe: " + ", ".join(missing))

    base = dataframe[list(dict.fromkeys(requested))].copy()
    base["__selected__"] = _selection_status(base[selection_variable])
    for variable in [dependent_variable, *independent_variables, *selection_variables]:
        base[variable] = pd.to_numeric(base[variable], errors="coerce")

    selection_required = [selection_variable, *selection_variables, *fixed_effects]
    selection_data = base.dropna(subset=selection_required).copy()
    if selection_data.empty:
        raise ValueError("Heckman selection equation has no complete observations.")
    if selection_data["__selected__"].nunique() != 2:
        raise ValueError("Heckman selection equation needs both selected and non-selected cases.")
    outcome_required = [dependent_variable, *independent_variables, *fixed_effects]
    outcome_data = selection_data[selection_data["__selected__"] == 1].dropna(subset=outcome_required).copy()
    if outcome_data.empty:
        raise ValueError("Heckman outcome equation has no observed selected outcomes.")
    if len(outcome_data) <= len(independent_variables) + 2:
        raise ValueError("Heckman selected outcome sample is too small for estimation.")

    selection_x = _build_design_matrix(
        selection_data,
        variables=selection_variables,
        fixed_effects=fixed_effects,
        add_intercept=add_intercept,
    )
    selection_y = selection_data["__selected__"].astype(float)
    selection_model = sm.Probit(selection_y, selection_x)
    selection_result = selection_model.fit(maxiter=maximum_iterations, disp=False)

    selected_index = outcome_data.index
    selected_selection_x = selection_x.loc[selected_index]
    selection_linear = np.asarray(selected_selection_x @ selection_result.params, dtype=float)
    selection_probability = np.clip(stats.norm.cdf(selection_linear), 1e-12, 1.0)
    inverse_mills = pd.Series(
        stats.norm.pdf(selection_linear) / selection_probability,
        index=selected_index,
        name="inverse_mills_ratio",
    )

    outcome_x = _build_design_matrix(
        outcome_data,
        variables=independent_variables,
        fixed_effects=fixed_effects,
        add_intercept=add_intercept,
    )
    second_stage_x = outcome_x.copy()
    second_stage_x["inverse_mills_ratio"] = inverse_mills
    outcome_y = outcome_data[dependent_variable].astype(float)
    outcome_model = sm.OLS(outcome_y, second_stage_x)
    if covariance_type == "nonrobust":
        outcome_result = outcome_model.fit()
    else:
        outcome_result = outcome_model.fit(cov_type=covariance_type)

    confidence_intervals = outcome_result.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in outcome_result.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(outcome_result.params[term]),
                standard_error=float(outcome_result.bse[term]),
                statistic=float(outcome_result.tvalues[term]),
                p_value=float(outcome_result.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    rho_sigma = float(outcome_result.params.get("inverse_mills_ratio", np.nan))
    residual_sigma = float(np.sqrt(np.mean(np.asarray(outcome_result.resid, dtype=float) ** 2)))
    rho = rho_sigma / residual_sigma if residual_sigma > 0 and np.isfinite(rho_sigma) else np.nan
    rho = float(np.clip(rho, -0.999, 0.999)) if np.isfinite(rho) else np.nan
    warnings: list[str] = []
    imr_p = float(outcome_result.pvalues.get("inverse_mills_ratio", np.nan))
    if np.isfinite(imr_p) and imr_p < 0.05:
        warnings.append("The inverse Mills ratio is statistically significant; sample selection appears relevant.")
    excluded = [variable for variable in selection_variables if variable not in independent_variables]
    if not excluded:
        warnings.append("No exclusion restriction was supplied; Heckman identification relies on distributional assumptions.")

    raw_result = HeckmanRawResult(
        outcome_result=outcome_result,
        selection_result=selection_result,
        fittedvalues=pd.Series(outcome_result.fittedvalues, index=selected_index),
        resid=pd.Series(outcome_result.resid, index=selected_index),
        model=outcome_result.model,
        inverse_mills_ratio=inverse_mills,
    )

    return RegressionResult(
        model_id=model_id,
        model_type="heckman_selection",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome_data)),
        coefficients=coefficients,
        fit_statistics={
            "selected_sample_size": int(len(outcome_data)),
            "selection_sample_size": int(len(selection_data)),
            "selection_rate": float(selection_data["__selected__"].mean()),
            "outcome_r_squared": float(outcome_result.rsquared),
            "outcome_adjusted_r_squared": float(outcome_result.rsquared_adj),
            "inverse_mills_coefficient": rho_sigma,
            "inverse_mills_p_value": imr_p,
            "rho": rho,
            "sigma": residual_sigma,
            "selection_log_likelihood": float(selection_result.llf),
            "exclusion_restriction_count": len(excluded),
        },
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "selection_variable": selection_variable,
            "selection_variables": selection_variables,
            "outcome_variables": independent_variables,
            "exclusion_restrictions": excluded,
            "fixed_effects": fixed_effects,
            "selection_coefficients": {str(k): float(v) for k, v in selection_result.params.items()},
            "selection_p_values": {str(k): float(v) for k, v in selection_result.pvalues.items()},
            "inverse_mills_ratio": inverse_mills.tolist(),
            "selected_row_labels": [str(index) for index in selected_index],
            "selection_row_labels": [str(index) for index in selection_data.index],
            "selection_probabilities": selection_result.predict(selection_x).tolist(),
            "design_matrix_columns": [str(column) for column in second_stage_x.columns],
        },
        raw_result=raw_result,
    )
