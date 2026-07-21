"""Two-stage least squares instrumental-variable regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


@dataclass(slots=True)
class IV2SLSRawResult:
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    tvalues: pd.Series
    fittedvalues: pd.Series
    resid: pd.Series
    model: Any
    covariance: np.ndarray

    def conf_int(self) -> pd.DataFrame:
        lower = self.params - 1.96 * self.bse
        upper = self.params + 1.96 * self.bse
        return pd.DataFrame({0: lower, 1: upper})


def _as_unique_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(value) for value in dict.fromkeys(values or [])]


def _projection(matrix: np.ndarray) -> np.ndarray:
    return matrix @ np.linalg.pinv(matrix.T @ matrix) @ matrix.T


def _first_stage_f_statistic(
    y: np.ndarray,
    exogenous: pd.DataFrame,
    instruments: pd.DataFrame,
) -> tuple[float | None, float | None, float, float]:
    full_x = pd.concat([exogenous, instruments], axis=1)
    restricted_x = exogenous
    full = sm.OLS(y, full_x).fit()
    restricted = sm.OLS(y, restricted_x).fit()
    q = instruments.shape[1]
    df_denom = full.df_resid
    if q <= 0 or df_denom <= 0:
        return None, None, float(full.rsquared), float(restricted.rsquared)
    numerator = (restricted.ssr - full.ssr) / q
    denominator = full.ssr / df_denom
    if denominator <= 0:
        return None, None, float(full.rsquared), float(restricted.rsquared)
    statistic = float(numerator / denominator)
    p_value = float(stats.f.sf(statistic, q, df_denom))
    return statistic, p_value, float(full.rsquared), float(restricted.rsquared)


def fit_iv_2sls_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    endogenous_variables: list[str],
    instrument_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "iv_2sls_regression_1",
    add_intercept: bool = True,
) -> RegressionResult:
    """Fit a two-stage least-squares instrumental-variable model."""
    independent_variables = _as_unique_list(independent_variables)
    endogenous_variables = _as_unique_list(endogenous_variables)
    instrument_variables = _as_unique_list(instrument_variables)
    fixed_effects = _as_unique_list(fixed_effects)
    if not endogenous_variables:
        raise ValueError("IV 2SLS requires at least one endogenous variable.")
    if not instrument_variables:
        raise ValueError("IV 2SLS requires at least one instrument variable.")
    overlap = set(endogenous_variables) & set(instrument_variables)
    if overlap:
        raise ValueError("Endogenous variables cannot also be instruments: " + ", ".join(sorted(overlap)))
    exogenous_variables = [variable for variable in independent_variables if variable not in endogenous_variables]
    validate_model_variables(dataframe, dependent_variable, exogenous_variables + endogenous_variables)
    missing_instruments = [variable for variable in instrument_variables if variable not in dataframe.columns]
    if missing_instruments:
        raise KeyError("Instrument variables are missing from dataframe: " + ", ".join(missing_instruments))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=exogenous_variables + endogenous_variables + instrument_variables,
        fixed_effects=fixed_effects,
        model_label="IV 2SLS",
    )
    outcome = design.outcome.astype(float)
    matrix = design.predictors.astype(float)
    exogenous = matrix[[column for column in matrix.columns if column in exogenous_variables or column in design.fixed_effect_columns]].copy()
    endogenous = matrix[endogenous_variables].copy()
    instruments = matrix[instrument_variables].copy()
    if exogenous.empty:
        exogenous = pd.DataFrame(index=matrix.index)
    if add_intercept:
        exogenous = sm.add_constant(exogenous, has_constant="add")

    if instruments.shape[1] < endogenous.shape[1]:
        raise ValueError("IV 2SLS requires at least as many instruments as endogenous variables.")
    z_matrix = pd.concat([exogenous, instruments], axis=1)
    if np.linalg.matrix_rank(z_matrix.to_numpy(dtype=float)) < z_matrix.shape[1]:
        raise ValueError("Instrument matrix is rank deficient.")

    x_actual = pd.concat([exogenous, endogenous], axis=1)
    z = z_matrix.to_numpy(dtype=float)
    x = x_actual.to_numpy(dtype=float)
    y = outcome.to_numpy(dtype=float)
    pz = _projection(z)
    xpzx = x.T @ pz @ x
    if np.linalg.matrix_rank(xpzx) < xpzx.shape[0]:
        raise ValueError("Projected IV design matrix is rank deficient.")
    beta = np.linalg.pinv(xpzx) @ (x.T @ pz @ y)
    fitted_values = x @ beta
    residuals = y - fitted_values
    df_resid = len(y) - len(beta)
    sigma2 = float(residuals.T @ residuals / df_resid) if df_resid > 0 else np.nan
    covariance = sigma2 * np.linalg.pinv(xpzx)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    statistics = np.divide(beta, standard_errors, out=np.full_like(beta, np.nan), where=standard_errors > 0)
    p_values = 2.0 * stats.t.sf(np.abs(statistics), df_resid) if df_resid > 0 else np.full_like(beta, np.nan)
    terms = [str(column) for column in x_actual.columns]
    params = pd.Series(beta, index=terms)
    bse = pd.Series(standard_errors, index=terms)
    tvalues = pd.Series(statistics, index=terms)
    pvalue_series = pd.Series(p_values, index=terms)
    raw_result = IV2SLSRawResult(
        params=params,
        bse=bse,
        pvalues=pvalue_series,
        tvalues=tvalues,
        fittedvalues=pd.Series(fitted_values, index=outcome.index),
        resid=pd.Series(residuals, index=outcome.index),
        model=type(
            "IV2SLSModelData",
            (),
            {
                "endog": y,
                "exog": x,
                "exog_names": terms,
                "data": type("IV2SLSModelRows", (), {"row_labels": outcome.index.tolist()})(),
            },
        )(),
        covariance=covariance,
    )
    confidence_intervals = raw_result.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in terms:
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=float(params[term]),
                standard_error=float(bse[term]),
                statistic=float(tvalues[term]),
                p_value=float(pvalue_series[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    first_stage: dict[str, dict[str, float | None]] = {}
    fitted_endogenous: dict[str, list[float]] = {}
    for variable in endogenous_variables:
        first_stage_model = sm.OLS(endogenous[variable], z_matrix).fit()
        fitted_endogenous[variable] = np.asarray(first_stage_model.fittedvalues, dtype=float).tolist()
        f_stat, f_p, r2_full, r2_restricted = _first_stage_f_statistic(
            endogenous[variable].to_numpy(dtype=float),
            exogenous,
            instruments,
        )
        first_stage[variable] = {
            "excluded_instrument_f_statistic": f_stat,
            "excluded_instrument_p_value": f_p,
            "r_squared": r2_full,
            "restricted_r_squared": r2_restricted,
        }

    ss_resid = float(np.sum(residuals**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan
    min_first_stage_f = min(
        [value["excluded_instrument_f_statistic"] for value in first_stage.values() if value["excluded_instrument_f_statistic"] is not None],
        default=None,
    )
    warnings: list[str] = []
    if min_first_stage_f is not None and float(min_first_stage_f) < 10.0:
        warnings.append("Weak instruments are possible; minimum first-stage F statistic is below 10.")

    return RegressionResult(
        model_id=model_id,
        model_type="iv_2sls_regression",
        dependent_variable=dependent_variable,
        independent_variables=exogenous_variables + endogenous_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics={
            "r_squared": float(r_squared),
            "root_mean_squared_error": float(np.sqrt(np.mean(residuals**2))),
            "mean_absolute_error": float(np.mean(np.abs(residuals))),
            "residual_degrees_of_freedom": float(df_resid),
            "endogenous_variable_count": len(endogenous_variables),
            "instrument_count": len(instrument_variables),
            "overidentified": len(instrument_variables) > len(endogenous_variables),
            "minimum_first_stage_f_statistic": min_first_stage_f,
        },
        converged=True,
        standard_error_type="2sls_homoskedastic",
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "exogenous_variables": exogenous_variables,
            "endogenous_variables": endogenous_variables,
            "instrument_variables": instrument_variables,
            "first_stage": first_stage,
            "fitted_endogenous": fitted_endogenous,
            "design_matrix_columns": terms,
            "instrument_matrix_columns": [str(column) for column in z_matrix.columns],
            **design.metadata,
        },
        raw_result=raw_result,
    )
