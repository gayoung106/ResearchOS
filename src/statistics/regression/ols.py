"""OLS 회귀분석 구현."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    prepare_model_data,
)

SUPPORTED_COVARIANCE_TYPES = {
    "nonrobust",
    "HC0",
    "HC1",
    "HC2",
    "HC3",
}


def fit_ols(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_id: str = "ols_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
) -> RegressionResult:
    """OLS 모형을 적합하고 공통 결과형식으로 반환한다."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"지원하지 않는 공분산 추정방식입니다: {covariance_type}")

    model_data = prepare_model_data(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    outcome = model_data[dependent_variable]
    predictors = model_data[independent_variables]

    if add_intercept:
        predictors = sm.add_constant(
            predictors,
            has_constant="add",
        )

    model = sm.OLS(outcome, predictors)

    if covariance_type == "nonrobust":
        fitted = model.fit()
    else:
        fitted = model.fit(cov_type=covariance_type)

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

    warnings: list[str] = []
    if len(model_data) <= len(predictors.columns) + 1:
        warnings.append("표본 수가 추정 모수 수에 비해 매우 적습니다.")

    fit_statistics: dict[str, Any] = {
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "f_statistic": (
            float(fitted.fvalue)
            if fitted.fvalue is not None and not math.isnan(float(fitted.fvalue))
            else None
        ),
        "f_p_value": (
            float(fitted.f_pvalue)
            if fitted.f_pvalue is not None and not math.isnan(float(fitted.f_pvalue))
            else None
        ),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "residual_degrees_of_freedom": float(fitted.df_resid),
    }

    return RegressionResult(
        model_id=model_id,
        model_type="ols",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "dropped_case_count": len(dataframe) - len(model_data),
        },
        raw_result=fitted,
    )
