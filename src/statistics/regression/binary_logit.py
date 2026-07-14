"""이항 로지스틱 회귀분석 구현."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

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


def fit_binary_logit(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_id: str = "logit_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """이항 로짓 모형을 적합하고 오즈비를 포함해 반환한다."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"지원하지 않는 공분산 추정방식입니다: {covariance_type}")

    model_data = prepare_model_data(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    unique_outcomes = sorted(model_data[dependent_variable].unique().tolist())
    if unique_outcomes != [0, 1]:
        raise ValueError(
            f"이항 로짓 종속변수는 0과 1로 코딩되어야 합니다. 현재 값: {unique_outcomes}"
        )

    outcome = model_data[dependent_variable]
    predictors = model_data[independent_variables]

    if add_intercept:
        predictors = sm.add_constant(
            predictors,
            has_constant="add",
        )

    model = sm.Logit(outcome, predictors)

    try:
        if covariance_type == "nonrobust":
            fitted = model.fit(
                disp=False,
                maxiter=maximum_iterations,
            )
        else:
            fitted = model.fit(
                disp=False,
                maxiter=maximum_iterations,
                cov_type=covariance_type,
            )
    except PerfectSeparationError as error:
        raise ValueError("완전분리로 인해 이항 로짓 모형을 추정할 수 없습니다.") from error

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
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )

    warnings: list[str] = []
    converged = bool(fitted.mle_retvals.get("converged", False))

    if not converged:
        warnings.append("이항 로짓 모형이 수렴하지 않았습니다.")

    event_count = int(outcome.sum())
    non_event_count = int(len(outcome) - event_count)
    if min(event_count, non_event_count) < 10:
        warnings.append("사건 또는 비사건 사례가 10개 미만이어서 추정이 불안정할 수 있습니다.")

    fit_statistics = {
        "log_likelihood": float(fitted.llf),
        "null_log_likelihood": float(fitted.llnull),
        "likelihood_ratio_statistic": float(fitted.llr),
        "likelihood_ratio_p_value": float(fitted.llr_pvalue),
        "pseudo_r_squared_mcfadden": float(fitted.prsquared),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "event_count": event_count,
        "non_event_count": non_event_count,
    }

    return RegressionResult(
        model_id=model_id,
        model_type="binary_logit",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "dropped_case_count": len(dataframe) - len(model_data),
        },
        raw_result=fitted,
    )
