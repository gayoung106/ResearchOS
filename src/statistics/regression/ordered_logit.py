"""순서형 로지스틱 회귀분석 구현."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    prepare_model_data,
)


def fit_ordered_logit(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_id: str = "ordered_logit_1",
    maximum_iterations: int = 200,
) -> RegressionResult:
    """순서형 로짓 모형을 적합하고 공통 결과형식으로 반환한다."""
    model_data = prepare_model_data(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    outcome = model_data[dependent_variable]
    unique_outcomes = sorted(outcome.unique().tolist())

    if len(unique_outcomes) < 3:
        raise ValueError("순서형 로짓 종속변수는 최소 3개 범주가 필요합니다.")

    predictors = model_data[independent_variables]

    model = OrderedModel(
        outcome,
        predictors,
        distr="logit",
    )
    fitted = model.fit(
        method="bfgs",
        disp=False,
        maxiter=maximum_iterations,
    )

    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    threshold_terms: list[str] = []

    for term in fitted.params.index:
        estimate = float(fitted.params[term])
        lower = float(confidence_intervals.loc[term, 0])
        upper = float(confidence_intervals.loc[term, 1])
        is_threshold = "/" in str(term)

        if is_threshold:
            threshold_terms.append(str(term))

        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=estimate,
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=(None if is_threshold else float(np.exp(estimate))),
            )
        )

    converged = bool(fitted.mle_retvals.get("converged", False))
    warnings: list[str] = []

    if not converged:
        warnings.append("순서형 로짓 모형이 수렴하지 않았습니다.")

    category_counts = {
        str(category): int(count) for category, count in outcome.value_counts().sort_index().items()
    }
    if min(category_counts.values()) < 10:
        warnings.append("일부 종속변수 범주의 사례 수가 10개 미만입니다.")

    return RegressionResult(
        model_id=model_id,
        model_type="ordered_logit",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=len(model_data),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "category_count": len(unique_outcomes),
        },
        converged=converged,
        standard_error_type="maximum_likelihood",
        warnings=warnings,
        metadata={
            "threshold_terms": threshold_terms,
            "category_counts": category_counts,
            "maximum_iterations": maximum_iterations,
            "dropped_case_count": len(dataframe) - len(model_data),
        },
        raw_result=fitted,
    )
