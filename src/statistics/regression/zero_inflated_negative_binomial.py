"""영과잉 음이항 회귀분석 구현."""

from __future__ import annotations

import warnings as python_warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP,
)

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
)
from src.statistics.regression.design_matrix import (
    prepare_regression_design_matrix,
)

SUPPORTED_COVARIANCE_TYPES = {
    "nonrobust",
    "HC0",
    "HC1",
    "HC2",
    "HC3",
}


def _warning_records_to_metadata(
    records: list[python_warnings.WarningMessage],
) -> list[dict[str, str]]:
    """적합 과정 경고를 직렬화 가능한 형태로 변환한다."""
    output: list[dict[str, str]] = []

    for record in records:
        item = {
            "category": record.category.__name__,
            "message": str(record.message),
        }
        if item not in output:
            output.append(item)

    return output


def fit_zero_inflated_negative_binomial(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "zero_inflated_negative_binomial_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 500,
) -> RegressionResult:
    """절편 전용 inflation 식을 사용하는 ZINB-P 모형을 적합한다."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"지원하지 않는 공분산 추정방식입니다: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="영과잉 음이항",
    )

    outcome = design.outcome
    predictors = design.predictors

    if (outcome < 0).any():
        raise ValueError("영과잉 음이항 종속변수는 0 이상의 값이어야 합니다.")

    rounded_outcome = np.round(outcome)
    if not np.allclose(outcome, rounded_outcome):
        raise ValueError("영과잉 음이항 종속변수는 음이 아닌 정수로 코딩되어야 합니다.")

    outcome = rounded_outcome.astype(float)

    if add_intercept:
        predictors = sm.add_constant(
            predictors,
            has_constant="add",
        )

    inflation_predictors = np.ones(
        (
            len(outcome),
            1,
        ),
        dtype=float,
    )

    model = ZeroInflatedNegativeBinomialP(
        outcome,
        predictors,
        exog_infl=inflation_predictors,
        inflation="logit",
        p=2,
    )

    fit_options: dict[str, Any] = {
        "disp": False,
        "maxiter": maximum_iterations,
    }
    if covariance_type != "nonrobust":
        fit_options["cov_type"] = covariance_type

    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always")

        fitted = model.fit(**fit_options)
        confidence_intervals = fitted.conf_int()
        predicted_zero_proportion = float(np.mean(fitted.predict(which="prob-zero")))

    optimization_warnings = _warning_records_to_metadata(list(captured))

    coefficients: list[ModelCoefficient] = []

    for term in fitted.params.index:
        if str(term).lower() == "alpha":
            continue

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

    converged = bool(
        fitted.mle_retvals.get(
            "converged",
            False,
        )
    )
    result_warnings: list[str] = []

    if not converged:
        result_warnings.append("영과잉 음이항 모형이 수렴하지 않았습니다.")

    if optimization_warnings:
        result_warnings.append("영과잉 음이항 최적화 과정에서 수치 경고가 기록되었습니다.")

    alpha = float(
        fitted.params.get(
            "alpha",
            np.nan,
        )
    )
    if np.isfinite(alpha) and alpha <= 0:
        result_warnings.append("영과잉 음이항 alpha가 0 이하입니다.")

    zero_count = int((outcome == 0).sum())
    zero_proportion = float(zero_count / len(outcome))

    return RegressionResult(
        model_id=model_id,
        model_type="zero_inflated_negative_binomial",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "alpha": alpha,
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": zero_count,
            "zero_proportion": zero_proportion,
            "predicted_zero_proportion": (predicted_zero_proportion),
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=result_warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "inflation_model": "logit_intercept_only",
            "negative_binomial_parameterization": "NB2",
            "optimization_warnings": optimization_warnings,
            "optimization_warning_count": len(optimization_warnings),
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
