"""포아송 회귀분석 구현."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

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


def fit_poisson(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "poisson_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 100,
) -> RegressionResult:
    """포아송 회귀모형을 적합하고 발생률비를 포함해 반환한다."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"지원하지 않는 공분산 추정방식입니다: {covariance_type}")

    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="포아송",
    )

    outcome = design.outcome
    predictors = design.predictors

    if (outcome < 0).any():
        raise ValueError("포아송 회귀 종속변수는 0 이상의 값이어야 합니다.")

    rounded_outcome = np.round(outcome)
    if not np.allclose(outcome, rounded_outcome):
        raise ValueError("포아송 회귀 종속변수는 음이 아닌 정수로 코딩되어야 합니다.")

    outcome = rounded_outcome.astype(float)

    if add_intercept:
        predictors = sm.add_constant(
            predictors,
            has_constant="add",
        )

    model = sm.GLM(
        outcome,
        predictors,
        family=sm.families.Poisson(),
    )

    fit_options: dict[str, Any] = {
        "maxiter": maximum_iterations,
    }
    if covariance_type != "nonrobust":
        fit_options["cov_type"] = covariance_type

    fitted = model.fit(**fit_options)

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

    converged = bool(getattr(fitted, "converged", True))
    warnings: list[str] = []

    if not converged:
        warnings.append("포아송 회귀모형이 수렴하지 않았습니다.")

    residual_degrees_of_freedom = float(fitted.df_resid)
    dispersion_ratio = (
        float(fitted.pearson_chi2 / residual_degrees_of_freedom)
        if residual_degrees_of_freedom > 0
        else math.nan
    )

    if math.isfinite(dispersion_ratio) and dispersion_ratio > 1.5:
        warnings.append(
            "Pearson 분산비가 1.5를 초과하여 과산포 가능성이 있습니다. "
            "Negative Binomial 모형을 검토하세요."
        )

    zero_count = int((outcome == 0).sum())
    zero_proportion = float(zero_count / len(outcome))

    if zero_proportion > 0.7:
        warnings.append(
            "종속변수의 0 비율이 70%를 초과합니다. "
            "영과잉 포아송 또는 영과잉 음이항 모형을 검토하세요."
        )

    null_deviance = float(fitted.null_deviance)
    deviance = float(fitted.deviance)
    pseudo_r_squared_deviance = (
        float(1 - deviance / null_deviance) if not np.isclose(null_deviance, 0.0) else None
    )

    bic_value = getattr(fitted, "bic_llf", None)
    if bic_value is None:
        bic_value = fitted.bic

    fit_statistics: dict[str, Any] = {
        "log_likelihood": float(fitted.llf),
        "deviance": deviance,
        "null_deviance": null_deviance,
        "pearson_chi_square": float(fitted.pearson_chi2),
        "dispersion_ratio": dispersion_ratio,
        "pseudo_r_squared_deviance": pseudo_r_squared_deviance,
        "aic": float(fitted.aic),
        "bic": float(bic_value),
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "outcome_mean": float(outcome.mean()),
        "outcome_variance": float(outcome.var(ddof=1)),
        "zero_count": zero_count,
        "zero_proportion": zero_proportion,
    }

    return RegressionResult(
        model_id=model_id,
        model_type="poisson",
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
            **design.metadata,
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
