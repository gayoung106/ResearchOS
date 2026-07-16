"""음이항 회귀분석 구현."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix

SUPPORTED_COVARIANCE_TYPES = {"nonrobust", "HC0", "HC1", "HC2", "HC3"}


def fit_negative_binomial(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "negative_binomial_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 200,
) -> RegressionResult:
    """NB2 음이항 회귀모형을 적합하고 발생률비를 반환한다."""
    if covariance_type not in SUPPORTED_COVARIANCE_TYPES:
        raise ValueError(f"지원하지 않는 공분산 추정방식입니다: {covariance_type}")
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="음이항",
    )
    outcome = design.outcome
    predictors = design.predictors
    if (outcome < 0).any():
        raise ValueError("음이항 회귀 종속변수는 0 이상의 값이어야 합니다.")
    rounded = np.round(outcome)
    if not np.allclose(outcome, rounded):
        raise ValueError("음이항 회귀 종속변수는 음이 아닌 정수로 코딩되어야 합니다.")
    outcome = rounded.astype(float)
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")
    model = sm.NegativeBinomial(outcome, predictors, loglike_method="nb2")
    options: dict[str, Any] = {"disp": False, "maxiter": maximum_iterations}
    if covariance_type != "nonrobust":
        options["cov_type"] = covariance_type
    fitted = model.fit(**options)
    ci = fitted.conf_int()
    coefficients = []
    for term in fitted.params.index:
        if str(term).lower() == "alpha":
            continue
        estimate = float(fitted.params[term])
        lower = float(ci.loc[term, 0])
        upper = float(ci.loc[term, 1])
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
    converged = bool(fitted.mle_retvals.get("converged", False))
    warnings = []
    if not converged:
        warnings.append("음이항 회귀모형이 수렴하지 않았습니다.")
    alpha = float(fitted.params.get("alpha", np.nan))
    if np.isfinite(alpha) and alpha <= 0:
        warnings.append("추정된 과산포 모수 alpha가 0 이하입니다. Poisson 모형을 우선 검토하세요.")
    zero_count = int((outcome == 0).sum())
    zero_proportion = float(zero_count / len(outcome))
    if zero_proportion > 0.7:
        warnings.append("종속변수의 0 비율이 70%를 초과합니다. 영과잉 음이항 모형을 검토하세요.")
    llnull = getattr(fitted, "llnull", None)
    pseudo = (
        float(1 - fitted.llf / llnull)
        if llnull is not None and not np.isclose(float(llnull), 0.0)
        else None
    )
    return RegressionResult(
        model_id=model_id,
        model_type="negative_binomial",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics={
            "log_likelihood": float(fitted.llf),
            "null_log_likelihood": float(llnull) if llnull is not None else None,
            "pseudo_r_squared_mcfadden": pseudo,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "alpha": alpha,
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": zero_count,
            "zero_proportion": zero_proportion,
        },
        converged=converged,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "add_intercept": add_intercept,
            "maximum_iterations": maximum_iterations,
            "negative_binomial_parameterization": "NB2",
            **design.metadata,
            "design_matrix_columns": [str(c) for c in predictors.columns],
            "fixed_effect_column_count": len(design.fixed_effect_columns),
        },
        raw_result=fitted,
    )
