"""계수형 종속변수 회귀모형 자동선택."""

from __future__ import annotations

import math

import pandas as pd

from src.statistics.regression.base import RegressionResult
from src.statistics.regression.negative_binomial import fit_negative_binomial
from src.statistics.regression.poisson import fit_poisson


def fit_count_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "count_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 200,
    dispersion_threshold: float = 1.5,
) -> RegressionResult:
    """Poisson을 우선 적합하고 과산포 시 NB2를 선택한다."""
    if dispersion_threshold <= 1:
        raise ValueError("과산포 자동선택 기준은 1보다 커야 합니다.")
    poisson = fit_poisson(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_id=model_id,
        covariance_type=covariance_type,
        add_intercept=add_intercept,
        maximum_iterations=maximum_iterations,
    )
    dispersion = float(poisson.fit_statistics["dispersion_ratio"])
    common = {
        "count_model_selection_method": "poisson_pearson_dispersion",
        "dispersion_threshold": dispersion_threshold,
        "poisson_dispersion_ratio": dispersion,
        "poisson_aic": float(poisson.fit_statistics["aic"]),
        "poisson_bic": float(poisson.fit_statistics["bic"]),
    }
    if not math.isfinite(dispersion) or dispersion <= dispersion_threshold:
        poisson.metadata.update(
            {
                **common,
                "selected_count_model": "poisson",
                "selection_reason": "Poisson Pearson 분산비가 기준 이하이므로 Poisson 모형을 유지했습니다.",
                "negative_binomial_fitted": False,
            }
        )
        return poisson
    nb = fit_negative_binomial(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_id=model_id,
        covariance_type=covariance_type,
        add_intercept=add_intercept,
        maximum_iterations=maximum_iterations,
    )
    nb.metadata.update(
        {
            **common,
            "selected_count_model": "negative_binomial",
            "selection_reason": "Poisson Pearson 분산비가 기준을 초과하여 NB2 음이항 모형을 선택했습니다.",
            "negative_binomial_fitted": True,
            "negative_binomial_aic": float(nb.fit_statistics["aic"]),
            "negative_binomial_bic": float(nb.fit_statistics["bic"]),
            "negative_binomial_alpha": float(nb.fit_statistics["alpha"]),
        }
    )
    nb.warnings.insert(
        0, "Poisson 모형에서 과산포가 확인되어 NB2 음이항 모형으로 자동 전환했습니다."
    )
    return nb
