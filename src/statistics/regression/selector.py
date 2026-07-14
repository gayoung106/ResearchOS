"""측정수준에 따라 회귀모형을 선택하는 실행기."""

from __future__ import annotations

import pandas as pd

from src.statistics.regression.base import RegressionResult
from src.statistics.regression.binary_logit import fit_binary_logit
from src.statistics.regression.ols import fit_ols
from src.statistics.regression.ordered_logit import fit_ordered_logit


def fit_regression_by_level(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    measurement_level: str,
    model_id: str = "model_1",
) -> RegressionResult:
    """종속변수 측정수준에 맞는 기본 회귀모형을 실행한다."""
    if measurement_level == "continuous":
        return fit_ols(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            model_id=model_id,
            covariance_type="HC3",
        )

    if measurement_level == "binary":
        return fit_binary_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            model_id=model_id,
            covariance_type="HC3",
        )

    if measurement_level in {"ordinal", "scale_item"}:
        return fit_ordered_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            model_id=model_id,
        )

    raise ValueError(
        f"현재 Regression Core에서 지원하지 않는 종속변수 측정수준입니다: {measurement_level}"
    )
