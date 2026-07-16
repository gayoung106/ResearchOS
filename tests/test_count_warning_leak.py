"""Count 자동선택의 기본모형 경고 누출 방지 테스트."""

from __future__ import annotations

import warnings

from src.statistics.regression.count import fit_count_regression
from tests.test_zero_inflated_count_regression import (
    make_zip_dataframe,
)


def test_count_auto_captures_baseline_model_warnings() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        result = fit_count_regression(
            make_zip_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
        )

    assert captured == []
    assert "poisson_fit_warnings" in result.metadata
    assert "negative_binomial_fit_warnings" in result.metadata
