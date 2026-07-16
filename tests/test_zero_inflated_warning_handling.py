"""영과잉 모형의 경고 수집과 후보 품질검사 테스트."""

from __future__ import annotations

import warnings

from src.statistics.regression.count import (
    fit_count_regression,
)
from src.statistics.regression.zero_inflated_poisson import (
    fit_zero_inflated_poisson,
)
from tests.test_zero_inflated_count_regression import (
    make_zip_dataframe,
)


def test_zero_inflated_fit_does_not_leak_warnings() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        result = fit_zero_inflated_poisson(
            make_zip_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
        )

    assert captured == []
    assert "optimization_warnings" in result.metadata
    assert "optimization_warning_count" in result.metadata


def test_count_auto_does_not_leak_candidate_warnings() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        result = fit_count_regression(
            make_zip_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
        )

    assert captured == []
    assert result.metadata["count_model_selection_method"] == "dispersion_then_zero_inflation_aic"
    assert "candidate_errors" in result.metadata
