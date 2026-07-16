"""영과잉 후보 적합 경고 누출 방지 테스트."""

from __future__ import annotations

import warnings

from src.statistics.regression.count import fit_count_regression
from tests.test_zero_inflated_count_regression import (
    make_zip_dataframe,
)


def test_zero_inflated_candidate_warnings_are_captured() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        result = fit_count_regression(
            make_zip_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
        )

    assert captured == []
    assert result.metadata["zero_inflated_candidates_fitted"] is True
    assert "candidate_errors" in result.metadata
