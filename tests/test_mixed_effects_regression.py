"""Random Intercept 혼합효과 회귀모형 테스트."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.statistics.regression.mixed_effects import (
    fit_random_intercept,
)


def make_random_intercept_dataframe() -> pd.DataFrame:
    """재현 가능한 Random Intercept 검증용 자료를 만든다."""
    rng = np.random.default_rng(20260720)

    group_count = 24
    observations_per_group = 8

    groups = np.repeat(
        np.arange(group_count),
        observations_per_group,
    )

    x = rng.normal(
        size=(group_count * observations_per_group),
    )

    random_intercepts = rng.normal(
        loc=0.0,
        scale=1.2,
        size=group_count,
    )

    error = rng.normal(
        loc=0.0,
        scale=0.5,
        size=len(groups),
    )

    y = 1.5 + 2.0 * x + random_intercepts[groups] + error

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "group": groups,
        }
    )


def test_random_intercept_recovers_fixed_effect() -> None:
    result = fit_random_intercept(
        make_random_intercept_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )

    coefficient = next(item for item in result.coefficients if item.term == "x")

    assert result.model_type == "mixed_random_intercept"
    assert result.converged is True
    assert result.sample_size == 192

    assert coefficient.estimate == pytest.approx(
        2.0,
        abs=0.15,
    )

    assert result.fit_statistics["group_count"] == 24

    assert result.fit_statistics["random_intercept_variance"] > 0

    assert 0 < result.fit_statistics["intraclass_correlation"] < 1

    assert result.fit_statistics["singleton_group_count"] == 0

    assert result.metadata["group_variable"] == "group"

    assert result.standard_error_type == "model_based"


def test_random_intercept_drops_missing_and_infinite_cases() -> None:
    dataframe = make_random_intercept_dataframe()

    dataframe.loc[0, "y"] = np.nan
    dataframe.loc[1, "x"] = np.nan
    dataframe.loc[2, "group"] = np.nan
    dataframe.loc[3, "x"] = np.inf

    result = fit_random_intercept(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )

    assert result.sample_size == 188
    assert result.metadata["dropped_case_count"] == 4


def test_random_intercept_requires_existing_group_variable() -> None:
    with pytest.raises(
        KeyError,
        match="그룹변수",
    ):
        fit_random_intercept(
            make_random_intercept_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="missing_group",
        )


def test_random_intercept_rejects_single_group() -> None:
    dataframe = make_random_intercept_dataframe()
    dataframe["group"] = "only_group"

    with pytest.raises(
        ValueError,
        match="두 개 이상의 그룹",
    ):
        fit_random_intercept(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="group",
        )


def test_random_intercept_rejects_constant_outcome() -> None:
    dataframe = make_random_intercept_dataframe()
    dataframe["y"] = 1.0

    with pytest.raises(
        ValueError,
        match="종속변수가 상수",
    ):
        fit_random_intercept(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="group",
        )


def test_random_intercept_rejects_unsupported_optimizer() -> None:
    with pytest.raises(
        ValueError,
        match=("지원하지 않는 혼합효과 최적화 방식"),
    ):
        fit_random_intercept(
            make_random_intercept_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="group",
            method="unsupported",
        )


def test_random_intercept_rejects_nonpositive_max_iterations() -> None:
    with pytest.raises(
        ValueError,
        match="최대 반복 횟수",
    ):
        fit_random_intercept(
            make_random_intercept_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="group",
            max_iterations=0,
        )


def test_random_intercept_rejects_collinear_predictors() -> None:
    dataframe = make_random_intercept_dataframe()
    dataframe["x_duplicate"] = dataframe["x"] * 2

    with pytest.raises(
        ValueError,
        match="완전 다중공선성",
    ):
        fit_random_intercept(
            dataframe,
            dependent_variable="y",
            independent_variables=[
                "x",
                "x_duplicate",
            ],
            group_variable="group",
        )


def test_random_intercept_warns_about_small_and_singleton_groups() -> None:
    dataframe = make_random_intercept_dataframe().iloc[:16].copy()

    dataframe["group"] = [0] * 7 + [1] * 7 + [2] + [3]

    result = fit_random_intercept(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )

    assert result.fit_statistics["group_count"] == 4

    assert result.fit_statistics["singleton_group_count"] == 2

    assert any("그룹 수가 5개 미만" in message for message in result.warnings)

    assert any("관측치가 1개뿐인 그룹이 2개" in message for message in result.warnings)


def test_random_intercept_fit_warnings_do_not_leak() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        result = fit_random_intercept(
            make_random_intercept_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="group",
        )

    assert captured == []
    assert isinstance(result.warnings, list)


def test_random_intercept_does_not_modify_original_dataframe() -> None:
    dataframe = make_random_intercept_dataframe()
    original = dataframe.copy(deep=True)

    fit_random_intercept(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )
