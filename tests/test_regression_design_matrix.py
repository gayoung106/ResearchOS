"""회귀 공통 설계행렬 생성 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src.statistics.regression.design_matrix import (
    RegressionDesignMatrix,
    prepare_regression_design_matrix,
)


def design_dataframe() -> pd.DataFrame:
    """설계행렬 테스트용 데이터를 생성한다."""
    return pd.DataFrame(
        {
            "y": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "x": [
                10,
                20,
                30,
                40,
                50,
                60,
            ],
            "country": [
                "US",
                "JP",
                "KR",
                "US",
                "JP",
                "KR",
            ],
        }
    )


def test_prepare_design_matrix_without_fixed_effects() -> None:
    result = prepare_regression_design_matrix(
        design_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    assert isinstance(
        result,
        RegressionDesignMatrix,
    )
    assert result.outcome.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    assert result.predictors.columns.tolist() == [
        "x",
    ]
    assert result.fixed_effect_columns == []
    assert result.metadata["fixed_effects"] == []
    assert result.metadata["fixed_effect_reference_categories"] == {}
    assert result.metadata["dropped_case_count"] == 0


def test_prepare_design_matrix_encodes_fixed_effects() -> None:
    result = prepare_regression_design_matrix(
        design_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    assert result.predictors.columns.tolist() == [
        "x",
        "country_KR",
        "country_US",
    ]
    assert result.fixed_effect_columns == [
        "country_KR",
        "country_US",
    ]
    assert result.metadata["fixed_effect_reference_categories"] == {
        "country": "JP",
    }

    assert result.predictors["country_KR"].tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result.predictors["country_US"].tolist() == [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]


def test_prepare_design_matrix_drops_incomplete_cases() -> None:
    dataframe = design_dataframe()
    dataframe.loc[0, "x"] = None
    dataframe.loc[1, "country"] = None

    result = prepare_regression_design_matrix(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    assert len(result.outcome) == 4
    assert len(result.predictors) == 4
    assert result.metadata["dropped_case_count"] == 2


def test_prepare_design_matrix_does_not_modify_original() -> None:
    dataframe = design_dataframe()
    original = dataframe.copy(deep=True)

    prepare_regression_design_matrix(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


def test_prepare_design_matrix_converts_numeric_strings() -> None:
    dataframe = design_dataframe()
    dataframe["y"] = dataframe["y"].astype(str)
    dataframe["x"] = dataframe["x"].astype(str)

    result = prepare_regression_design_matrix(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
    )

    assert result.outcome.dtype.kind == "f"
    assert result.predictors["x"].dtype.kind == "f"


def test_prepare_design_matrix_deduplicates_variables() -> None:
    result = prepare_regression_design_matrix(
        design_dataframe(),
        dependent_variable="y",
        independent_variables=[
            "x",
            "x",
        ],
        fixed_effects=[
            "country",
            "country",
        ],
    )

    assert result.predictors.columns.tolist() == [
        "x",
        "country_KR",
        "country_US",
    ]
    assert result.metadata["fixed_effects"] == [
        "country",
    ]


def test_missing_fixed_effect_raises() -> None:
    with pytest.raises(
        KeyError,
        match="고정효과 변수가 없습니다",
    ):
        prepare_regression_design_matrix(
            design_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            fixed_effects=["missing_country"],
        )


def test_duplicate_predictor_and_fixed_effect_raises() -> None:
    with pytest.raises(
        ValueError,
        match="중복 지정",
    ):
        prepare_regression_design_matrix(
            design_dataframe(),
            dependent_variable="y",
            independent_variables=[
                "x",
                "country",
            ],
            fixed_effects=["country"],
        )


def test_constant_predictor_raises() -> None:
    dataframe = design_dataframe()
    dataframe["x"] = 1

    with pytest.raises(
        ValueError,
        match="상수 독립변수",
    ):
        prepare_regression_design_matrix(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
        )


def test_constant_fixed_effect_raises() -> None:
    dataframe = design_dataframe()
    dataframe["country"] = "KR"

    with pytest.raises(
        ValueError,
        match="유효 범주가 하나뿐",
    ):
        prepare_regression_design_matrix(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
            fixed_effects=["country"],
        )


def test_empty_complete_cases_raise() -> None:
    dataframe = design_dataframe()
    dataframe["x"] = None

    with pytest.raises(
        ValueError,
        match="완전사례가 없습니다",
    ):
        prepare_regression_design_matrix(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
        )


def test_fixed_effect_dummy_collision_raises() -> None:
    dataframe = design_dataframe()
    dataframe["country_KR"] = [
        0,
        0,
        1,
        0,
        0,
        1,
    ]

    with pytest.raises(
        ValueError,
        match="기존 독립변수와 충돌",
    ):
        prepare_regression_design_matrix(
            dataframe,
            dependent_variable="y",
            independent_variables=[
                "x",
                "country_KR",
            ],
            fixed_effects=["country"],
        )


def test_mixed_type_categories_use_deterministic_order() -> None:
    dataframe = design_dataframe()
    dataframe["country"] = [
        1,
        "2",
        1,
        "2",
        3,
        3,
    ]

    result = prepare_regression_design_matrix(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    assert result.metadata["fixed_effect_reference_categories"] == {
        "country": 1,
    }
    assert result.fixed_effect_columns == [
        "country_2",
        "country_3",
    ]
