"""명목형 변수 더미 인코딩 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess.executor import execute_preprocessing_plan
from src.preprocess.planner import (
    PreprocessingAction,
    PreprocessingPlan,
)


def dummy_plan(
    variable_name: str,
    *,
    parameters: dict | None = None,
    status: str = "approved",
) -> PreprocessingPlan:
    """더미 인코딩 작업 하나를 포함한 계획을 생성한다."""
    return PreprocessingPlan(
        actions=[
            PreprocessingAction(
                variable_name=variable_name,
                action_type="dummy_encode_nominal",
                status=status,
                reason="테스트",
                parameters=parameters or {},
                requires_confirmation=True,
                priority="high",
            )
        ],
        warnings=[],
        blocked_variables=[],
    )


def test_nominal_variable_is_dummy_encoded() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
                "서울",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan("region"),
    )

    assert result.records[0].status == "completed"
    assert "region_부산" in result.dataframe.columns
    assert "region_서울" in result.dataframe.columns
    assert "region_대구" not in result.dataframe.columns

    assert result.dataframe["region_부산"].tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert result.dataframe["region_서울"].tolist() == [
        1.0,
        0.0,
        0.0,
        1.0,
    ]


def test_explicit_reference_category_is_used() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
                "서울",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            parameters={
                "reference_category": "서울",
            },
        ),
    )

    assert "region_서울" not in result.dataframe.columns
    assert "region_부산" in result.dataframe.columns
    assert "region_대구" in result.dataframe.columns

    assert result.records[0].details["reference_category"] == "서울"


def test_numeric_reference_category_is_normalized() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                1,
                2,
                3,
                1,
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            parameters={
                "reference_category": "2",
            },
        ),
    )

    assert "region_2" not in result.dataframe.columns
    assert "region_1" in result.dataframe.columns
    assert "region_3" in result.dataframe.columns
    assert result.records[0].details["reference_category"] == 2


def test_missing_values_are_preserved_in_dummy_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                None,
                "부산",
                "대구",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            parameters={
                "reference_category": "서울",
            },
        ),
    )

    generated_columns = result.records[0].details["generated_columns"]

    for column in generated_columns:
        assert np.isnan(
            result.dataframe.loc[
                1,
                column,
            ]
        )


def test_original_variable_is_kept_by_default() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan("region"),
    )

    assert "region" in result.dataframe.columns


def test_original_variable_can_be_dropped() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            parameters={
                "drop_original": True,
            },
        ),
    )

    assert "region" not in result.dataframe.columns


def test_custom_prefix_is_used() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            parameters={
                "prefix": "area",
                "reference_category": "서울",
            },
        ),
    )

    assert "area_부산" in result.dataframe.columns
    assert "area_대구" in result.dataframe.columns


def test_existing_column_name_collision_fails_action() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
            ],
            "region_서울": [
                9,
                9,
                9,
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan("region"),
    )

    assert result.records[0].status == "failed"
    assert "충돌" in result.records[0].message
    assert result.warnings


def test_unknown_reference_category_fails_action() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            parameters={
                "reference_category": "제주",
            },
        ),
    )

    assert result.records[0].status == "failed"
    assert "실제 데이터에 없습니다" in result.records[0].message


def test_single_category_fails_action() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "서울",
                None,
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan("region"),
    )

    assert result.records[0].status == "failed"
    assert "2개 이상" in result.records[0].message


def test_unapproved_dummy_encoding_is_skipped() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "서울",
                "부산",
                "대구",
            ],
        }
    )

    result = execute_preprocessing_plan(
        dataframe,
        dummy_plan(
            "region",
            status="planned",
        ),
        require_approval=True,
    )

    assert result.records[0].status == "skipped"
    assert list(result.dataframe.columns) == [
        "region",
    ]
