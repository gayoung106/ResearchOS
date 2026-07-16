"""범주형 변수 측정수준 탐지 테스트."""

from __future__ import annotations

import pandas as pd

from src.common.config_models import VariableMap
from src.preprocess.detector import (
    detect_dataframe_variables,
    detect_variable_level,
)


def test_explicit_nominal_level_overrides_numeric_inference() -> None:
    result = detect_variable_level(
        "region",
        pd.Series([1, 2, 3, 1, 2, 3]),
        declared_level="nominal",
    )

    assert result.detected_level == "nominal"
    assert result.status == "detected"
    assert result.confidence == 1.0
    assert "VariableMap" in result.evidence.notes[0]


def test_explicit_continuous_level_overrides_small_integer_range() -> None:
    result = detect_variable_level(
        "score",
        pd.Series([1, 2, 3, 4, 5]),
        declared_level="continuous",
    )

    assert result.detected_level == "continuous"
    assert result.status == "detected"
    assert result.confidence == 1.0


def test_unknown_declared_level_does_not_override_detection() -> None:
    result = detect_variable_level(
        "education",
        pd.Series([1, 2, 3, 4, 3, 2]),
        declared_level="unknown",
    )

    assert result.detected_level == "ordinal"
    assert result.status == "review_required"


def test_numeric_codes_with_nominal_label_are_nominal_candidate() -> None:
    result = detect_variable_level(
        "region",
        pd.Series([1, 2, 3, 1, 2, 3]),
        variable_label="거주 지역",
        value_labels={
            1: "서울",
            2: "부산",
            3: "대구",
        },
    )

    assert result.detected_level == "nominal"
    assert result.status == "review_required"
    assert "ordinal" in result.alternatives


def test_numeric_codes_with_nominal_value_labels_are_nominal_candidate() -> None:
    result = detect_variable_level(
        "occupation",
        pd.Series([1, 2, 3, 1, 2, 3]),
        value_labels={
            1: "관리자",
            2: "전문가",
            3: "사무직",
        },
    )

    assert result.detected_level == "nominal"
    assert result.status == "review_required"


def test_small_integer_range_without_nominal_evidence_is_ordinal() -> None:
    result = detect_variable_level(
        "education_level",
        pd.Series([1, 2, 3, 4, 3, 2]),
    )

    assert result.detected_level == "ordinal"
    assert result.status == "review_required"
    assert "nominal" in result.alternatives


def test_missing_values_do_not_prevent_ordinal_detection() -> None:
    result = detect_variable_level(
        "rank",
        pd.Series([1, 2, None, 3, 4, None, 2]),
    )

    assert result.detected_level == "ordinal"
    assert result.evidence.non_missing_count == 5
    assert result.evidence.unique_count == 4


def test_two_numeric_values_remain_binary_candidate() -> None:
    result = detect_variable_level(
        "group",
        pd.Series([0, 1, 0, None, 1]),
    )

    assert result.detected_level == "binary"
    assert result.status == "review_required"
    assert "nominal" in result.alternatives


def test_string_categories_are_nominal_candidate() -> None:
    result = detect_variable_level(
        "department",
        pd.Series(["인사", "재무", "개발", "인사"]),
    )

    assert result.detected_level == "nominal"
    assert result.status == "review_required"
    assert "string" in result.alternatives


def test_dataframe_detection_prioritizes_variable_map() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [1, 2, 3, 1],
            "score": [1, 2, 3, 4],
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "region": {
                    "measurement_level": "nominal",
                },
                "score": {
                    "measurement_level": "continuous",
                },
            }
        }
    )

    detections = detect_dataframe_variables(
        dataframe,
        variable_map=variable_map,
    )
    levels = {detection.variable_name: detection.detected_level for detection in detections}

    assert levels == {
        "region": "nominal",
        "score": "continuous",
    }


def test_dataframe_detection_uses_metadata_when_variable_map_is_unknown() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [1, 2, 3, 1],
        }
    )
    variable_metadata = pd.DataFrame(
        {
            "variable_name": ["region"],
            "variable_label": ["거주 지역"],
            "value_labels": [
                {
                    1: "서울",
                    2: "부산",
                    3: "대구",
                }
            ],
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "region": {
                    "measurement_level": "unknown",
                }
            }
        }
    )

    detections = detect_dataframe_variables(
        dataframe,
        variable_metadata=variable_metadata,
        variable_map=variable_map,
    )

    assert detections[0].detected_level == "nominal"
    assert detections[0].status == "review_required"
