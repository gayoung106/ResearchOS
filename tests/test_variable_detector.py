"""변수 측정수준 자동 탐지기 테스트."""

import pandas as pd

from src.preprocess.detector import (
    detect_dataframe_variables,
    detect_variable_level,
    detection_summary,
)


def test_boolean_is_detected_as_binary() -> None:
    result = detect_variable_level(
        "flag",
        pd.Series([True, False, True]),
    )

    assert result.detected_level == "binary"
    assert result.status == "detected"


def test_numeric_two_values_requires_review() -> None:
    result = detect_variable_level(
        "group",
        pd.Series([1, 2, 1, 2]),
    )

    assert result.detected_level == "binary"
    assert result.status == "review_required"


def test_likert_candidate_uses_label_evidence() -> None:
    result = detect_variable_level(
        "satisfaction",
        pd.Series([1, 2, 3, 4, 5]),
        variable_label="직무 만족도",
        value_labels={
            1: "전혀 만족하지 않음",
            5: "매우 만족함",
        },
    )

    assert result.detected_level == "scale_item"
    assert result.status == "review_required"


def test_small_integer_range_is_ordinal_candidate() -> None:
    result = detect_variable_level(
        "education",
        pd.Series([1, 2, 3, 4, 3, 2]),
    )

    assert result.detected_level == "ordinal"
    assert "nominal" in result.alternatives


def test_large_integer_range_is_count_candidate() -> None:
    result = detect_variable_level(
        "event_count",
        pd.Series(range(20)),
    )

    assert result.detected_level == "count"


def test_float_is_detected_as_continuous() -> None:
    result = detect_variable_level(
        "income",
        pd.Series([10.5, 20.1, 35.2, 41.7]),
    )

    assert result.detected_level == "continuous"
    assert result.status == "detected"


def test_dataframe_detection_and_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "binary": [0, 1, 0, 1],
            "score": [1.2, 2.3, 3.4, 4.5],
            "category": ["A", "B", "C", "A"],
        }
    )

    detections = detect_dataframe_variables(dataframe)
    summary = detection_summary(detections)

    assert summary["variable_count"] == 3
    assert summary["level_counts"]["binary"] == 1
    assert summary["level_counts"]["continuous"] == 1
    assert summary["level_counts"]["nominal"] == 1
