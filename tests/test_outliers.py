"""이상치 진단 모듈 테스트."""

import pandas as pd

from src.preprocess.outliers import (
    build_outlier_report,
    detect_iqr_outliers,
    detect_mahalanobis_outliers,
    detect_univariate_outliers,
    detect_zscore_outliers,
    outlier_report_summary,
)


def test_zscore_detects_extreme_value() -> None:
    series = pd.Series(
        [0] * 20 + [100],
        name="score",
    )

    result = detect_zscore_outliers(
        series,
        threshold=3.0,
    )

    assert result.outlier_count == 1
    assert result.outlier_indices == [20]


def test_iqr_detects_extreme_value() -> None:
    series = pd.Series(
        [1, 2, 2, 3, 100],
        name="score",
    )

    result = detect_iqr_outliers(series)

    assert result.outlier_count == 1
    assert result.outlier_indices == [4]


def test_constant_variable_returns_warning() -> None:
    series = pd.Series(
        [5, 5, 5, 5],
        name="constant",
    )

    result = detect_zscore_outliers(series)

    assert result.outlier_count == 0
    assert result.warnings


def test_univariate_detection_runs_two_methods() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 100],
            "group": ["A", "B", "A", "B"],
        }
    )

    results = detect_univariate_outliers(dataframe)

    assert len(results) == 2
    assert {result.method for result in results} == {
        "zscore",
        "iqr",
    }


def test_mahalanobis_returns_distances() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 10, 4],
            "y": [0, 1, 2, 3, -10, 4],
        }
    )

    result = detect_mahalanobis_outliers(
        dataframe,
        ["x", "y"],
        significance_level=0.05,
    )

    assert result.valid_case_count == 6
    assert len(result.distances) == 6
    assert result.cutoff > 0


def test_mahalanobis_uses_complete_cases() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [0, 1, 2, None, 4],
            "y": [0, 1, 2, 3, 4],
        }
    )

    result = detect_mahalanobis_outliers(
        dataframe,
        ["x", "y"],
        significance_level=0.05,
    )

    assert result.valid_case_count == 4
    assert 3 not in result.distances.index


def test_build_outlier_report() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 100, 4, 5],
            "y": [2, 3, 4, -100, 5, 6],
        }
    )

    report = build_outlier_report(
        dataframe,
        mahalanobis_variables=["x", "y"],
    )
    summary = outlier_report_summary(report)

    assert summary["univariate_result_count"] == 4
    assert summary["mahalanobis_available"] is True


def test_invalid_mahalanobis_input_is_reported() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3],
        }
    )

    report = build_outlier_report(
        dataframe,
        mahalanobis_variables=["x"],
    )

    assert report.mahalanobis_result is None
    assert report.warnings
