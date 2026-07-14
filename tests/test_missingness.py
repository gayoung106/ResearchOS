"""결측치 진단 및 처리 추천 테스트."""

import pandas as pd

from src.preprocess.missingness import (
    build_missingness_report,
    case_missingness_summary,
    missingness_pattern_summary,
    missingness_report_summary,
    recommend_missingness_strategy,
    variable_missingness_summary,
)


def test_variable_missingness_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": [1, 2, 3],
        }
    )

    summary = variable_missingness_summary(dataframe)

    row_a = summary.loc[summary["variable_name"] == "a"].iloc[0]

    assert row_a["missing_count"] == 1
    assert row_a["missing_rate"] == 1 / 3


def test_case_missingness_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, None],
            "b": [None, 2],
        }
    )

    summary = case_missingness_summary(dataframe)

    assert summary["missing_count"].tolist() == [1, 1]
    assert summary["complete_case"].tolist() == [False, False]


def test_missingness_pattern_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, None, 1],
            "b": [1, 2, None],
        }
    )

    summary = missingness_pattern_summary(dataframe)

    assert summary["frequency"].sum() == 3
    assert set(summary["pattern"]) == {"00", "10", "01"}


def test_no_missing_recommends_no_action() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    report = build_missingness_report(dataframe)

    assert report.overall_missing_rate == 0
    assert report.recommendations[0].strategy == "no_action"


def test_low_missingness_recommends_complete_case_review() -> None:
    dataframe = pd.DataFrame(
        {
            "a": list(range(100)),
            "b": [None] + list(range(99)),
        }
    )
    report = build_missingness_report(dataframe)

    strategies = {recommendation.strategy for recommendation in report.recommendations}

    assert "complete_case_review" in strategies


def test_moderate_missingness_recommends_multiple_imputation() -> None:
    variable_summary = pd.DataFrame(
        {
            "variable_name": ["a"],
            "missing_rate": [0.10],
        }
    )

    recommendations = recommend_missingness_strategy(
        variable_summary,
        overall_missing_rate=0.10,
    )

    strategies = {recommendation.strategy for recommendation in recommendations}

    assert "multiple_imputation_review" in strategies


def test_high_missingness_variable_is_flagged() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [None, None, 3, 4],
            "b": [1, 2, 3, 4],
        }
    )
    report = build_missingness_report(dataframe)

    strategies = {recommendation.strategy for recommendation in report.recommendations}

    assert "high_missingness_variable_review" in strategies


def test_missingness_report_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": [1, 2, 3],
        }
    )

    report = build_missingness_report(dataframe)
    summary = missingness_report_summary(report)

    assert summary["row_count"] == 3
    assert summary["column_count"] == 2
    assert summary["total_missing_count"] == 1
    assert summary["complete_case_count"] == 2
