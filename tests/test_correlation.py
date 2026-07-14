"""상관분석 엔진 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.context import ResearchContext
from src.pipeline.correlation_step import CorrelationAnalysisStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.correlation import (
    correlate_pair,
    run_correlation_analysis,
)


def test_pearson_perfect_correlation() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        }
    )

    result = correlate_pair(
        dataframe,
        "x",
        "y",
        method="pearson",
    )

    assert result.coefficient == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.0)
    assert result.sample_size == 5


def test_spearman_correlation() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
        }
    )

    result = correlate_pair(
        dataframe,
        "x",
        "y",
        method="spearman",
    )

    assert result.coefficient == pytest.approx(1.0)


def test_pairwise_missing_cases() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [1, None, 3, 4],
        }
    )

    result = correlate_pair(dataframe, "x", "y")

    assert result.sample_size == 2
    assert result.coefficient is None
    assert result.warnings


def test_multiple_testing_adjustment() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [2, 4, 6, 8, 10, 12],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    report = run_correlation_analysis(
        dataframe,
        ["a", "b", "c"],
        p_adjust_method="holm",
    )

    assert len(report.results) == 3
    assert all(result.adjusted_p_value is not None for result in report.results)


def test_high_correlation_warning() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [1, 2, 3, 4, 5],
        }
    )

    report = run_correlation_analysis(
        dataframe,
        ["x", "y"],
    )

    assert report.warnings


def test_publication_table_is_created() -> None:
    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 3, 4, 5],
            "c": [5, 4, 3, 2],
        }
    )

    report = run_correlation_analysis(
        dataframe,
        ["a", "b", "c"],
    )

    assert list(report.publication_table.columns[:4]) == [
        "번호",
        "변수",
        "평균",
        "표준편차",
    ]
    assert len(report.publication_table) == 3


def test_correlation_pipeline_step(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime(
        dataframe=pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2, 3, 4, 5],
            }
        )
    )
    step = CorrelationAnalysisStep(
        runtime,
        ["a", "b"],
    )
    context = ResearchContext(
        project_name="테스트",
    )

    result = step.run(context, tmp_path)

    assert result.success is True
    assert len(result.output_files) == 5
    assert all(Path(path).exists() for path in result.output_files)
    assert runtime.get_artifact("correlation_report").results
