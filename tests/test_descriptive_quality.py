"""기술통계 및 데이터 품질 엔진 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.context import ResearchContext
from src.pipeline.descriptive_step import DescriptiveStatisticsStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.descriptive import (
    build_descriptive_report,
    generate_quality_warnings,
    summarize_categorical_variable,
    summarize_numeric_variable,
)


def test_numeric_summary() -> None:
    series = pd.Series(
        [1, 2, 3, 4, 5],
        name="score",
    )

    result = summarize_numeric_variable(series)

    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["range"] == pytest.approx(4.0)


def test_categorical_summary() -> None:
    series = pd.Series(
        ["A", "A", "B", None],
        name="group",
    )

    result = summarize_categorical_variable(series)

    assert result["frequency"].sum() == 4
    assert "valid_percent" in result.columns


def test_quality_warning_for_constant_variable() -> None:
    dataframe = pd.DataFrame(
        {
            "constant": [1, 1, 1],
        }
    )

    warnings = generate_quality_warnings(dataframe)

    assert any(warning.warning_type == "constant" for warning in warnings)


def test_quality_warning_for_duplicate_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 1, 2],
            "y": [3, 3, 4],
        }
    )

    warnings = generate_quality_warnings(dataframe)

    assert any(warning.warning_type == "duplicate_rows" for warning in warnings)


def test_build_descriptive_report() -> None:
    dataframe = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0],
            "group": ["A", "B", "A"],
        }
    )

    report = build_descriptive_report(dataframe)

    assert len(report.numeric_summary) == 1
    assert not report.categorical_summary.empty
    assert report.dataset_summary["row_count"] == 3


def test_descriptive_pipeline_step(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime(
        dataframe=pd.DataFrame(
            {
                "score": [1.0, 2.0, 3.0],
                "group": ["A", "B", "A"],
            }
        )
    )
    context = ResearchContext(
        project_name="테스트",
    )
    step = DescriptiveStatisticsStep(runtime)

    result = step.run(context, tmp_path)

    assert result.success is True
    assert len(result.output_files) == 4
    assert all(Path(path).exists() for path in result.output_files)
    assert runtime.get_artifact("descriptive_report").dataset_summary["row_count"] == 3
