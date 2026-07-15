"""파일 로딩 단계와 기본 Pipeline Builder 테스트."""

from pathlib import Path

import pandas as pd

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from src.pipeline.io_steps import DataLoadingStep
from src.pipeline.runtime import PipelineRuntime


def context() -> ResearchContext:
    return ResearchContext(
        project_name="테스트 연구",
        research_topic="테스트",
        research_questions=["질문"],
    )


def test_data_loading_step_with_explicit_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "outcome": [0, 1, 0],
            "score": [1.2, 2.3, 3.4],
        }
    ).to_csv(
        source,
        index=False,
        encoding="utf-8-sig",
    )

    runtime = PipelineRuntime()
    step = DataLoadingStep(
        runtime,
        source_file=source,
    )

    result = step.run(
        context(),
        tmp_path,
    )

    assert result.success is True
    assert runtime.dataframe is not None
    assert runtime.dataframe.shape == (3, 2)
    assert len(result.output_files) == 2


def test_data_loading_step_rejects_multiple_files(
    tmp_path: Path,
) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()

    pd.DataFrame(
        {
            "x": [1],
        }
    ).to_csv(
        rawdata / "a.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "x": [2],
        }
    ).to_csv(
        rawdata / "b.csv",
        index=False,
    )

    runtime = PipelineRuntime()
    step = DataLoadingStep(runtime)

    try:
        step.run(
            context(),
            tmp_path,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_default_pipeline_registers_expected_steps(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate({})
    variable_map = VariableMap.model_validate(
        {
            "variables": {},
        }
    )

    orchestrator, runtime = build_default_pipeline(
        context=context(),
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        working_directory=tmp_path,
    )

    assert runtime.dataframe is None

    expected = {
        "01_data_loading",
        "02_variable_detection",
        "02_evidence_resolution",
        "03_preprocessing_plan",
        "04_scale_reliability",
        "05_missingness",
        "06_outliers",
        "07_descriptive_statistics",
        "08_correlation_analysis",
    }

    actual = set(orchestrator.registry.names())

    assert expected.issubset(actual)


def test_default_pipeline_runs_minimal_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    source = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "outcome": [0, 1, 0, 1],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(
        source,
        index=False,
        encoding="utf-8-sig",
    )

    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
            }
        }
    )

    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "role": "dependent",
                    "measurement_level": "binary",
                    "evidence": {
                        "codebook_level": "binary",
                    },
                }
            }
        }
    )

    orchestrator, runtime = build_default_pipeline(
        context=context(),
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        working_directory=tmp_path,
        source_file=source,
    )

    result = orchestrator.run()

    assert result.success is True
    assert runtime.dataframe is not None
    assert runtime.detections
    assert runtime.resolved_levels
    assert runtime.missingness_report is not None
    assert runtime.outlier_report is not None
