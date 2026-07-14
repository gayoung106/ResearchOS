"""파이프라인 단계 어댑터 테스트."""

from pathlib import Path

import pandas as pd

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.analysis_steps import (
    MissingnessStep,
    OutlierStep,
    VariableDetectionStep,
)
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime


def context() -> ResearchContext:
    return ResearchContext(
        project_name="테스트 연구",
        research_topic="테스트",
        research_questions=["질문"],
    )


def test_runtime_requires_dataframe() -> None:
    runtime = PipelineRuntime()

    try:
        runtime.require_dataframe()
        raised = False
    except RuntimeError:
        raised = True

    assert raised is True


def test_variable_detection_step(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime(
        dataframe=pd.DataFrame(
            {
                "binary": [0, 1, 0],
                "score": [1.2, 2.3, 3.4],
            }
        )
    )
    step = VariableDetectionStep(runtime)

    result = step.run(context(), tmp_path)

    assert result.success is True
    assert len(runtime.detections) == 2
    assert Path(result.output_files[0]).exists()


def test_missingness_step(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime(
        dataframe=pd.DataFrame(
            {
                "a": [1, None, 3],
                "b": [1, 2, 3],
            }
        )
    )
    step = MissingnessStep(runtime)

    result = step.run(context(), tmp_path)

    assert result.success is True
    assert runtime.missingness_report is not None
    assert len(result.output_files) == 4


def test_outlier_step(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime(
        dataframe=pd.DataFrame(
            {
                "x": [1, 2, 3, 100, 4, 5],
                "y": [2, 3, 4, -100, 5, 6],
            }
        )
    )
    step = OutlierStep(
        runtime,
        mahalanobis_variables=["x", "y"],
    )

    result = step.run(context(), tmp_path)

    assert result.success is True
    assert runtime.outlier_report is not None
    assert len(result.output_files) >= 1


def test_runtime_artifact_storage() -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact("analysis_plan", {"enabled": True})

    assert runtime.get_artifact("analysis_plan") == {"enabled": True}


def test_analysis_plan_and_variable_map_can_be_created() -> None:
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
                    "measurement_level": "binary",
                    "role": "dependent",
                }
            }
        }
    )

    assert analysis_plan.variables.dependent == ["outcome"]
    assert variable_map.variables["outcome"].role == "dependent"
