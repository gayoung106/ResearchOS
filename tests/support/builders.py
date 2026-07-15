"""테스트용 파이프라인 객체 생성 helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import register_regression_pipeline
from src.pipeline.runtime import PipelineRuntime


def build_orchestrator(
    tmp_path: Path,
    *,
    project_name: str = "테스트",
) -> tuple[ResearchOrchestrator, PipelineRuntime]:
    """테스트용 orchestrator와 runtime을 생성한다."""
    runtime = PipelineRuntime()
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(
            project_name=project_name,
        ),
        working_directory=tmp_path,
    )
    return orchestrator, runtime


def build_regression_pipeline(
    tmp_path: Path,
    *,
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
    project_name: str = "테스트",
) -> tuple[
    ResearchOrchestrator,
    PipelineRuntime,
    Any,
]:
    """회귀 파이프라인을 등록한 테스트 객체를 반환한다."""
    orchestrator, runtime = build_orchestrator(
        tmp_path,
        project_name=project_name,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
    )

    return orchestrator, runtime, registration


def make_empty_analysis_plan() -> AnalysisPlan:
    """분석 기능이 비활성화된 빈 분석 계획을 생성한다."""
    return AnalysisPlan.model_validate({})


def make_empty_variable_map() -> VariableMap:
    """변수가 없는 빈 variable map을 생성한다."""
    return VariableMap.model_validate(
        {
            "variables": {},
        }
    )


def make_continuous_variable_map(
    *,
    dependent: str = "y",
    independent: str = "x",
) -> VariableMap:
    """연속형 종속변수와 독립변수로 구성된 variable map을 생성한다."""
    return VariableMap.model_validate(
        {
            "variables": {
                dependent: {
                    "role": "dependent",
                    "measurement_level": "continuous",
                },
                independent: {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
            }
        }
    )


def make_binary_variable_map(
    *,
    dependent: str = "y",
    independent: str = "x",
) -> VariableMap:
    """이항형 종속변수와 연속형 독립변수 variable map을 생성한다."""
    return VariableMap.model_validate(
        {
            "variables": {
                dependent: {
                    "role": "dependent",
                    "measurement_level": "binary",
                },
                independent: {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
            }
        }
    )


def make_ordinal_variable_map(
    *,
    dependent: str = "y",
    independent: str = "x",
) -> VariableMap:
    """서열형 종속변수와 연속형 독립변수 variable map을 생성한다."""
    return VariableMap.model_validate(
        {
            "variables": {
                dependent: {
                    "role": "dependent",
                    "measurement_level": "ordinal",
                },
                independent: {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
            }
        }
    )
