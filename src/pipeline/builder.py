"""전체 연구 파이프라인을 구성하는 빌더."""

from __future__ import annotations

from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.analysis_steps import (
    MissingnessStep,
    OutlierStep,
    PreprocessingPlanningStep,
    ScaleReliabilityStep,
    VariableDetectionStep,
)
from src.pipeline.context import ResearchContext
from src.pipeline.correlation_step import CorrelationAnalysisStep
from src.pipeline.descriptive_step import DescriptiveStatisticsStep
from src.pipeline.io_steps import (
    DataLoadingStep,
    EvidenceResolutionStep,
)
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import (
    register_regression_pipeline,
)
from src.pipeline.runtime import PipelineRuntime


def build_default_pipeline(
    *,
    context: ResearchContext,
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
    working_directory: str | Path = ".",
    source_file: str | Path | None = None,
    mahalanobis_variables: list[str] | None = None,
) -> tuple[ResearchOrchestrator, PipelineRuntime]:
    """
    현재 구현된 기본 연구 파이프라인 전체를 구성한다.

    실행 순서:
    01. 데이터 로딩
    02. 변수 측정수준 자동 탐지
    02. 외부 근거 통합
    03. 전처리 계획
    04. 척도·신뢰도
    05. 결측치 진단
    06. 이상치 진단
    07. 기술통계
    08. 상관분석
    09. 회귀분석
    10. 회귀진단
    11. 강건성 분석
    12. 고급 강건성 분석
    13. 효과크기
    14. 회귀 보고서
    15. 회귀 시각화
    16. 연구 품질 감사
    """
    runtime = PipelineRuntime()

    orchestrator = ResearchOrchestrator(
        context=context,
        working_directory=working_directory,
    )

    orchestrator.register(
        DataLoadingStep(
            runtime,
            source_file=source_file,
            order=10,
        )
    )

    orchestrator.register(VariableDetectionStep(runtime))

    orchestrator.register(
        EvidenceResolutionStep(
            runtime,
            variable_map,
            order=25,
        )
    )

    orchestrator.register(
        PreprocessingPlanningStep(
            runtime,
            analysis_plan,
            variable_map,
        )
    )

    orchestrator.register(
        ScaleReliabilityStep(
            runtime,
            variable_map,
        )
    )

    orchestrator.register(MissingnessStep(runtime))

    orchestrator.register(
        OutlierStep(
            runtime,
            mahalanobis_variables=mahalanobis_variables,
        )
    )

    orchestrator.register(DescriptiveStatisticsStep(runtime))

    correlation_variables = list(
        dict.fromkeys(
            analysis_plan.variables.dependent
            + analysis_plan.variables.independent
            + analysis_plan.variables.mediators
            + analysis_plan.variables.moderators
            + analysis_plan.variables.controls
        )
    )

    orchestrator.register(
        CorrelationAnalysisStep(
            runtime,
            correlation_variables,
            method="pearson",
            p_adjust_method="holm",
        )
    )

    regression_registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
    )

    runtime.set_artifact(
        "regression_registration",
        regression_registration,
    )

    return orchestrator, runtime
