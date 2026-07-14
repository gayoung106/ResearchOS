"""기존 분석 모듈을 Research Orchestrator에 연결하는 단계 어댑터."""

from __future__ import annotations

from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.preprocess.detector import (
    detect_dataframe_variables,
    detections_to_dataframe,
)
from src.preprocess.missingness import (
    build_missingness_report,
    recommendations_to_dataframe,
)
from src.preprocess.outliers import (
    build_outlier_report,
    mahalanobis_distances_to_dataframe,
    univariate_results_to_dataframe,
)
from src.preprocess.planner import (
    plan_preprocessing,
    preprocessing_plan_to_dataframe,
)
from src.preprocess.scales import (
    build_all_scales,
    collect_scale_definitions,
    scale_records_to_dataframe,
)
from src.statistics.reliability import (
    reliability_result_to_dataframe,
    run_reliability_analysis,
)


class RuntimeStep(PipelineStep):
    """PipelineRuntime을 공유하는 단계 기반 클래스."""

    def __init__(
        self,
        *,
        runtime: PipelineRuntime,
        name: str,
        order: int,
        required: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            order=order,
            required=required,
        )
        self.runtime = runtime


class VariableDetectionStep(RuntimeStep):
    """변수 측정수준 후보 탐지 단계."""

    def __init__(self, runtime: PipelineRuntime) -> None:
        super().__init__(
            runtime=runtime,
            name="02_variable_detection",
            order=20,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        detections = detect_dataframe_variables(
            dataframe,
            variable_metadata=self.runtime.variable_metadata,
        )
        self.runtime.detections = detections

        output_dir = working_directory / "result" / "02_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "variable_detections.xlsx"

        detections_to_dataframe(detections).to_excel(
            output_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(output_path)],
            metadata={"variable_count": len(detections)},
        )


class PreprocessingPlanningStep(RuntimeStep):
    """전처리 계획 생성 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        analysis_plan: AnalysisPlan,
        variable_map: VariableMap,
    ) -> None:
        super().__init__(
            runtime=runtime,
            name="03_preprocessing_plan",
            order=30,
        )
        self.analysis_plan = analysis_plan
        self.variable_map = variable_map

    def should_run(self, context: ResearchContext) -> bool:
        return bool(self.runtime.resolved_levels)

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        plan = plan_preprocessing(
            self.analysis_plan,
            self.variable_map,
            self.runtime.resolved_levels,
        )
        self.runtime.preprocessing_plan = plan

        output_dir = working_directory / "result" / "03_preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "preprocessing_plan.xlsx"

        preprocessing_plan_to_dataframe(plan).to_excel(
            output_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(output_path)],
            warnings=plan.warnings,
            metadata={
                "action_count": len(plan.actions),
                "blocked_variables": plan.blocked_variables,
            },
        )


class ScaleReliabilityStep(RuntimeStep):
    """척도 생성 및 신뢰도 분석 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        variable_map: VariableMap,
    ) -> None:
        super().__init__(
            runtime=runtime,
            name="04_scale_reliability",
            order=40,
            required=False,
        )
        self.variable_map = variable_map

    def should_run(self, context: ResearchContext) -> bool:
        return any(definition.scale_name for definition in self.variable_map.variables.values())

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        definitions = collect_scale_definitions(self.variable_map)
        self.runtime.scale_definitions = definitions

        output, records = build_all_scales(
            dataframe,
            definitions,
        )
        self.runtime.dataframe = output
        self.runtime.scale_records = records

        output_dir = working_directory / "result" / "06_measurement"
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: list[str] = []

        records_path = output_dir / "scale_build_records.xlsx"
        scale_records_to_dataframe(records).to_excel(
            records_path,
            index=False,
        )
        generated_files.append(str(records_path))

        for definition in definitions:
            result, item_table = run_reliability_analysis(
                output[definition.items],
                scale_name=definition.scale_name,
            )
            self.runtime.reliability_results[definition.scale_name] = result

            summary_path = output_dir / f"{definition.scale_name}_reliability.xlsx"
            item_path = output_dir / f"{definition.scale_name}_item_statistics.xlsx"

            reliability_result_to_dataframe(result).to_excel(
                summary_path,
                index=False,
            )
            item_table.to_excel(item_path, index=False)

            generated_files.extend([str(summary_path), str(item_path)])

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=generated_files,
            metadata={"scale_count": len(definitions)},
        )


class MissingnessStep(RuntimeStep):
    """결측치 진단 단계."""

    def __init__(self, runtime: PipelineRuntime) -> None:
        super().__init__(
            runtime=runtime,
            name="05_missingness",
            order=50,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        report = build_missingness_report(dataframe)
        self.runtime.missingness_report = report

        output_dir = working_directory / "result" / "05_missing"
        output_dir.mkdir(parents=True, exist_ok=True)

        variable_path = output_dir / "variable_missingness.xlsx"
        case_path = output_dir / "case_missingness.xlsx"
        pattern_path = output_dir / "missingness_patterns.xlsx"
        recommendation_path = output_dir / "missingness_recommendations.xlsx"

        report.variable_summary.to_excel(variable_path, index=False)
        report.case_summary.to_excel(case_path, index=False)
        report.pattern_summary.to_excel(pattern_path, index=False)
        recommendations_to_dataframe(report.recommendations).to_excel(
            recommendation_path, index=False
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(variable_path),
                str(case_path),
                str(pattern_path),
                str(recommendation_path),
            ],
            warnings=report.warnings,
        )


class OutlierStep(RuntimeStep):
    """이상치 진단 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        mahalanobis_variables: list[str] | None = None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            name="06_outliers",
            order=60,
        )
        self.mahalanobis_variables = mahalanobis_variables

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        report = build_outlier_report(
            dataframe,
            mahalanobis_variables=self.mahalanobis_variables,
        )
        self.runtime.outlier_report = report

        output_dir = working_directory / "result" / "06_outlier"
        output_dir.mkdir(parents=True, exist_ok=True)

        univariate_path = output_dir / "univariate_outliers.xlsx"
        univariate_results_to_dataframe(report.univariate_results).to_excel(
            univariate_path, index=False
        )

        generated_files = [str(univariate_path)]

        if report.mahalanobis_result is not None:
            mahalanobis_path = output_dir / "mahalanobis_distances.xlsx"
            mahalanobis_distances_to_dataframe(report.mahalanobis_result).to_excel(
                mahalanobis_path, index=False
            )
            generated_files.append(str(mahalanobis_path))

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=generated_files,
            warnings=report.warnings,
        )
