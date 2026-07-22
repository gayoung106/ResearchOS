"""One-call automatic rawdata analysis runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.auto.analysis_plan import AutoAnalysisPlanStep
from src.auto.pipeline import AutoRegressionPipelineBuildResult, build_auto_regression_orchestrator
from src.auto.rawdata_loader import AutoRawDataLoadingStep
from src.auto.variable_inference import AutoVariableInferenceStep
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import OrchestratorResult, ResearchOrchestrator
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import StepResult


@dataclass(slots=True)
class AutoRawDataAnalysisResult:
    success: bool
    context: ResearchContext
    runtime: PipelineRuntime
    setup_step_results: list[StepResult] = field(default_factory=list)
    pipeline_build_result: AutoRegressionPipelineBuildResult | None = None
    orchestrator: ResearchOrchestrator | None = None
    orchestrator_result: OrchestratorResult | None = None
    output_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed_stage: str | None = None


def _record_step_result(
    *,
    context: ResearchContext,
    result: StepResult,
    output_files: list[str],
    warnings: list[str],
) -> None:
    for path in result.output_files:
        context.add_generated_file(path)
        output_files.append(path)
    for warning in result.warnings:
        context.add_warning(f"{result.stage_name}: {warning}")
        warnings.append(warning)
    if result.success:
        context.mark_stage_completed(result.stage_name)


def _apply_plan_to_context(context: ResearchContext, analysis_plan: AnalysisPlan, variable_map: VariableMap) -> None:
    context.dependent_variables = list(analysis_plan.variables.dependent)
    context.independent_variables = list(analysis_plan.variables.independent)
    context.mediator_variables = list(analysis_plan.variables.mediators)
    context.moderator_variables = list(analysis_plan.variables.moderators)
    context.control_variables = list(analysis_plan.variables.controls)
    context.analysis_plan = analysis_plan.model_dump(mode="json")
    context.variable_map = variable_map.model_dump(mode="json")


def _write_auto_run_summary(
    *,
    working_directory: Path,
    result: AutoRawDataAnalysisResult,
) -> str:
    import pandas as pd

    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "auto_run_summary.xlsx"
    rows: list[dict[str, object]] = []
    for step_result in result.setup_step_results:
        rows.append(
            {
                "stage_name": step_result.stage_name,
                "success": step_result.success,
                "output_file_count": len(step_result.output_files),
                "warning_count": len(step_result.warnings),
            }
        )
    if result.pipeline_build_result is not None:
        rows.append(
            {
                "stage_name": "04_auto_pipeline_registration",
                "success": result.pipeline_build_result.success,
                "output_file_count": 0,
                "warning_count": len(result.pipeline_build_result.warnings),
            }
        )
    if result.orchestrator_result is not None:
        rows.append(
            {
                "stage_name": "05_auto_pipeline_execution",
                "success": result.orchestrator_result.success,
                "output_file_count": len(result.context.generated_files),
                "warning_count": len(result.orchestrator_result.warnings),
            }
        )
    pd.DataFrame(rows).to_excel(summary_path, index=False)
    return str(summary_path)


def run_auto_rawdata_analysis(
    working_directory: str | Path = ".",
    *,
    rawdata_dir: str | Path = "rawdata",
    source_file: str | Path | None = None,
    project_name: str = "auto_rawdata_analysis",
    enable_robustness: bool = False,
    run_analysis: bool = True,
    model_id: str = "main_model",
) -> AutoRawDataAnalysisResult:
    """Run the rawdata-only workflow through automatic planning and analysis."""
    root = Path(working_directory).expanduser().resolve()
    context = ResearchContext(project_name=project_name)
    runtime = PipelineRuntime()
    output_files: list[str] = []
    warnings: list[str] = []
    setup_results: list[StepResult] = []

    setup_steps = [
        AutoRawDataLoadingStep(runtime, rawdata_dir=rawdata_dir, source_file=source_file),
        AutoVariableInferenceStep(runtime),
        AutoAnalysisPlanStep(runtime, enable_robustness=enable_robustness),
    ]
    for step in setup_steps:
        try:
            step_result = step.run(context, root)
        except Exception as error:  # noqa: BLE001 - convert setup failures into a structured auto-run result.
            step_result = StepResult(
                stage_name=step.name,
                success=False,
                warnings=[str(error)],
                metadata={"error_message": str(error)},
            )
        setup_results.append(step_result)
        _record_step_result(
            context=context,
            result=step_result,
            output_files=output_files,
            warnings=warnings,
        )
        if not step_result.success:
            result = AutoRawDataAnalysisResult(
                success=False,
                context=context,
                runtime=runtime,
                setup_step_results=setup_results,
                output_files=output_files,
                warnings=warnings,
                failed_stage=step_result.stage_name,
            )
            summary_path = _write_auto_run_summary(working_directory=root, result=result)
            result.output_files.append(summary_path)
            context.add_generated_file(summary_path)
            return result

    analysis_plan = runtime.get_artifact("auto_analysis_plan")
    variable_map = runtime.get_artifact("auto_variable_map")
    if isinstance(analysis_plan, AnalysisPlan) and isinstance(variable_map, VariableMap):
        _apply_plan_to_context(context, analysis_plan, variable_map)

    orchestrator, build_result = build_auto_regression_orchestrator(
        context=context,
        runtime=runtime,
        working_directory=root,
        model_id=model_id,
    )
    warnings.extend(build_result.warnings)
    for warning in build_result.warnings:
        context.add_warning(f"04_auto_pipeline_registration: {warning}")

    orchestrator_result: OrchestratorResult | None = None
    success = build_result.success
    failed_stage = None if success else "04_auto_pipeline_registration"
    if build_result.success and run_analysis:
        orchestrator_result = orchestrator.run()
        success = orchestrator_result.success
        failed_stage = orchestrator_result.failed_stage
        warnings.extend(orchestrator_result.warnings)
        output_files = list(dict.fromkeys(output_files + context.generated_files))

    result = AutoRawDataAnalysisResult(
        success=success,
        context=context,
        runtime=runtime,
        setup_step_results=setup_results,
        pipeline_build_result=build_result,
        orchestrator=orchestrator,
        orchestrator_result=orchestrator_result,
        output_files=list(dict.fromkeys(output_files)),
        warnings=warnings,
        failed_stage=failed_stage,
    )
    summary_path = _write_auto_run_summary(working_directory=root, result=result)
    result.output_files.append(summary_path)
    context.add_generated_file(summary_path)
    return result
