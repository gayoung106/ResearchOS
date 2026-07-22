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


def _format_bool(value: bool) -> str:
    return "??" if value else "??"


def _artifact_or_none(runtime: PipelineRuntime, key: str) -> object | None:
    try:
        return runtime.get_artifact(key)
    except KeyError:
        return None


def _write_auto_run_markdown(
    *,
    working_directory: Path,
    result: AutoRawDataAnalysisResult,
) -> str:
    output_dir = working_directory / "result" / "00_auto_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "auto_run_report.md"

    rawdata = _artifact_or_none(result.runtime, "auto_rawdata_load_result")
    analysis_plan = _artifact_or_none(result.runtime, "auto_analysis_plan")
    variable_map = _artifact_or_none(result.runtime, "auto_variable_map")
    registration = result.pipeline_build_result.registration if result.pipeline_build_result else None

    lines = [
        "# ?? ?? ?? ??",
        "",
        f"- ?????: {result.context.project_name}",
        f"- ?? ??: {_format_bool(result.success)}",
        f"- ?? ??: {result.failed_stage or '-'}",
        f"- ?? ?: {len(result.warnings)}",
        "",
        "## ??? ??",
    ]
    if rawdata is not None:
        candidate = rawdata.selected_candidate
        lines.extend(
            [
                f"- ??: `{candidate.source_path}`",
                f"- Sheet: {candidate.sheet_name or '-'}",
                f"- ? ?: {candidate.row_count}",
                f"- ? ?: {candidate.column_count}",
                f"- ?? ?: {len(rawdata.candidates)}",
            ]
        )
    else:
        lines.append("- ???? ???? ?????.")

    lines.extend(["", "## ?? ?? ??"])
    if isinstance(variable_map, VariableMap):
        role_rows = [
            (name, definition.role, definition.measurement_level)
            for name, definition in variable_map.variables.items()
        ]
        lines.extend(["| ?? | ?? | ???? |", "| --- | --- | --- |"])
        lines.extend(f"| `{name}` | {role} | {level} |" for name, role, level in role_rows)
    else:
        lines.append("- ?? ??? ???? ?????.")

    lines.extend(["", "## ?? ????"])
    if isinstance(analysis_plan, AnalysisPlan):
        lines.extend(
            [
                f"- ????: {', '.join(analysis_plan.variables.dependent) or '-'}",
                f"- ????: {', '.join(analysis_plan.variables.independent) or '-'}",
                f"- ????: {', '.join(analysis_plan.variables.controls) or '-'}",
                f"- ????: {', '.join(analysis_plan.variables.clusters) or '-'}",
                f"- ??? ??: {', '.join(analysis_plan.variables.weights) or '-'}",
                f"- ???? ???: {_format_bool(analysis_plan.analyses.regression.enabled)}",
                f"- Panel ?? ???: {_format_bool(analysis_plan.analyses.panel.enabled)}",
                f"- ??? ?? ???: {_format_bool(analysis_plan.analyses.robustness.enabled)}",
                f"- ?? ??: `{analysis_plan.analyses.regression.options}`",
            ]
        )
    else:
        lines.append("- ????? ???? ?????.")

    lines.extend(["", "## ??? ??"])
    if registration is not None:
        lines.extend(
            [
                f"- ?? ID: {registration.model_id}",
                f"- ?? ??: {registration.model_type}",
                f"- ????: {registration.dependent_variable}",
                f"- ????: {', '.join(registration.independent_variables)}",
                f"- ?? ??: {_format_bool(registration.diagnostics_registered)}",
                f"- ???? ??: {_format_bool(registration.effect_size_registered)}",
                f"- ?? ??: {_format_bool(registration.reporting_registered)}",
                f"- ??? ??: {_format_bool(registration.visualization_registered)}",
                f"- Audit ??: {_format_bool(registration.audit_registered)}",
            ]
        )
    else:
        lines.append("- ?? ?????? ???? ?????.")

    lines.extend(["", "## ??? ??", "| ?? | ?? | ??? ? | ?? ? |", "| --- | --- | ---: | ---: |"])
    for step_result in result.setup_step_results:
        lines.append(
            f"| {step_result.stage_name} | {_format_bool(step_result.success)} | "
            f"{len(step_result.output_files)} | {len(step_result.warnings)} |"
        )
    if result.pipeline_build_result is not None:
        lines.append(
            "| 04_auto_pipeline_registration | "
            f"{_format_bool(result.pipeline_build_result.success)} | 0 | "
            f"{len(result.pipeline_build_result.warnings)} |"
        )
    if result.orchestrator_result is not None:
        lines.append(
            "| 05_auto_pipeline_execution | "
            f"{_format_bool(result.orchestrator_result.success)} | "
            f"{len(result.context.generated_files)} | {len(result.orchestrator_result.warnings)} |"
        )

    if result.warnings:
        lines.extend(["", "## ??"])
        lines.extend(f"- {warning}" for warning in result.warnings)

    lines.extend(["", "## ?? ???"])
    lines.extend(f"- `{output_file}`" for output_file in result.output_files)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


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
            report_path = _write_auto_run_markdown(working_directory=root, result=result)
            result.output_files.extend([summary_path, report_path])
            context.add_generated_file(summary_path)
            context.add_generated_file(report_path)
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
    report_path = _write_auto_run_markdown(working_directory=root, result=result)
    result.output_files.extend([summary_path, report_path])
    context.add_generated_file(summary_path)
    context.add_generated_file(report_path)
    return result
