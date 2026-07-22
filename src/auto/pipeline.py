"""Automatic pipeline registration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.auto.multi_outcome import AutoMultiOutcomeAnalysisPlanResult
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import OrchestratorResult, ResearchOrchestrator
from src.pipeline.regression_builder import RegressionRegistration, register_regression_pipeline
from src.pipeline.runtime import PipelineRuntime


@dataclass(slots=True)
class AutoMultiOutcomePipelineBuildResult:
    success: bool
    model_results: dict[str, AutoRegressionPipelineBuildResult] = field(default_factory=dict)
    orchestrators: dict[str, ResearchOrchestrator] = field(default_factory=dict)
    runtimes: dict[str, PipelineRuntime] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutoMultiOutcomePipelineRunResult:
    success: bool
    model_run_results: dict[str, OrchestratorResult] = field(default_factory=dict)
    completed_models: list[str] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)
    skipped_models: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutoRegressionPipelineBuildResult:
    success: bool
    registered_step_names: list[str] = field(default_factory=list)
    registration: RegressionRegistration | None = None
    warnings: list[str] = field(default_factory=list)


def _runtime_artifact(runtime: PipelineRuntime, key: str) -> object | None:
    try:
        return runtime.get_artifact(key)
    except KeyError:
        return None


def register_auto_regression_pipeline(
    *,
    orchestrator: ResearchOrchestrator,
    runtime: PipelineRuntime,
    model_id: str = "main_model",
    analysis_plan_artifact_key: str = "auto_analysis_plan",
    variable_map_artifact_key: str = "auto_variable_map",
) -> AutoRegressionPipelineBuildResult:
    """Register the regression pipeline from auto-generated planning artifacts."""
    analysis_plan = _runtime_artifact(runtime, analysis_plan_artifact_key)
    variable_map = _runtime_artifact(runtime, variable_map_artifact_key)
    warnings: list[str] = []

    if not isinstance(analysis_plan, AnalysisPlan):
        warnings.append(f"{analysis_plan_artifact_key} artifact is required before pipeline registration.")
    if not isinstance(variable_map, VariableMap):
        warnings.append(f"{variable_map_artifact_key} artifact is required before pipeline registration.")
    if warnings:
        result = AutoRegressionPipelineBuildResult(success=False, warnings=warnings)
        runtime.set_artifact("auto_regression_pipeline_build_result", result)
        return result

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        model_id=model_id,
    )
    registered_step_names = orchestrator.registry.names()
    success = registration.registered
    warnings.extend(registration.warnings)
    result = AutoRegressionPipelineBuildResult(
        success=success,
        registered_step_names=registered_step_names,
        registration=registration,
        warnings=warnings,
    )
    runtime.set_artifact("auto_regression_pipeline_build_result", result)
    runtime.set_artifact("regression_registration", registration)
    return result


def build_auto_regression_orchestrator(
    *,
    context: ResearchContext,
    runtime: PipelineRuntime,
    working_directory: str | Path = ".",
    model_id: str = "main_model",
) -> tuple[ResearchOrchestrator, AutoRegressionPipelineBuildResult]:
    """Create an orchestrator and register the auto-planned regression pipeline."""
    orchestrator = ResearchOrchestrator(
        context=context,
        working_directory=working_directory,
    )
    result = register_auto_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
        model_id=model_id,
    )
    return orchestrator, result


def build_auto_multi_outcome_regression_orchestrators(
    *,
    context: ResearchContext,
    runtime: PipelineRuntime,
    working_directory: str | Path = ".",
    multi_outcome_result_artifact_key: str = "auto_multi_outcome_plan_result",
) -> AutoMultiOutcomePipelineBuildResult:
    """Create one registered regression orchestrator for each auto outcome plan."""
    multi_outcome_result = _runtime_artifact(runtime, multi_outcome_result_artifact_key)
    if not isinstance(multi_outcome_result, AutoMultiOutcomeAnalysisPlanResult):
        result = AutoMultiOutcomePipelineBuildResult(
            success=False,
            warnings=[f"{multi_outcome_result_artifact_key} artifact is required before multi-outcome registration."],
        )
        runtime.set_artifact("auto_multi_outcome_pipeline_build_result", result)
        return result

    model_results: dict[str, AutoRegressionPipelineBuildResult] = {}
    orchestrators: dict[str, ResearchOrchestrator] = {}
    runtimes: dict[str, PipelineRuntime] = {}
    warnings: list[str] = []

    for outcome_plan in multi_outcome_result.outcome_plans:
        model_runtime = PipelineRuntime(
            dataframe=runtime.dataframe,
            variable_metadata=runtime.variable_metadata,
            detections=list(runtime.detections),
            resolved_levels=list(runtime.resolved_levels),
        )
        model_runtime.set_artifact("auto_variable_map", outcome_plan.variable_map)
        model_runtime.set_artifact("auto_analysis_plan", outcome_plan.analysis_plan)
        model_context = ResearchContext(
            project_name=f"{context.project_name}:{outcome_plan.model_id}",
            research_topic=context.research_topic,
            research_questions=list(context.research_questions),
            hypotheses=list(context.hypotheses),
            raw_data_files=list(context.raw_data_files),
            questionnaire_files=list(context.questionnaire_files),
            codebook_files=list(context.codebook_files),
        )
        model_orchestrator = ResearchOrchestrator(
            context=model_context,
            working_directory=Path(working_directory) / "result" / "multi_outcome_runs" / outcome_plan.model_id,
        )
        model_result = register_auto_regression_pipeline(
            orchestrator=model_orchestrator,
            runtime=model_runtime,
            model_id=outcome_plan.model_id,
        )
        model_results[outcome_plan.model_id] = model_result
        orchestrators[outcome_plan.model_id] = model_orchestrator
        runtimes[outcome_plan.model_id] = model_runtime
        warnings.extend(model_result.warnings)

    result = AutoMultiOutcomePipelineBuildResult(
        success=all(item.success for item in model_results.values()) and bool(model_results),
        model_results=model_results,
        orchestrators=orchestrators,
        runtimes=runtimes,
        warnings=list(dict.fromkeys(warnings)),
    )
    runtime.set_artifact("auto_multi_outcome_pipeline_build_result", result)
    return result


def run_auto_multi_outcome_regression_orchestrators(
    build_result: AutoMultiOutcomePipelineBuildResult,
    *,
    runtime: PipelineRuntime | None = None,
    start_from: str | None = None,
    end_at: str | None = None,
    rerun_completed: bool = False,
) -> AutoMultiOutcomePipelineRunResult:
    """Run every registered multi-outcome regression orchestrator."""
    model_run_results: dict[str, OrchestratorResult] = {}
    completed_models: list[str] = []
    failed_models: list[str] = []
    skipped_models: list[str] = []
    warnings = list(build_result.warnings)

    if not build_result.orchestrators:
        result = AutoMultiOutcomePipelineRunResult(
            success=False,
            warnings=warnings + ["No multi-outcome orchestrators are available to run."],
        )
        if runtime is not None:
            runtime.set_artifact("auto_multi_outcome_pipeline_run_result", result)
        return result

    for model_id, orchestrator in build_result.orchestrators.items():
        model_build_result = build_result.model_results.get(model_id)
        if model_build_result is not None and not model_build_result.success:
            skipped_models.append(model_id)
            warnings.append(f"{model_id} was skipped because pipeline registration failed.")
            continue

        run_result = orchestrator.run(
            start_from=start_from,
            end_at=end_at,
            rerun_completed=rerun_completed,
        )
        model_run_results[model_id] = run_result
        warnings.extend(f"{model_id}: {warning}" for warning in run_result.warnings)
        if run_result.success:
            completed_models.append(model_id)
        else:
            failed_models.append(model_id)

    result = AutoMultiOutcomePipelineRunResult(
        success=bool(model_run_results) and not failed_models and not skipped_models,
        model_run_results=model_run_results,
        completed_models=completed_models,
        failed_models=failed_models,
        skipped_models=skipped_models,
        warnings=list(dict.fromkeys(warnings)),
    )
    if runtime is not None:
        runtime.set_artifact("auto_multi_outcome_pipeline_run_result", result)
    return result
