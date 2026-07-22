"""Automatic pipeline registration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import RegressionRegistration, register_regression_pipeline
from src.pipeline.runtime import PipelineRuntime


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
