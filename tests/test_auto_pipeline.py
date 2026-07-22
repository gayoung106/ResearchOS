import pandas as pd

from src.auto.analysis_plan import build_auto_analysis_plan
from src.auto.pipeline import build_auto_regression_orchestrator, register_auto_regression_pipeline
from src.auto.variable_inference import build_auto_variable_map
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.runtime import PipelineRuntime


def _analysis_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "gender": [0, 1, 1, 0, 1, 0, 1, 0],
        }
    )


def _runtime_with_auto_plan() -> PipelineRuntime:
    data = _analysis_dataframe()
    inference = build_auto_variable_map(data)
    plan_result = build_auto_analysis_plan(inference.variable_map)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("auto_variable_map", inference.variable_map)
    runtime.set_artifact("auto_analysis_plan", plan_result.analysis_plan)
    return runtime


def test_register_auto_regression_pipeline_uses_existing_builder() -> None:
    runtime = _runtime_with_auto_plan()
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="auto regression registration"),
        working_directory=".",
    )

    result = register_auto_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
    )

    assert result.success is True
    assert result.registration is not None
    assert result.registration.model_type == "ols"
    assert result.registration.diagnostics_registered is True
    assert result.registration.effect_size_registered is True
    assert result.registration.reporting_registered is True
    assert result.registration.visualization_registered is True
    assert result.registration.audit_registered is True
    assert result.registered_step_names == [
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    ]
    assert runtime.get_artifact("regression_registration").model_type == "ols"
    assert runtime.get_artifact("auto_regression_pipeline_build_result").success is True


def test_build_auto_regression_orchestrator_returns_registered_orchestrator() -> None:
    runtime = _runtime_with_auto_plan()

    orchestrator, result = build_auto_regression_orchestrator(
        context=ResearchContext(project_name="auto regression orchestrator"),
        runtime=runtime,
        working_directory=".",
    )

    assert result.success is True
    assert orchestrator.registry.names()[0] == "09_regression_analysis"
    assert result.registration is not None
    assert result.registration.dependent_variable == "outcome_score"


def test_register_auto_regression_pipeline_reports_missing_artifacts() -> None:
    runtime = PipelineRuntime()
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="missing auto artifacts"),
        working_directory=".",
    )

    result = register_auto_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
    )

    assert result.success is False
    assert result.registration is None
    assert orchestrator.registry.names() == []
    assert result.warnings == [
        "auto_analysis_plan artifact is required before pipeline registration.",
        "auto_variable_map artifact is required before pipeline registration.",
    ]
    assert runtime.get_artifact("auto_regression_pipeline_build_result").success is False
