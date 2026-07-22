from pathlib import Path

import pandas as pd

from src.auto.analysis_plan import (
    AutoAnalysisPlanStep,
    auto_analysis_plan_decisions_to_dataframe,
    auto_analysis_plan_summary_to_dataframe,
    build_auto_analysis_plan,
)
from src.auto.variable_inference import build_auto_variable_map
from src.common.config_loader import load_analysis_plan, load_variable_map
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import register_regression_pipeline
from src.pipeline.runtime import PipelineRuntime


def _basic_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2],
            "age": [21, 35, 44, 51, 39, 28],
            "gender": [0, 1, 1, 0, 1, 0],
        }
    )


def _panel_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "wave": [1, 2, 1, 2, 1, 2, 1, 2],
            "outcome_score": [2.0, 2.4, 2.7, 3.1, 3.4, 3.8, 4.1, 4.4],
            "age": [21, 22, 35, 36, 44, 45, 51, 52],
            "gender": [0, 0, 1, 1, 1, 1, 0, 0],
        }
    )


def _weighted_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2],
            "sample_weight": [1.1, 0.9, 1.0, 1.2, 0.8, 1.3],
            "age": [21, 35, 44, 51, 39, 28],
            "gender": [0, 1, 1, 0, 1, 0],
        }
    )


def test_build_auto_analysis_plan_enables_basic_regression() -> None:
    inference = build_auto_variable_map(_basic_dataframe())

    result = build_auto_analysis_plan(inference.variable_map)
    plan = result.analysis_plan

    assert plan.variables.dependent == ["outcome_score"]
    assert plan.variables.independent == ["age", "gender"]
    assert plan.analyses.regression.enabled is True
    assert plan.analyses.regression.options == {}
    assert plan.analyses.robustness.enabled is False
    assert result.warnings == []
    assert "regression_enabled" in set(auto_analysis_plan_summary_to_dataframe(result)["item"])
    assert "regression_estimator" in set(auto_analysis_plan_decisions_to_dataframe(result.decisions)["item"])


def test_build_auto_analysis_plan_selects_panel_fixed_effects_when_entity_and_time_exist() -> None:
    inference = build_auto_variable_map(_panel_dataframe())

    result = build_auto_analysis_plan(inference.variable_map)
    plan = result.analysis_plan

    assert plan.analyses.regression.enabled is True
    assert plan.analyses.regression.options["estimator"] == "panel_fe"
    assert plan.analyses.panel.enabled is True
    assert plan.analyses.panel.options == {"entity_variable": "person_id", "time_variable": "wave"}

    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="auto panel plan"),
        working_directory=Path("."),
    )
    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=inference.variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_fixed_effects"


def test_build_auto_analysis_plan_selects_wls_when_weight_exists() -> None:
    inference = build_auto_variable_map(_weighted_dataframe())

    result = build_auto_analysis_plan(inference.variable_map, enable_robustness=True)
    plan = result.analysis_plan

    assert plan.analyses.regression.options == {
        "estimator": "wls",
        "weight_variable": "sample_weight",
    }
    assert plan.variables.weights == ["sample_weight"]
    assert plan.analyses.robustness.enabled is True


def test_auto_analysis_plan_step_populates_runtime_and_outputs(tmp_path: Path) -> None:
    inference = build_auto_variable_map(_basic_dataframe())
    runtime = PipelineRuntime(dataframe=_basic_dataframe())
    runtime.set_artifact("auto_variable_map", inference.variable_map)

    step_result = AutoAnalysisPlanStep(runtime).run(
        ResearchContext(project_name="auto analysis plan"),
        tmp_path,
    )

    assert step_result.success is True
    assert runtime.get_artifact("auto_analysis_plan").analyses.regression.enabled is True
    assert runtime.get_artifact("auto_analysis_plan_result").decisions
    assert step_result.metadata["dependent_variable"] == "outcome_score"
    assert {Path(path).name for path in step_result.output_files} == {
        "analysis_plan_summary.xlsx",
        "analysis_plan_decisions.xlsx",
        "auto_analysis_plan.yaml",
        "auto_variable_map.yaml",
    }
    loaded_plan = load_analysis_plan(step_result.metadata["analysis_plan_path"])
    loaded_variable_map = load_variable_map(step_result.metadata["variable_map_path"])
    assert loaded_plan.variables.dependent == ["outcome_score"]
    assert loaded_variable_map.variables["outcome_score"].role == "dependent"


def test_auto_analysis_plan_step_requires_variable_map_artifact(tmp_path: Path) -> None:
    step_result = AutoAnalysisPlanStep(PipelineRuntime()).run(
        ResearchContext(project_name="missing auto map"),
        tmp_path,
    )

    assert step_result.success is False
    assert step_result.warnings == ["auto_variable_map artifact is required before auto analysis planning."]
