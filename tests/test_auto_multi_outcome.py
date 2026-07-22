from pathlib import Path

import pandas as pd

from src.auto.multi_outcome import (
    AutoMultiOutcomeAnalysisPlanStep,
    auto_multi_outcome_candidates_to_dataframe,
    auto_multi_outcome_plans_to_dataframe,
    build_auto_multi_outcome_analysis_plans,
    infer_auto_outcome_candidates,
)
from src.auto.pipeline import build_auto_multi_outcome_regression_orchestrators
from src.auto.variable_inference import build_auto_variable_map
from src.common.config_loader import load_analysis_plan, load_variable_map
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime


def _multi_outcome_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame(
        {
            "baseline_score": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
            "satisfaction": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "performance": [10.0, 11.2, 12.1, 13.0, 14.5, 15.1, 15.8, 16.2],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "site": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    metadata = pd.DataFrame(
        {
            "variable_name": ["baseline_score", "satisfaction", "performance", "age", "site"],
            "variable_label": [
                "baseline predictor score",
                "job satisfaction outcome score",
                "performance outcome score",
                "age",
                "school site",
            ],
        }
    )
    return data, metadata


def _variable_map():
    data, metadata = _multi_outcome_data()
    return build_auto_variable_map(data, variable_metadata=metadata).variable_map


def test_infer_auto_outcome_candidates_finds_multiple_labeled_outcomes() -> None:
    variable_map = _variable_map()

    candidates = infer_auto_outcome_candidates(variable_map)

    assert [candidate.variable_name for candidate in candidates[:2]] == ["satisfaction", "performance"]
    assert candidates[0].score >= candidates[1].score
    assert "variable_name" in auto_multi_outcome_candidates_to_dataframe(candidates).columns


def test_build_auto_multi_outcome_analysis_plans_creates_one_plan_per_outcome() -> None:
    variable_map = _variable_map()

    result = build_auto_multi_outcome_analysis_plans(
        variable_map,
        max_outcomes=2,
        model_id_prefix="auto_model",
    )

    assert [item.model_id for item in result.outcome_plans] == ["auto_model_1", "auto_model_2"]
    assert [item.dependent_variable for item in result.outcome_plans] == ["satisfaction", "performance"]
    assert result.outcome_plans[0].analysis_plan.variables.dependent == ["satisfaction"]
    assert "performance" in result.outcome_plans[0].analysis_plan.variables.independent
    assert result.outcome_plans[1].analysis_plan.variables.dependent == ["performance"]
    assert "satisfaction" in result.outcome_plans[1].analysis_plan.variables.independent
    assert auto_multi_outcome_plans_to_dataframe(result).shape[0] == 2


def test_auto_multi_outcome_analysis_plan_step_outputs_yaml_files(tmp_path: Path) -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact("auto_variable_map", _variable_map())

    step_result = AutoMultiOutcomeAnalysisPlanStep(
        runtime,
        max_outcomes=2,
        model_id_prefix="auto_model",
    ).run(ResearchContext(project_name="multi outcome plan"), tmp_path)

    assert step_result.success is True
    assert step_result.metadata["outcome_plan_count"] == 2
    assert runtime.get_artifact("auto_multi_outcome_plan_result").outcome_plans
    assert {Path(path).name for path in step_result.output_files} >= {
        "outcome_candidates.xlsx",
        "outcome_analysis_plans.xlsx",
        "analysis_plan.yaml",
        "variable_map.yaml",
    }
    analysis_plan_path = tmp_path / "result" / "03_auto_plan" / "multi_outcome" / "auto_model_1" / "analysis_plan.yaml"
    variable_map_path = tmp_path / "result" / "03_auto_plan" / "multi_outcome" / "auto_model_1" / "variable_map.yaml"
    assert load_analysis_plan(analysis_plan_path).variables.dependent == ["satisfaction"]
    assert load_variable_map(variable_map_path).variables["satisfaction"].role == "dependent"


def test_build_auto_multi_outcome_regression_orchestrators_registers_each_model(tmp_path: Path) -> None:
    data, _ = _multi_outcome_data()
    runtime = PipelineRuntime(dataframe=data)
    plan_result = build_auto_multi_outcome_analysis_plans(
        _variable_map(),
        max_outcomes=2,
        model_id_prefix="auto_model",
    )
    runtime.set_artifact("auto_multi_outcome_plan_result", plan_result)

    result = build_auto_multi_outcome_regression_orchestrators(
        context=ResearchContext(project_name="multi outcome registration"),
        runtime=runtime,
        working_directory=tmp_path,
    )

    assert result.success is True
    assert sorted(result.model_results) == ["auto_model_1", "auto_model_2"]
    assert result.model_results["auto_model_1"].registration is not None
    assert result.model_results["auto_model_1"].registration.dependent_variable == "satisfaction"
    assert result.model_results["auto_model_2"].registration is not None
    assert result.model_results["auto_model_2"].registration.dependent_variable == "performance"
    assert result.orchestrators["auto_model_1"].registry.names() == [
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    ]
    assert runtime.get_artifact("auto_multi_outcome_pipeline_build_result").success is True


def test_build_auto_multi_outcome_regression_orchestrators_requires_plan_artifact(tmp_path: Path) -> None:
    result = build_auto_multi_outcome_regression_orchestrators(
        context=ResearchContext(project_name="missing multi outcome registration"),
        runtime=PipelineRuntime(),
        working_directory=tmp_path,
    )

    assert result.success is False
    assert result.warnings == [
        "auto_multi_outcome_plan_result artifact is required before multi-outcome registration."
    ]
