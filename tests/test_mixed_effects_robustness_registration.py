"""Random Intercept 강건성 단계 등록 테스트."""

from __future__ import annotations

from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.robustness_step import MixedEffectsRobustnessStep
from tests.support.builders import build_regression_pipeline


def make_plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["group"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {
                    "enabled": True,
                    "options": {"group_variable": "group", "optimizer": "lbfgs"},
                },
                "robustness": {
                    "enabled": True,
                    "options": {"mixed_optimizers": ["lbfgs", "bfgs"]},
                },
            },
        }
    )


def make_variable_map() -> VariableMap:
    return VariableMap.model_validate(
        {
            "variables": {
                "y": {"role": "dependent", "measurement_level": "continuous"},
                "x": {"role": "independent", "measurement_level": "continuous"},
                "group": {"role": "cluster", "measurement_level": "nominal"},
            }
        }
    )


def test_builder_registers_mixed_effects_robustness(tmp_path: Path) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=make_plan(),
        variable_map=make_variable_map(),
        project_name="Mixed robustness",
    )

    assert registration.robustness_registered is True
    assert registration.advanced_robustness_registered is True
    assert orchestrator.registry.names() == [
        "09_regression_analysis",
        "10_regression_diagnostics",
        "11_robustness_analysis",
        "12_advanced_robustness",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    ]
    step = orchestrator.registry.get("11_robustness_analysis")
    assert isinstance(step, MixedEffectsRobustnessStep)
    assert step.optimizers == ("lbfgs", "bfgs")
