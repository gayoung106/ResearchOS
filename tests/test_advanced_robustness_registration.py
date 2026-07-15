"""고급 강건성 파이프라인 등록 테스트."""

from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from tests.support.assertions import (
    assert_step_order,
    assert_steps_not_registered,
    assert_steps_registered,
)
from tests.support.builders import build_regression_pipeline


def test_advanced_robustness_is_registered_by_default(
    tmp_path: Path,
    ols_with_robustness_analysis_plan: AnalysisPlan,
    continuous_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=continuous_variable_map,
    )

    assert registration.advanced_robustness_registered is True

    assert_steps_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
        "13_effect_size_analysis",
    )

    assert_step_order(
        orchestrator,
        before="11_robustness_analysis",
        after="12_advanced_robustness",
    )
    assert_step_order(
        orchestrator,
        before="12_advanced_robustness",
        after="13_effect_size_analysis",
    )


def test_advanced_robustness_can_be_disabled(
    tmp_path: Path,
    advanced_robustness_disabled_analysis_plan: AnalysisPlan,
    continuous_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=advanced_robustness_disabled_analysis_plan,
        variable_map=continuous_variable_map,
    )

    assert registration.advanced_robustness_registered is False

    assert_steps_registered(
        orchestrator,
        "11_robustness_analysis",
        "13_effect_size_analysis",
    )
    assert_steps_not_registered(
        orchestrator,
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="11_robustness_analysis",
        after="13_effect_size_analysis",
    )


def test_cluster_variable_comes_from_analysis_variables(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["country"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
                "robustness": {
                    "enabled": True,
                },
            },
        }
    )

    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {
                    "role": "dependent",
                    "measurement_level": "continuous",
                },
                "x": {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
                "country": {
                    "role": "cluster",
                    "measurement_level": "nominal",
                },
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
    )

    step = orchestrator.registry.get("12_advanced_robustness")

    assert registration.advanced_robustness_registered is True
    assert_steps_registered(
        orchestrator,
        "12_advanced_robustness",
    )
    assert step.cluster_variable == "country"


def test_binary_model_has_no_advanced_robustness(
    tmp_path: Path,
    ols_with_robustness_analysis_plan: AnalysisPlan,
    binary_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=binary_variable_map,
    )

    assert registration.advanced_robustness_registered is False

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "13_effect_size_analysis",
    )
    assert_steps_not_registered(
        orchestrator,
        "10_regression_diagnostics",
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="09_regression_analysis",
        after="13_effect_size_analysis",
    )
