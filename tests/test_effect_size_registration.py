"""효과크기 단계 자동 등록 테스트."""

from pathlib import Path

from src.common.config_models import (
    AnalysisPlan,
    VariableMap,
)
from tests.support.assertions import (
    assert_step_order,
    assert_steps_not_registered,
    assert_steps_registered,
)
from tests.support.builders import (
    build_regression_pipeline,
)


def test_ols_registers_effect_size_after_robustness(
    tmp_path: Path,
    continuous_variable_map: VariableMap,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
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

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=(continuous_variable_map),
    )

    assert registration.effect_size_registered is True

    assert_steps_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
        "13_effect_size_analysis",
        "14_regression_reporting",
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
    assert_step_order(
        orchestrator,
        before="13_effect_size_analysis",
        after="14_regression_reporting",
    )


def test_ols_without_robustness_still_registers_effect_size(
    tmp_path: Path,
    continuous_variable_map: VariableMap,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
                "robustness": {
                    "enabled": False,
                },
            },
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=(continuous_variable_map),
    )

    assert registration.effect_size_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
    )
    assert_steps_not_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="10_regression_diagnostics",
        after="13_effect_size_analysis",
    )
    assert_step_order(
        orchestrator,
        before="13_effect_size_analysis",
        after="14_regression_reporting",
    )


def test_binary_logit_registers_effect_size(
    tmp_path: Path,
    binary_variable_map: VariableMap,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
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

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=(binary_variable_map),
    )

    assert registration.effect_size_registered is True
    assert registration.diagnostics_registered is True
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
    )
    assert_steps_not_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="09_regression_analysis",
        after="10_regression_diagnostics",
    )
    assert_step_order(
        orchestrator,
        before="10_regression_diagnostics",
        after="13_effect_size_analysis",
    )
    assert_step_order(
        orchestrator,
        before="13_effect_size_analysis",
        after="14_regression_reporting",
    )


def test_ordered_logit_registers_effect_size(
    tmp_path: Path,
    ordinal_variable_map: VariableMap,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                }
            },
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=(ordinal_variable_map),
    )

    assert registration.model_type == "ordered_logit"
    assert registration.effect_size_registered is True
    assert registration.diagnostics_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
    )
    assert_steps_not_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="09_regression_analysis",
        after="10_regression_diagnostics",
    )
    assert_step_order(
        orchestrator,
        before="10_regression_diagnostics",
        after="13_effect_size_analysis",
    )
    assert_step_order(
        orchestrator,
        before="13_effect_size_analysis",
        after="14_regression_reporting",
    )


def test_unregistered_regression_has_no_effect_size(
    tmp_path: Path,
    empty_analysis_plan: AnalysisPlan,
    empty_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=(empty_analysis_plan),
        variable_map=(empty_variable_map),
    )

    assert registration.registered is False
    assert registration.effect_size_registered is False

    assert_steps_not_registered(
        orchestrator,
        "09_regression_analysis",
        "13_effect_size_analysis",
        "14_regression_reporting",
    )


def test_mixed_effects_registers_effect_size_after_diagnostics(
    tmp_path: Path,
    continuous_variable_map: VariableMap,
) -> None:
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                **{
                    name: definition.model_dump()
                    for name, definition in continuous_variable_map.variables.items()
                },
                "group": {
                    "role": "cluster",
                    "measurement_level": "nominal",
                },
            }
        }
    )
    analysis_plan = AnalysisPlan.model_validate(
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
                    "options": {"group_variable": "group"},
                },
            },
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
    )

    assert registration.model_type == "mixed_random_intercept"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "11_robustness_analysis",
        "13_effect_size_analysis",
        "15_regression_visualization",
        "16_research_audit",
    )
    assert_steps_registered(
        orchestrator,
        "12_advanced_robustness",
    )
    assert_step_order(
        orchestrator,
        before="10_regression_diagnostics",
        after="13_effect_size_analysis",
    )
