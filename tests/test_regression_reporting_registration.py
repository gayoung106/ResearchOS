"""회귀 보고서 단계 자동 등록 테스트."""

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


def _analysis_plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
            },
        }
    )


def test_ols_registers_reporting_after_effect_size(
    tmp_path: Path,
    continuous_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=_analysis_plan(),
        variable_map=(continuous_variable_map),
    )

    assert registration.reporting_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
    )

    assert_step_order(
        orchestrator,
        before="13_effect_size_analysis",
        after="14_regression_reporting",
    )


def test_binary_logit_registers_reporting(
    tmp_path: Path,
    binary_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=_analysis_plan(),
        variable_map=(binary_variable_map),
    )

    assert registration.diagnostics_registered is True
    assert registration.reporting_registered is True
    assert registration.robustness_registered is False

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
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


def test_ordered_logit_registers_reporting(
    tmp_path: Path,
    ordinal_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=_analysis_plan(),
        variable_map=(ordinal_variable_map),
    )

    assert registration.diagnostics_registered is True
    assert registration.reporting_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
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


def test_unregistered_regression_has_no_reporting(
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
    assert registration.reporting_registered is False

    assert_steps_not_registered(
        orchestrator,
        "09_regression_analysis",
        "14_regression_reporting",
    )
