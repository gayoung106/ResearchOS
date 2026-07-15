"""Research Audit 단계 자동 등록 테스트."""

from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from tests.support.assertions import (
    assert_step_order,
    assert_steps_not_registered,
    assert_steps_registered,
)
from tests.support.builders import build_regression_pipeline


def test_ols_registers_audit_after_visualization(
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
        variable_map=continuous_variable_map,
    )

    assert registration.audit_registered is True

    assert_steps_registered(
        orchestrator,
        "15_regression_visualization",
        "16_research_audit",
    )

    assert_step_order(
        orchestrator,
        before="15_regression_visualization",
        after="16_research_audit",
    )


def test_ols_without_robustness_registers_audit(
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
        variable_map=continuous_variable_map,
    )

    assert registration.audit_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    )
    assert_steps_not_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="15_regression_visualization",
        after="16_research_audit",
    )


def test_binary_logit_registers_audit(
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
            },
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=binary_variable_map,
    )

    assert registration.audit_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    )
    assert_steps_not_registered(
        orchestrator,
        "10_regression_diagnostics",
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="15_regression_visualization",
        after="16_research_audit",
    )


def test_ordered_logit_registers_audit(
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
                },
            },
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=ordinal_variable_map,
    )

    assert registration.audit_registered is True

    assert_steps_registered(
        orchestrator,
        "09_regression_analysis",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    )
    assert_steps_not_registered(
        orchestrator,
        "10_regression_diagnostics",
        "11_robustness_analysis",
        "12_advanced_robustness",
    )

    assert_step_order(
        orchestrator,
        before="15_regression_visualization",
        after="16_research_audit",
    )


def test_unregistered_regression_has_no_audit(
    tmp_path: Path,
    empty_analysis_plan: AnalysisPlan,
    empty_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=empty_analysis_plan,
        variable_map=empty_variable_map,
    )

    assert registration.registered is False
    assert registration.audit_registered is False

    assert_steps_not_registered(
        orchestrator,
        "09_regression_analysis",
        "15_regression_visualization",
        "16_research_audit",
    )
