"""Research Audit 단계 자동 등록 테스트."""

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


def test_ols_registers_audit(
    tmp_path: Path,
    continuous_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=_analysis_plan(),
        variable_map=(continuous_variable_map),
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


def test_binary_logit_registers_audit(
    tmp_path: Path,
    binary_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=_analysis_plan(),
        variable_map=(binary_variable_map),
    )

    assert registration.diagnostics_registered is True
    assert registration.audit_registered is True
    assert registration.robustness_registered is False

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
        before="15_regression_visualization",
        after="16_research_audit",
    )


def test_ordered_logit_registers_audit(
    tmp_path: Path,
    ordinal_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=_analysis_plan(),
        variable_map=(ordinal_variable_map),
    )

    assert registration.diagnostics_registered is True
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
        before="10_regression_diagnostics",
        after="13_effect_size_analysis",
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
        analysis_plan=(empty_analysis_plan),
        variable_map=(empty_variable_map),
    )

    assert registration.registered is False
    assert registration.audit_registered is False

    assert_steps_not_registered(
        orchestrator,
        "09_regression_analysis",
        "16_research_audit",
    )


def test_mixed_effects_registers_audit(
    tmp_path: Path,
) -> None:
    plan = AnalysisPlan.model_validate(
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
                "robustness": {"enabled": False},
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
                "group": {
                    "role": "cluster",
                    "measurement_level": "nominal",
                },
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=plan,
        variable_map=variable_map,
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
