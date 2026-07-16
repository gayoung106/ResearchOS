"""Regression Pipeline Builder 테스트."""

from pathlib import Path

from src.common.config_models import (
    AnalysisPlan,
    VariableMap,
)
from tests.support.assertions import (
    assert_registry_matches,
)
from tests.support.builders import (
    build_regression_pipeline,
)
from tests.support.expected_pipeline import (
    count_pipeline,
    logit_pipeline,
    ols_pipeline,
)


def test_disabled_regression_is_not_registered(
    tmp_path: Path,
    empty_analysis_plan: AnalysisPlan,
    empty_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=empty_analysis_plan,
        variable_map=empty_variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is False
    assert registration.diagnostics_registered is False
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False
    assert registration.effect_size_registered is False
    assert registration.reporting_registered is False
    assert registration.visualization_registered is False
    assert registration.audit_registered is False
    assert orchestrator.registry.names() == []


def test_continuous_outcome_registers_full_ols_pipeline(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x1"],
                "controls": ["x2"],
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
                "x1": {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
                "x2": {
                    "role": "control",
                    "measurement_level": "continuous",
                },
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is True
    assert registration.model_type == "ols"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.robustness_registered is True
    assert registration.advanced_robustness_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True

    assert_registry_matches(
        orchestrator,
        ols_pipeline(
            robustness=True,
            advanced_robustness=True,
        ),
    )


def test_ols_without_robustness_enabled_skips_robustness(
    tmp_path: Path,
    ols_without_robustness_analysis_plan: AnalysisPlan,
    continuous_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=ols_without_robustness_analysis_plan,
        variable_map=continuous_variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is True
    assert registration.model_type == "ols"
    assert registration.diagnostics_registered is True
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True

    assert_registry_matches(
        orchestrator,
        ols_pipeline(
            robustness=False,
            advanced_robustness=False,
        ),
    )

    assert any("강건성 분석 설정이 비활성화" in warning for warning in registration.warnings)


def test_binary_outcome_registers_logit_pipeline(
    tmp_path: Path,
    ols_with_robustness_analysis_plan: AnalysisPlan,
    binary_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=binary_variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is True
    assert registration.model_type == "binary_logit"
    assert registration.measurement_level == "binary"
    assert registration.diagnostics_registered is True
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True

    assert_registry_matches(
        orchestrator,
        logit_pipeline(),
    )

    assert any("강건성 단계는 OLS 모형만 지원" in warning for warning in registration.warnings)


def test_unknown_measurement_level_blocks_registration(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x1"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                }
            },
        }
    )

    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {
                    "measurement_level": "unknown",
                },
                "x1": {
                    "measurement_level": "continuous",
                },
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is False
    assert registration.diagnostics_registered is False
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False
    assert registration.effect_size_registered is False
    assert registration.reporting_registered is False
    assert registration.visualization_registered is False
    assert registration.audit_registered is False
    assert orchestrator.registry.names() == []
    assert registration.warnings


def test_multiple_outcomes_are_rejected(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": [
                    "y1",
                    "y2",
                ],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                }
            },
        }
    )

    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y1": {
                    "measurement_level": "continuous",
                },
                "y2": {
                    "measurement_level": "continuous",
                },
                "x": {
                    "measurement_level": "continuous",
                },
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is False
    assert registration.diagnostics_registered is False
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False
    assert registration.effect_size_registered is False
    assert registration.reporting_registered is False
    assert registration.visualization_registered is False
    assert registration.audit_registered is False
    assert "종속변수 1개" in registration.warnings[0]
    assert orchestrator.registry.names() == []


def test_predictors_are_collected_in_role_order(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": [
                    "x1",
                    "x2",
                ],
                "mediators": ["mediator"],
                "moderators": ["moderator"],
                "controls": ["x3"],
                "fixed_effects": ["country"],
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

    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {
                    "role": "dependent",
                    "measurement_level": "continuous",
                },
                "x1": {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
                "x2": {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
                "mediator": {
                    "role": "mediator",
                    "measurement_level": "continuous",
                },
                "moderator": {
                    "role": "moderator",
                    "measurement_level": "continuous",
                },
                "x3": {
                    "role": "control",
                    "measurement_level": "continuous",
                },
                "country": {
                    "role": "fixed_effect",
                    "measurement_level": "nominal",
                },
            }
        }
    )

    _, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        project_name="테스트 연구",
    )

    assert registration.registered is True
    assert registration.independent_variables == [
        "x1",
        "x2",
        "mediator",
        "moderator",
        "x3",
    ]
    assert registration.fixed_effects == [
        "country",
    ]


def test_count_outcome_registers_auto_count_pipeline(
    tmp_path: Path,
    ols_with_robustness_analysis_plan: AnalysisPlan,
    count_variable_map: VariableMap,
) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=count_variable_map,
        project_name="계수형 연구",
    )

    assert registration.registered is True
    assert registration.model_type == "count_auto"
    assert registration.measurement_level == "count"
    assert registration.diagnostics_registered is True
    assert registration.robustness_registered is False
    assert registration.advanced_robustness_registered is False
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True

    assert_registry_matches(
        orchestrator,
        count_pipeline(),
    )

    assert any("강건성 단계는 OLS 모형만 지원" in warning for warning in registration.warnings)
