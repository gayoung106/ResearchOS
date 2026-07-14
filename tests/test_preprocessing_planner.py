"""전처리 계획 생성기 테스트."""

from src.common.config_models import AnalysisPlan, VariableMap
from src.preprocess.evidence_resolver import ResolvedVariableLevel
from src.preprocess.planner import (
    plan_preprocessing,
    preprocessing_plan_summary,
)


def resolved(
    variable_name: str,
    level: str,
    status: str = "confirmed",
) -> ResolvedVariableLevel:
    return ResolvedVariableLevel(
        variable_name=variable_name,
        detected_level=level,
        resolved_level=level,
        status=status,
        confidence=0.95,
        supporting_sources=["codebook"],
        conflicts=[],
        notes=[],
    )


def test_binary_variable_creates_recoding_review() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "role": "dependent",
                    "measurement_level": "binary",
                }
            }
        }
    )

    plan = plan_preprocessing(
        analysis_plan,
        variable_map,
        [resolved("outcome", "binary")],
    )

    assert any(action.action_type == "review_binary_recoding" for action in plan.actions)


def test_missing_and_reverse_rules_are_planned() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "independent": ["scale_item_1"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "scale_item_1": {
                    "role": "independent",
                    "measurement_level": "scale_item",
                    "missing_values": [8, 9],
                    "reverse_coded": True,
                    "scale_name": "trust",
                    "coding": {
                        "min": 1,
                        "max": 5,
                    },
                }
            }
        }
    )

    plan = plan_preprocessing(
        analysis_plan,
        variable_map,
        [resolved("scale_item_1", "scale_item")],
    )

    action_types = {action.action_type for action in plan.actions}

    assert "replace_missing_values" in action_types
    assert "reverse_code" in action_types
    assert "assign_scale_item" in action_types


def test_continuous_moderator_creates_centering_plan() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "moderators": ["age"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "age": {
                    "role": "moderator",
                    "measurement_level": "continuous",
                }
            }
        }
    )

    plan = plan_preprocessing(
        analysis_plan,
        variable_map,
        [resolved("age", "continuous")],
    )

    assert any(action.action_type == "mean_center" for action in plan.actions)


def test_conflicted_variable_is_blocked() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "role": "dependent",
                    "measurement_level": "unknown",
                }
            }
        }
    )
    conflict = ResolvedVariableLevel(
        variable_name="outcome",
        detected_level="binary",
        resolved_level="unknown",
        status="conflict",
        confidence=0.0,
        supporting_sources=["questionnaire", "codebook"],
        conflicts=["근거 충돌"],
        notes=[],
    )

    plan = plan_preprocessing(
        analysis_plan,
        variable_map,
        [conflict],
    )

    assert plan.blocked_variables == ["outcome"]
    assert plan.actions == []


def test_configured_recoding_and_derived_variable_are_planned() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "preprocessing": {
                "recoding_rules": [
                    {
                        "variable": "gender",
                        "mapping": {
                            "1": 0,
                            "2": 1,
                        },
                    }
                ],
                "derived_variables": [
                    {
                        "name": "age_squared",
                        "expression": "age ** 2",
                    }
                ],
            }
        }
    )

    plan = plan_preprocessing(
        analysis_plan,
        VariableMap.model_validate({"variables": {}}),
        [],
    )

    action_types = [action.action_type for action in plan.actions]

    assert "configured_recoding" in action_types
    assert "create_derived_variable" in action_types


def test_missing_variable_definition_blocks_variable() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "controls": ["age"],
            }
        }
    )

    plan = plan_preprocessing(
        analysis_plan,
        VariableMap.model_validate({"variables": {}}),
        [resolved("age", "continuous")],
    )

    assert plan.blocked_variables == ["age"]
    assert plan.warnings


def test_preprocessing_plan_summary() -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "role": "dependent",
                    "measurement_level": "binary",
                }
            }
        }
    )

    plan = plan_preprocessing(
        analysis_plan,
        variable_map,
        [resolved("outcome", "binary")],
    )
    summary = preprocessing_plan_summary(plan)

    assert summary["action_count"] == 1
    assert summary["confirmation_required_count"] == 1
    assert summary["blocked_variable_count"] == 0
