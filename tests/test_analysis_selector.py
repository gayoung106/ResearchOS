"""분석 지식베이스 및 추천기 테스트."""

from src.common.config_models import AnalysisPlan, ResearchPlan, VariableMap
from src.planning.analysis_selector import (
    recommend_analysis_methods,
    recommendation_summary,
)
from src.planning.knowledge_base import (
    get_analysis_knowledge,
    get_methods_for_outcome,
)


def test_binary_outcome_recommends_logit() -> None:
    research_plan = ResearchPlan.model_validate({})
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
                "independent": ["public_sector"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "measurement_level": "binary",
                    "role": "dependent",
                }
            }
        }
    )

    result = recommend_analysis_methods(
        research_plan,
        analysis_plan,
        variable_map,
    )

    assert result.measurement_level == "binary"
    assert result.recommendations[0].method_id == "binary_logit"


def test_continuous_outcome_recommends_ols() -> None:
    research_plan = ResearchPlan.model_validate({})
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["score"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "score": {
                    "measurement_level": "continuous",
                    "role": "dependent",
                }
            }
        }
    )

    result = recommend_analysis_methods(
        research_plan,
        analysis_plan,
        variable_map,
    )

    assert result.recommendations[0].method_id == "ols"


def test_unknown_measurement_level_returns_warning() -> None:
    result = recommend_analysis_methods(
        ResearchPlan.model_validate({}),
        AnalysisPlan.model_validate({"variables": {"dependent": ["unknown_outcome"]}}),
        VariableMap.model_validate({"variables": {}}),
    )

    assert result.recommendations == []
    assert "측정수준이 확인되지 않았습니다" in result.warnings[0]


def test_cluster_and_weight_requirements_are_added() -> None:
    research_plan = ResearchPlan.model_validate(
        {
            "data": {
                "cluster_variable": "country",
                "weight_variable": "weight",
            }
        }
    )
    analysis_plan = AnalysisPlan.model_validate({"variables": {"dependent": ["outcome"]}})
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "measurement_level": "binary",
                    "role": "dependent",
                }
            }
        }
    )

    result = recommend_analysis_methods(
        research_plan,
        analysis_plan,
        variable_map,
    )
    requirements = result.recommendations[0].additional_requirements

    assert any("군집변수" in item for item in requirements)
    assert any("표본가중치" in item for item in requirements)


def test_mediation_and_moderation_generate_warnings() -> None:
    research_plan = ResearchPlan.model_validate({})
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
                "mediators": ["mediator"],
                "moderators": ["moderator"],
            }
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "outcome": {
                    "measurement_level": "continuous",
                    "role": "dependent",
                }
            }
        }
    )

    result = recommend_analysis_methods(
        research_plan,
        analysis_plan,
        variable_map,
    )

    assert any("매개변수" in warning for warning in result.warnings)
    assert any("조절변수" in warning for warning in result.warnings)


def test_knowledge_base_lookup() -> None:
    knowledge = get_analysis_knowledge("ordered_logit")

    assert knowledge.korean_name == "순서형 로지스틱 회귀분석"
    assert "비례오즈 가정" in knowledge.required_diagnostics
    assert get_methods_for_outcome("ordinal") == ("ordered_logit",)


def test_recommendation_summary() -> None:
    result = recommend_analysis_methods(
        ResearchPlan.model_validate({}),
        AnalysisPlan.model_validate({"variables": {"dependent": ["count_var"]}}),
        VariableMap.model_validate(
            {
                "variables": {
                    "count_var": {
                        "measurement_level": "count",
                        "role": "dependent",
                    }
                }
            }
        ),
    )

    summary = recommendation_summary(result)

    assert summary["recommended_methods"] == ["poisson"]
    assert summary["measurement_level"] == "count"
