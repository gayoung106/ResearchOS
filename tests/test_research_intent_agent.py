from pathlib import Path

import yaml

from src.auto import (
    AgentHypothesis,
    AgentResearchModel,
    AgentVariableMatch,
    ResearchIntent,
    agent_research_model_from_dict,
    agent_research_model_validation_to_dataframe,
    apply_agent_research_model_to_analysis_plan,
    apply_agent_research_model_to_variable_map,
    build_agent_analysis_strategy_models,
    build_claude_research_model_prompt,
    build_research_concept_variable_matches,
    build_research_context_packet,
    draft_agent_research_model_from_intent,
    draft_research_model_quality_to_dataframe,
    evaluate_draft_research_model_quality,
    infer_research_intent_structure,
    load_agent_research_model,
    load_research_intent,
    research_concept_variable_matches_to_dataframe,
    research_intent_extraction_to_dict,
    validate_agent_research_model,
    write_agent_research_model,
    write_claude_research_model_prompt,
    write_research_context_packet,
    write_research_intent_template,
)
from src.common.config_models import AnalysisPlan, VariableDefinition, VariableMap


def _variable_map() -> VariableMap:
    return VariableMap(
        variables={
            "satisfaction": VariableDefinition(
                original_name="satisfaction",
                korean_name="satisfaction",
                label="Job satisfaction",
                question_text="How satisfied are you with your job?",
                role="other",
                measurement_level="continuous",
            ),
            "autonomy": VariableDefinition(
                original_name="autonomy",
                korean_name="autonomy",
                label="Job autonomy",
                question_text="I can decide how to do my work.",
                role="other",
                measurement_level="continuous",
            ),
            "burnout": VariableDefinition(
                original_name="burnout",
                korean_name="burnout",
                label="Burnout",
                question_text="I feel exhausted by work.",
                role="other",
                measurement_level="continuous",
            ),
            "gender": VariableDefinition(
                original_name="gender",
                korean_name="gender",
                label="Gender",
                role="other",
                measurement_level="nominal",
            ),
            "team": VariableDefinition(
                original_name="team",
                korean_name="team",
                label="Team",
                role="cluster",
                measurement_level="nominal",
            ),
        }
    )


def test_research_intent_template_and_loader_round_trip(tmp_path: Path) -> None:
    template_path = write_research_intent_template(tmp_path / "research_intent.yaml")
    data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    data["research_topic"] = "work design and satisfaction"
    data["dependent_concepts"] = ["satisfaction"]
    template_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    intent = load_research_intent(template_path)
    plain = tmp_path / "intent.txt"
    plain.write_text("I want to study whether autonomy predicts satisfaction.", encoding="utf-8")

    assert intent.research_topic == "work design and satisfaction"
    assert intent.dependent_concepts == ["satisfaction"]
    assert load_research_intent(plain).raw_text.startswith("I want to study")


def test_build_research_context_packet_and_claude_prompt(tmp_path: Path) -> None:
    intent = ResearchIntent(
        research_topic="work design",
        research_goal="Explain satisfaction from autonomy and burnout.",
        dependent_concepts=["satisfaction"],
        independent_concepts=["autonomy"],
    )
    packet = build_research_context_packet(
        intent,
        _variable_map(),
        quality_report=[{"variable": "satisfaction", "missing_rate": 0.0}],
    )
    packet_path = write_research_context_packet(packet, tmp_path / "research_context_packet.json")
    prompt_path = write_claude_research_model_prompt(packet, tmp_path / "claude_prompt.txt")
    prompt = build_claude_research_model_prompt(packet)

    assert packet["research_intent"]["research_topic"] == "work design"
    assert packet["available_variables"][0]["name"] == "satisfaction"
    assert packet["available_variables"][0]["quality"]["missing_rate"] == 0.0
    assert packet_path.exists()
    assert prompt_path.exists()
    assert "Return YAML only" in prompt
    assert "dependent_variable" in prompt


def test_agent_research_model_validation_and_apply_to_plan() -> None:
    variable_map = _variable_map()
    analysis_plan = AnalysisPlan()
    model = AgentResearchModel(
        dependent_variable="satisfaction",
        independent_variables=["autonomy"],
        mediators=["burnout"],
        controls=["gender"],
        clusters=["team"],
        variable_matches=[AgentVariableMatch("autonomy", "independent", 0.82, "Matches intent")],
        hypotheses=[AgentHypothesis("H1", "Autonomy predicts satisfaction.", ["autonomy", "satisfaction"], "+")],
        model_rationale="The variables match the stated work design intent.",
        confidence=0.74,
        requires_human_review=False,
    )

    report = validate_agent_research_model(model, variable_map)
    updated_map = apply_agent_research_model_to_variable_map(variable_map, model)
    updated_plan = apply_agent_research_model_to_analysis_plan(analysis_plan, model, variable_map)

    assert report.passed is True
    assert agent_research_model_validation_to_dataframe(report).shape[0] >= 3
    assert updated_map.variables["satisfaction"].role == "dependent"
    assert updated_map.variables["burnout"].role == "mediator"
    assert updated_plan.variables.dependent == ["satisfaction"]
    assert updated_plan.variables.mediators == ["burnout"]
    assert updated_plan.analyses.mediation.enabled is True
    assert updated_plan.analyses.mediation.models[0]["mediator_variable"] == "burnout"
    assert updated_plan.analyses.mediation.methods == ["causal_steps_bootstrap_indirect_effect"]
    assert updated_plan.analyses.mediation.checks[0]["check"] == "temporal_order_review"
    assert updated_plan.analyses.regression.options["agent_hypotheses"][0]["hypothesis_id"] == "H1"


def test_agent_research_model_validation_rejects_unknown_variables() -> None:
    model = AgentResearchModel(
        dependent_variable="missing_outcome",
        independent_variables=["autonomy"],
    )

    report = validate_agent_research_model(model, _variable_map())

    assert report.passed is False
    assert any(item.item == "referenced_variables" and not item.passed for item in report.issues)


def test_load_agent_research_model_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "agent_research_model.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dependent_variable": "satisfaction",
                "independent_variables": ["autonomy"],
                "variable_matches": [
                    {"variable_name": "autonomy", "role": "independent", "confidence": 0.8}
                ],
                "hypotheses": [
                    {"hypothesis_id": "H1", "statement": "Autonomy matters.", "focal_variables": ["autonomy"]}
                ],
                "confidence": 0.7,
                "requires_human_review": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    model = load_agent_research_model(path)
    parsed = agent_research_model_from_dict({"dependent_variable": "satisfaction", "independent_variables": "autonomy"})

    assert model.dependent_variable == "satisfaction"
    assert model.variable_matches[0].confidence == 0.8
    assert parsed.independent_variables == ["autonomy"]


def test_infer_research_intent_structure_from_natural_language() -> None:
    intent = ResearchIntent(
        raw_text=(
            "I want to study whether job autonomy has a positive effect on job satisfaction. "
            "The dependent variable is job satisfaction. "
            "The independent variable is job autonomy. "
            "The mediator is burnout. "
            "The moderator is supervisor support. "
            "Control variables are age and gender."
        )
    )

    extraction = infer_research_intent_structure(intent)
    data = research_intent_extraction_to_dict(extraction)

    assert extraction.research_questions
    assert "job satisfaction" in extraction.dependent_concepts
    assert "job autonomy" in extraction.independent_concepts
    assert "burnout" in extraction.mediator_concepts
    assert "supervisor support" in extraction.moderator_concepts
    assert any("age" in concept for concept in extraction.control_concepts)
    assert extraction.hypothesis_candidates[0].expected_direction == "+"
    assert data["hypothesis_candidates"][0]["hypothesis_id"] == "H1"


def test_research_context_packet_contains_structured_intent() -> None:
    intent = ResearchIntent(
        raw_text=(
            "The dependent variable is satisfaction. "
            "The independent variable is autonomy."
        )
    )

    packet = build_research_context_packet(intent, _variable_map())

    assert packet["structured_research_intent"]["dependent_concepts"] == ["satisfaction"]
    assert packet["structured_research_intent"]["independent_concepts"] == ["autonomy"]
    assert packet["research_intent"]["structured_intent"]["dependent_concepts"] == ["satisfaction"]


def test_build_research_concept_variable_matches_scores_metadata_candidates() -> None:
    intent = ResearchIntent(
        raw_text=(
            "The dependent variable is job satisfaction. "
            "The independent variable is job autonomy. "
            "The mediator is burnout."
        )
    )
    extraction = infer_research_intent_structure(intent)

    matches = build_research_concept_variable_matches(extraction, _variable_map())
    frame = research_concept_variable_matches_to_dataframe(matches)

    assert matches[0].concept == "job satisfaction"
    assert matches[0].variable_name == "satisfaction"
    assert matches[0].score >= 0.8
    assert frame.loc[frame["concept"] == "job autonomy", "variable_name"].iloc[0] == "autonomy"
    assert "burnout" in set(frame["variable_name"])


def test_research_context_packet_contains_concept_variable_matches() -> None:
    intent = ResearchIntent(
        raw_text=(
            "The dependent variable is job satisfaction. "
            "The independent variable is job autonomy."
        )
    )

    packet = build_research_context_packet(intent, _variable_map())

    assert packet["concept_variable_matches"]
    assert packet["concept_variable_matches"][0]["variable_name"] == "satisfaction"
    assert packet["concept_variable_matches"][0]["role"] == "dependent"


def test_draft_agent_research_model_from_intent_uses_best_matches(tmp_path: Path) -> None:
    intent = ResearchIntent(
        raw_text=(
            "The dependent variable is job satisfaction. "
            "The independent variable is job autonomy. "
            "The mediator is burnout. "
            "Control variables are gender."
        )
    )
    extraction = infer_research_intent_structure(intent)
    matches = build_research_concept_variable_matches(extraction, _variable_map())

    model = draft_agent_research_model_from_intent(extraction, _variable_map(), matches=matches)
    model_path = write_agent_research_model(model, tmp_path / "draft_agent_research_model.yaml")
    loaded = load_agent_research_model(model_path)

    assert model.dependent_variable == "satisfaction"
    assert model.independent_variables == ["autonomy"]
    assert model.mediators == ["burnout"]
    assert model.controls == ["gender"]
    assert model.hypotheses[0].hypothesis_id == "H1"
    assert loaded.dependent_variable == "satisfaction"


def test_evaluate_draft_research_model_quality_reports_risk_items() -> None:
    intent = ResearchIntent(
        raw_text=(
            "The dependent variable is job satisfaction. "
            "The independent variable is job autonomy. "
            "The mediator is burnout."
        )
    )
    extraction = infer_research_intent_structure(intent)
    matches = build_research_concept_variable_matches(extraction, _variable_map())
    model = draft_agent_research_model_from_intent(extraction, _variable_map(), matches=matches)

    report = evaluate_draft_research_model_quality(model, extraction, _variable_map())
    frame = draft_research_model_quality_to_dataframe(report)

    assert report.overall_score > 0
    assert report.risk_level in {"low", "medium", "high"}
    assert "concept_variable_match_strength" in set(frame["item"])
    assert "model_complexity_risk" in set(frame["item"])


def test_build_agent_analysis_strategy_models_creates_mediation_and_moderation_specs() -> None:
    model = AgentResearchModel(
        dependent_variable="satisfaction",
        independent_variables=["autonomy"],
        mediators=["burnout"],
        moderators=["team"],
        controls=["gender"],
    )

    strategies = build_agent_analysis_strategy_models(model)

    assert strategies["mediation"][0]["method"] == "causal_steps_bootstrap_indirect_effect"
    assert strategies["mediation"][0]["mediator_variable"] == "burnout"
    assert strategies["moderation"][0]["method"] == "interaction_regression"
    assert strategies["moderation"][0]["interaction_term"] == "autonomy__x__team"
