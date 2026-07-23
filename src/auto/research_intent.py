"""Research-intent and external research-agent helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult

_AGENT_ROLES = {
    "dependent",
    "independent",
    "mediator",
    "moderator",
    "control",
    "cluster",
    "weight",
}
_PLAN_ROLE_FIELDS = {
    "dependent": "dependent",
    "independent": "independent",
    "mediator": "mediators",
    "moderator": "moderators",
    "control": "controls",
    "cluster": "clusters",
    "weight": "weights",
}


@dataclass(slots=True)
class ResearchIntent:
    """User-facing description of what the user wants to study."""

    research_topic: str = ""
    research_goal: str = ""
    target_population: str = ""
    unit_of_analysis: str = ""
    dependent_concepts: list[str] = field(default_factory=list)
    independent_concepts: list[str] = field(default_factory=list)
    mediator_concepts: list[str] = field(default_factory=list)
    moderator_concepts: list[str] = field(default_factory=list)
    control_concepts: list[str] = field(default_factory=list)
    raw_text: str = ""


@dataclass(slots=True)
class ResearchIntentHypothesisCandidate:
    """A lightweight hypothesis candidate inferred from intent text."""

    hypothesis_id: str
    statement: str
    dependent_concept: str = ""
    independent_concept: str = ""
    expected_direction: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class ResearchIntentExtraction:
    """Structured concepts inferred from natural-language research intent."""

    research_questions: list[str] = field(default_factory=list)
    dependent_concepts: list[str] = field(default_factory=list)
    independent_concepts: list[str] = field(default_factory=list)
    mediator_concepts: list[str] = field(default_factory=list)
    moderator_concepts: list[str] = field(default_factory=list)
    control_concepts: list[str] = field(default_factory=list)
    hypothesis_candidates: list[ResearchIntentHypothesisCandidate] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentVariableMatch:
    """A variable-role recommendation returned by an external research agent."""

    variable_name: str
    role: str
    confidence: float = 0.0
    rationale: str = ""


@dataclass(slots=True)
class AgentHypothesis:
    """A proposed hypothesis from an external research agent."""

    hypothesis_id: str
    statement: str
    focal_variables: list[str] = field(default_factory=list)
    expected_direction: str = ""


@dataclass(slots=True)
class AgentResearchModel:
    """Validated shape for a Claude or other LLM-proposed research model."""

    dependent_variable: str | None = None
    independent_variables: list[str] = field(default_factory=list)
    mediators: list[str] = field(default_factory=list)
    moderators: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    clusters: list[str] = field(default_factory=list)
    weights: list[str] = field(default_factory=list)
    variable_matches: list[AgentVariableMatch] = field(default_factory=list)
    hypotheses: list[AgentHypothesis] = field(default_factory=list)
    model_rationale: str = ""
    confidence: float = 0.0
    requires_human_review: bool = True


@dataclass(slots=True)
class AgentResearchModelValidationIssue:
    """One validation issue for an agent-proposed research model."""

    item: str
    passed: bool
    evidence: str
    suggestion: str = ""


@dataclass(slots=True)
class AgentResearchModelValidationReport:
    """Validation report for an agent-proposed research model."""

    passed: bool
    issues: list[AgentResearchModelValidationIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return output


def _normalize_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_OUTCOME_HINTS = (
    "outcome",
    "dependent variable",
    "result",
    "performance",
    "satisfaction",
    "effectiveness",
    "score",
    "??",
    "??",
    "??",
    "??",
)
_PREDICTOR_HINTS = (
    "predictor",
    "independent variable",
    "determinant",
    "factor",
    "effect of",
    "impact of",
    "influence of",
    "??",
    "????",
    "??",
)
_MEDIATOR_HINTS = ("mediator", "mediate", "mediation", "??", "????")
_MODERATOR_HINTS = ("moderator", "moderate", "moderation", "??", "????")
_CONTROL_HINTS = ("control", "covariate", "adjust", "??", "???")
_POSITIVE_HINTS = ("positive", "increase", "higher", "?", "??", "+")
_NEGATIVE_HINTS = ("negative", "decrease", "lower", "?", "??", "-")
_SPLIT_PATTERN = re.compile(r"[\n.;?]+")
_CONCEPT_SEPARATOR_PATTERN = re.compile(r",|/|\band\b|\bor\b")


def _clean_concept(value: str) -> str:
    value = re.sub(r"^[\s:?\-]+|[\s:?\-]+$", "", value.strip())
    value = re.sub(r"\s+", " ", value)
    return value[:80]


def _append_unique(values: list[str], value: str) -> None:
    cleaned = _clean_concept(value)
    if cleaned and cleaned not in values:
        values.append(cleaned)


def _extract_after_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for marker in markers:
        pattern = re.compile(
            rf"\b{re.escape(marker)}\b\s*"
            rf"(?:variables?|concepts?)?\s*"
            rf"(?:is|are|as|:|=)?\s*"
            rf"([^.;\n]+)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            fragment = match.group(1)
            for part in _CONCEPT_SEPARATOR_PATTERN.split(fragment):
                candidate = _clean_concept(part)
                if candidate and len(candidate.split()) <= 6:
                    _append_unique(values, candidate)
    return values


def _infer_direction(text: str) -> str:
    lowered = text.lower()
    if any(hint in lowered for hint in _NEGATIVE_HINTS):
        return "-"
    if any(hint in lowered for hint in _POSITIVE_HINTS):
        return "+"
    return ""


def _intent_text(intent: ResearchIntent) -> str:
    parts = [
        intent.research_topic,
        intent.research_goal,
        intent.target_population,
        intent.unit_of_analysis,
        " ".join(intent.dependent_concepts),
        " ".join(intent.independent_concepts),
        " ".join(intent.mediator_concepts),
        " ".join(intent.moderator_concepts),
        " ".join(intent.control_concepts),
        intent.raw_text,
    ]
    return "\n".join(part for part in parts if part).strip()


def infer_research_intent_structure(intent: ResearchIntent) -> ResearchIntentExtraction:
    """Infer first-pass research questions, concepts, and hypothesis candidates from intent text."""
    text = _intent_text(intent)
    extraction = ResearchIntentExtraction(
        dependent_concepts=list(intent.dependent_concepts),
        independent_concepts=list(intent.independent_concepts),
        mediator_concepts=list(intent.mediator_concepts),
        moderator_concepts=list(intent.moderator_concepts),
        control_concepts=list(intent.control_concepts),
    )
    if not text:
        extraction.warnings.append("No research intent text was provided.")
        return extraction

    for sentence in [_clean_concept(part) for part in _SPLIT_PATTERN.split(text) if _clean_concept(part)]:
        lowered = sentence.lower()
        if any(token in lowered for token in ["whether", "how", "what", "relationship", "effect", "impact", "influence"]):
            _append_unique(extraction.research_questions, sentence)
        if any(hint in lowered for hint in _OUTCOME_HINTS):
            for concept in _extract_after_markers(sentence, ("outcome", "dependent variable", "????", "????")):
                _append_unique(extraction.dependent_concepts, concept)
        if any(hint in lowered for hint in _PREDICTOR_HINTS):
            for concept in _extract_after_markers(sentence, ("predictor", "independent variable", "factor", "????", "????")):
                _append_unique(extraction.independent_concepts, concept)
        if any(hint in lowered for hint in _MEDIATOR_HINTS):
            for concept in _extract_after_markers(sentence, ("mediator", "mediating variable", "????")):
                _append_unique(extraction.mediator_concepts, concept)
        if any(hint in lowered for hint in _MODERATOR_HINTS):
            for concept in _extract_after_markers(sentence, ("moderator", "moderating variable", "????")):
                _append_unique(extraction.moderator_concepts, concept)
        if any(hint in lowered for hint in _CONTROL_HINTS):
            for concept in _extract_after_markers(sentence, ("control", "control variable", "covariate", "????")):
                _append_unique(extraction.control_concepts, concept)

    if not extraction.research_questions and text:
        _append_unique(extraction.research_questions, text.splitlines()[0])

    dependent = extraction.dependent_concepts[0] if extraction.dependent_concepts else ""
    direction = _infer_direction(text)
    for index, independent in enumerate(extraction.independent_concepts[:5], start=1):
        statement = f"{independent} is associated with {dependent}." if dependent else f"{independent} is associated with the outcome."
        extraction.hypothesis_candidates.append(
            ResearchIntentHypothesisCandidate(
                hypothesis_id=f"H{index}",
                statement=statement,
                dependent_concept=dependent,
                independent_concept=independent,
                expected_direction=direction,
                confidence=0.55 if dependent else 0.35,
            )
        )

    evidence_count = sum(
        bool(values)
        for values in [
            extraction.research_questions,
            extraction.dependent_concepts,
            extraction.independent_concepts,
            extraction.mediator_concepts,
            extraction.moderator_concepts,
            extraction.control_concepts,
        ]
    )
    extraction.confidence = min(0.85, 0.2 + evidence_count * 0.1)
    if not extraction.dependent_concepts:
        extraction.warnings.append("No dependent concept was clearly inferred from the research intent.")
    if not extraction.independent_concepts:
        extraction.warnings.append("No independent concept was clearly inferred from the research intent.")
    return extraction


def research_intent_extraction_to_dict(extraction: ResearchIntentExtraction) -> dict[str, Any]:
    return {
        "research_questions": list(extraction.research_questions),
        "dependent_concepts": list(extraction.dependent_concepts),
        "independent_concepts": list(extraction.independent_concepts),
        "mediator_concepts": list(extraction.mediator_concepts),
        "moderator_concepts": list(extraction.moderator_concepts),
        "control_concepts": list(extraction.control_concepts),
        "hypothesis_candidates": [
            {
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "dependent_concept": item.dependent_concept,
                "independent_concept": item.independent_concept,
                "expected_direction": item.expected_direction,
                "confidence": item.confidence,
            }
            for item in extraction.hypothesis_candidates
        ],
        "confidence": extraction.confidence,
        "warnings": list(extraction.warnings),
    }


def research_intent_to_dict(intent: ResearchIntent) -> dict[str, Any]:
    return {
        "research_topic": intent.research_topic,
        "research_goal": intent.research_goal,
        "target_population": intent.target_population,
        "unit_of_analysis": intent.unit_of_analysis,
        "dependent_concepts": list(intent.dependent_concepts),
        "independent_concepts": list(intent.independent_concepts),
        "mediator_concepts": list(intent.mediator_concepts),
        "moderator_concepts": list(intent.moderator_concepts),
        "control_concepts": list(intent.control_concepts),
        "raw_text": intent.raw_text,
        "structured_intent": research_intent_extraction_to_dict(infer_research_intent_structure(intent)),
    }


def load_research_intent(path: str | Path) -> ResearchIntent:
    """Load a research-intent YAML or plain-text file."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8-sig")
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Research intent YAML must contain a mapping at the top level.")
        return ResearchIntent(
            research_topic=str(data.get("research_topic", "") or ""),
            research_goal=str(data.get("research_goal", "") or ""),
            target_population=str(data.get("target_population", "") or ""),
            unit_of_analysis=str(data.get("unit_of_analysis", "") or ""),
            dependent_concepts=_normalize_string_list(data.get("dependent_concepts")),
            independent_concepts=_normalize_string_list(data.get("independent_concepts")),
            mediator_concepts=_normalize_string_list(data.get("mediator_concepts")),
            moderator_concepts=_normalize_string_list(data.get("moderator_concepts")),
            control_concepts=_normalize_string_list(data.get("control_concepts")),
            raw_text=str(data.get("raw_text", "") or ""),
        )
    return ResearchIntent(raw_text=text.strip())


def write_research_intent_template(path: str | Path) -> Path:
    """Write a Korean research-intent template for users."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "research_topic": "",
        "research_goal": "",
        "target_population": "",
        "unit_of_analysis": "",
        "dependent_concepts": [],
        "independent_concepts": [],
        "mediator_concepts": [],
        "moderator_concepts": [],
        "control_concepts": [],
        "raw_text": "",
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(template, file, allow_unicode=True, sort_keys=False)
    return output_path


def _variable_quality_rows(quality_report: Any | None) -> dict[str, dict[str, Any]]:
    if quality_report is None:
        return {}
    if isinstance(quality_report, pd.DataFrame):
        rows = quality_report.to_dict(orient="records")
    elif isinstance(quality_report, list):
        rows = [row for row in quality_report if isinstance(row, dict)]
    else:
        rows = []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("variable") or row.get("variable_name") or row.get("column")
        if name:
            output[str(name)] = dict(row)
    return output


def build_research_context_packet(
    intent: ResearchIntent,
    variable_map: VariableMap,
    *,
    quality_report: Any | None = None,
) -> dict[str, Any]:
    """Build the compact JSON payload to send to Claude or another research agent."""
    quality_by_variable = _variable_quality_rows(quality_report)
    variables: list[dict[str, Any]] = []
    for name, definition in variable_map.variables.items():
        variables.append(
            {
                "name": name,
                "original_name": definition.original_name,
                "korean_name": definition.korean_name,
                "label": definition.label,
                "question_text": definition.question_text,
                "inferred_role": definition.role,
                "measurement_level": definition.measurement_level,
                "coding": definition.coding,
                "review_status": definition.review_status,
                "notes": definition.notes,
                "quality": quality_by_variable.get(name, {}),
            }
        )
    return {
        "research_intent": research_intent_to_dict(intent),
        "structured_research_intent": research_intent_extraction_to_dict(infer_research_intent_structure(intent)),
        "available_variables": variables,
        "allowed_roles": sorted(_AGENT_ROLES),
        "required_output": "agent_research_model.yaml",
        "safety_rule": "Use only variables listed in available_variables.",
    }


def write_research_context_packet(packet: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def build_claude_research_model_prompt(packet: dict[str, Any]) -> str:
    """Create a Claude-ready prompt for research model extraction."""
    context_json = json.dumps(packet, ensure_ascii=False, indent=2)
    return f"""You are a cautious quantitative research design agent.

Task:
1. Read the research intent and available variables.
2. Select a theoretically plausible research model.
3. Use only variables listed in available_variables.
4. Prefer parsimonious models over exhaustive models.
5. Mark requires_human_review: true when theory, wording, or measurement is uncertain.

Return YAML only, with this exact top-level schema:

dependent_variable: null
independent_variables: []
mediators: []
moderators: []
controls: []
clusters: []
weights: []
variable_matches:
  - variable_name: ""
    role: "independent"
    confidence: 0.0
    rationale: ""
hypotheses:
  - hypothesis_id: "H1"
    statement: ""
    focal_variables: []
    expected_direction: ""
model_rationale: ""
confidence: 0.0
requires_human_review: true

Context packet:
{context_json}
"""


def write_claude_research_model_prompt(packet: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_claude_research_model_prompt(packet), encoding="utf-8")
    return output_path


def _agent_variable_match_from_dict(data: dict[str, Any]) -> AgentVariableMatch:
    return AgentVariableMatch(
        variable_name=str(data.get("variable_name", "") or "").strip(),
        role=str(data.get("role", "") or "").strip(),
        confidence=_normalize_float(data.get("confidence")),
        rationale=str(data.get("rationale", "") or ""),
    )


def _agent_hypothesis_from_dict(data: dict[str, Any]) -> AgentHypothesis:
    return AgentHypothesis(
        hypothesis_id=str(data.get("hypothesis_id", "") or "").strip(),
        statement=str(data.get("statement", "") or ""),
        focal_variables=_normalize_string_list(data.get("focal_variables")),
        expected_direction=str(data.get("expected_direction", "") or ""),
    )


def agent_research_model_from_dict(data: dict[str, Any]) -> AgentResearchModel:
    if not isinstance(data, dict):
        raise ValueError("Agent research model must be a mapping.")
    dependent = str(data.get("dependent_variable", "") or "").strip() or None
    return AgentResearchModel(
        dependent_variable=dependent,
        independent_variables=_normalize_string_list(data.get("independent_variables")),
        mediators=_normalize_string_list(data.get("mediators")),
        moderators=_normalize_string_list(data.get("moderators")),
        controls=_normalize_string_list(data.get("controls")),
        clusters=_normalize_string_list(data.get("clusters")),
        weights=_normalize_string_list(data.get("weights")),
        variable_matches=[
            _agent_variable_match_from_dict(item)
            for item in data.get("variable_matches", [])
            if isinstance(item, dict)
        ],
        hypotheses=[
            _agent_hypothesis_from_dict(item)
            for item in data.get("hypotheses", [])
            if isinstance(item, dict)
        ],
        model_rationale=str(data.get("model_rationale", "") or ""),
        confidence=_normalize_float(data.get("confidence")),
        requires_human_review=bool(data.get("requires_human_review", True)),
    )


def load_agent_research_model(path: str | Path) -> AgentResearchModel:
    file_path = Path(path)
    data = yaml.safe_load(file_path.read_text(encoding="utf-8-sig")) or {}
    return agent_research_model_from_dict(data)


def agent_research_model_to_dict(model: AgentResearchModel) -> dict[str, Any]:
    return {
        "dependent_variable": model.dependent_variable,
        "independent_variables": list(model.independent_variables),
        "mediators": list(model.mediators),
        "moderators": list(model.moderators),
        "controls": list(model.controls),
        "clusters": list(model.clusters),
        "weights": list(model.weights),
        "variable_matches": [
            {
                "variable_name": item.variable_name,
                "role": item.role,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in model.variable_matches
        ],
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "focal_variables": list(item.focal_variables),
                "expected_direction": item.expected_direction,
            }
            for item in model.hypotheses
        ],
        "model_rationale": model.model_rationale,
        "confidence": model.confidence,
        "requires_human_review": model.requires_human_review,
    }


def _all_model_variables(model: AgentResearchModel) -> dict[str, list[str]]:
    return {
        "dependent": [model.dependent_variable] if model.dependent_variable else [],
        "independent": list(model.independent_variables),
        "mediator": list(model.mediators),
        "moderator": list(model.moderators),
        "control": list(model.controls),
        "cluster": list(model.clusters),
        "weight": list(model.weights),
    }


def validate_agent_research_model(
    model: AgentResearchModel,
    variable_map: VariableMap,
) -> AgentResearchModelValidationReport:
    """Validate that an external-agent model can be safely applied."""
    issues: list[AgentResearchModelValidationIssue] = []
    warnings: list[str] = []
    defined = set(variable_map.variables)
    role_variables = _all_model_variables(model)
    referenced = {name for values in role_variables.values() for name in values}
    missing = sorted(referenced - defined)

    issues.append(
        AgentResearchModelValidationIssue(
            item="referenced_variables",
            passed=not missing,
            evidence="missing: " + ", ".join(missing) if missing else f"{len(referenced)} variables referenced",
            suggestion="Use only variables from the generated research_context_packet.json.",
        )
    )
    issues.append(
        AgentResearchModelValidationIssue(
            item="dependent_variable",
            passed=bool(model.dependent_variable),
            evidence=model.dependent_variable or "not provided",
            suggestion="Choose exactly one outcome variable.",
        )
    )
    predictor_count = len(model.independent_variables) + len(model.mediators) + len(model.moderators) + len(model.controls)
    issues.append(
        AgentResearchModelValidationIssue(
            item="predictors",
            passed=predictor_count > 0,
            evidence=f"{predictor_count} predictor-side variables",
            suggestion="Choose at least one independent, mediator, moderator, or control variable.",
        )
    )

    primary_roles = {role: values for role, values in role_variables.items() if role not in {"cluster", "weight"}}
    occurrences: dict[str, list[str]] = {}
    for role, values in primary_roles.items():
        for value in values:
            occurrences.setdefault(value, []).append(role)
    duplicates = {name: roles for name, roles in occurrences.items() if len(roles) > 1}
    issues.append(
        AgentResearchModelValidationIssue(
            item="role_overlap",
            passed=not duplicates,
            evidence="; ".join(f"{name}: {', '.join(roles)}" for name, roles in duplicates.items())
            if duplicates
            else "no primary role overlaps",
            suggestion="Assign each analytic variable to one primary role.",
        )
    )

    invalid_matches = [item.role for item in model.variable_matches if item.role not in _AGENT_ROLES]
    if invalid_matches:
        warnings.append("Invalid variable_match roles were ignored: " + ", ".join(sorted(set(invalid_matches))))
    if model.requires_human_review:
        warnings.append("Agent marked this model as requiring human review.")

    return AgentResearchModelValidationReport(
        passed=all(issue.passed for issue in issues),
        issues=issues,
        warnings=warnings,
    )


def agent_research_model_validation_to_dataframe(
    report: AgentResearchModelValidationReport,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": item.item,
                "passed": item.passed,
                "evidence": item.evidence,
                "suggestion": item.suggestion,
            }
            for item in report.issues
        ]
    )


def apply_agent_research_model_to_variable_map(
    variable_map: VariableMap,
    model: AgentResearchModel,
) -> VariableMap:
    """Return a copy of the variable map with agent-recommended roles applied."""
    validation = validate_agent_research_model(model, variable_map)
    if not validation.passed:
        failed = [item.item for item in validation.issues if not item.passed]
        raise ValueError("Agent research model did not pass validation: " + ", ".join(failed))

    output = variable_map.model_copy(deep=True)
    role_targets: dict[str, str] = {}
    for role, values in _all_model_variables(model).items():
        for variable in values:
            role_targets[variable] = role

    for name, definition in output.variables.items():
        if name in role_targets:
            definition.role = role_targets[name]  # type: ignore[assignment]
            definition.review_status = "agent_recommended"
            definition.evidence["agent_research_model_role"] = role_targets[name]
            definition.notes = "Role was recommended by an external research-design agent."
        elif definition.role in {"dependent", "mediator", "moderator", "cluster", "weight"}:
            definition.role = "other"  # type: ignore[assignment]
            definition.evidence["agent_research_model_displaced"] = True
    return output


def apply_agent_research_model_to_analysis_plan(
    analysis_plan: AnalysisPlan,
    model: AgentResearchModel,
    variable_map: VariableMap,
) -> AnalysisPlan:
    """Return a copy of the analysis plan with validated agent roles applied."""
    validation = validate_agent_research_model(model, variable_map)
    if not validation.passed:
        failed = [item.item for item in validation.issues if not item.passed]
        raise ValueError("Agent research model did not pass validation: " + ", ".join(failed))

    output = analysis_plan.model_copy(deep=True)
    output.variables.dependent = [model.dependent_variable] if model.dependent_variable else []
    output.variables.independent = list(model.independent_variables)
    output.variables.mediators = list(model.mediators)
    output.variables.moderators = list(model.moderators)
    output.variables.controls = list(model.controls)
    output.variables.clusters = list(model.clusters)
    output.variables.weights = list(model.weights)
    output.analyses.regression.enabled = bool(output.variables.dependent and output.variables.independent)
    output.analyses.mediation.enabled = bool(output.variables.mediators)
    output.analyses.moderation.enabled = bool(output.variables.moderators)
    if model.hypotheses:
        output.analyses.regression.options["agent_hypotheses"] = [
            {
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "focal_variables": list(item.focal_variables),
                "expected_direction": item.expected_direction,
            }
            for item in model.hypotheses
        ]
    output.analyses.regression.options["agent_model_rationale"] = model.model_rationale
    output.analyses.regression.options["agent_confidence"] = model.confidence
    return output


class AutoResearchIntentAgentStep(PipelineStep):
    """Generate Claude-ready research-agent inputs and optionally apply an agent model."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        research_intent_file: str | Path | None = None,
        research_intent_text: str | None = None,
        agent_research_model_file: str | Path | None = None,
        apply_agent_model: bool = True,
    ) -> None:
        super().__init__(name="03b_auto_research_intent_agent", order=35, required=False)
        self.runtime = runtime
        self.research_intent_file = Path(research_intent_file) if research_intent_file is not None else None
        self.research_intent_text = research_intent_text
        self.agent_research_model_file = Path(agent_research_model_file) if agent_research_model_file is not None else None
        self.apply_agent_model = apply_agent_model

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        output_dir = Path(working_directory) / "result" / "03_auto_plan" / "research_agent"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files: list[str] = []
        warnings: list[str] = []

        variable_map = self.runtime.get_artifact("auto_variable_map")
        if not isinstance(variable_map, VariableMap):
            raise TypeError("auto_variable_map must be a VariableMap before research-agent context is built.")

        if self.research_intent_file is not None:
            intent = load_research_intent(self.research_intent_file)
        else:
            intent = ResearchIntent(raw_text=(self.research_intent_text or "").strip())

        template_path = write_research_intent_template(output_dir / "research_intent_template.yaml")
        output_files.append(str(template_path))
        quality_report = self.runtime.artifacts.get("auto_rawdata_quality_report")
        structured_intent = infer_research_intent_structure(intent)
        packet = build_research_context_packet(intent, variable_map, quality_report=quality_report)
        packet_path = write_research_context_packet(packet, output_dir / "research_context_packet.json")
        prompt_path = write_claude_research_model_prompt(packet, output_dir / "claude_research_model_prompt.txt")
        output_files.extend([str(packet_path), str(prompt_path)])
        self.runtime.set_artifact("auto_research_intent", intent)
        self.runtime.set_artifact("auto_research_intent_extraction", structured_intent)
        self.runtime.set_artifact("auto_research_context_packet", packet)
        self.runtime.set_artifact("auto_claude_research_model_prompt", prompt_path.read_text(encoding="utf-8"))

        metadata: dict[str, Any] = {
            "available_variable_count": len(variable_map.variables),
            "has_research_intent": bool(intent.raw_text or intent.research_topic or intent.research_goal),
            "research_question_count": len(structured_intent.research_questions),
            "hypothesis_candidate_count": len(structured_intent.hypothesis_candidates),
            "agent_model_applied": False,
        }

        if self.agent_research_model_file is not None:
            agent_model = load_agent_research_model(self.agent_research_model_file)
            validation = validate_agent_research_model(agent_model, variable_map)
            validation_path = output_dir / "agent_research_model_validation.xlsx"
            agent_research_model_validation_to_dataframe(validation).to_excel(validation_path, index=False)
            output_files.append(str(validation_path))
            self.runtime.set_artifact("auto_agent_research_model", agent_model)
            self.runtime.set_artifact("auto_agent_research_model_validation", validation)
            if validation.warnings:
                warnings.extend(validation.warnings)
            if not validation.passed:
                warnings.extend(item.evidence for item in validation.issues if not item.passed)
                return StepResult(
                    stage_name=self.name,
                    success=False,
                    output_files=output_files,
                    warnings=warnings,
                    metadata={**metadata, "validation_passed": False},
                )
            if self.apply_agent_model:
                updated_map = apply_agent_research_model_to_variable_map(variable_map, agent_model)
                self.runtime.set_artifact("auto_variable_map", updated_map)
                map_path = output_dir / "agent_variable_map.yaml"
                standard_map_path = output_dir.parent / "auto_variable_map.yaml"
                for target_path in [map_path, standard_map_path]:
                    with target_path.open("w", encoding="utf-8") as file:
                        yaml.safe_dump(updated_map.model_dump(mode="json"), file, allow_unicode=True, sort_keys=False)
                output_files.append(str(map_path))

                analysis_plan = self.runtime.artifacts.get("auto_analysis_plan")
                if isinstance(analysis_plan, AnalysisPlan):
                    updated_plan = apply_agent_research_model_to_analysis_plan(analysis_plan, agent_model, updated_map)
                    self.runtime.set_artifact("auto_analysis_plan", updated_plan)
                    plan_path = output_dir / "agent_analysis_plan.yaml"
                    standard_plan_path = output_dir.parent / "auto_analysis_plan.yaml"
                    for target_path in [plan_path, standard_plan_path]:
                        with target_path.open("w", encoding="utf-8") as file:
                            yaml.safe_dump(updated_plan.model_dump(mode="json"), file, allow_unicode=True, sort_keys=False)
                    output_files.append(str(plan_path))
                metadata["agent_model_applied"] = True
                metadata["standard_config_updated"] = True
                metadata["validation_passed"] = True

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata=metadata,
        )
