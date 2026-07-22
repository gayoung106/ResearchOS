"""Automatic multi-outcome planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from src.auto.analysis_plan import AutoAnalysisPlanResult, build_auto_analysis_plan
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult

_ANALYZABLE_OUTCOME_LEVELS = {
    "binary",
    "continuous",
    "count",
    "nominal",
    "ordinal",
    "proportion",
    "scale_item",
}
_SPECIAL_ROLES = {"cluster", "fixed_effect", "id", "strata", "time", "weight"}
_PREDICTOR_CONTEXT_KEYWORDS = {
    "baseline",
    "pre",
    "pretest",
    "predictor",
    "covariate",
    "control",
    "independent",
    "treatment",
    "exposure",
    "\uc0ac\uc804",
    "\uae30\ucd08",
    "\uae30\uc900",
    "\uc608\uce21",
    "\uc608\uce21\ubcc0\uc218",
    "\uacf5\ubcc0\ub7c9",
    "\ud1b5\uc81c",
    "\ub3c5\ub9bd",
    "\ucc98\uce58",
    "\ub178\ucd9c",
}
_OUTCOME_KEYWORDS = {
    "outcome",
    "result",
    "score",
    "total",
    "target",
    "response",
    "post",
    "followup",
    "dependent",
    "satisfaction",
    "performance",
    "retention",
    "\uacb0\uacfc",
    "\uc131\uacfc",
    "\uc810\uc218",
    "\ucd1d\uc810",
    "\ud569\uacc4",
    "\ubaa9\ud45c",
    "\uc751\ub2f5",
    "\uc0ac\ud6c4",
    "\uc885\uc18d",
    "\ub9cc\uc871\ub3c4",
    "\ud6a8\uacfc",
    "\ud3c9\uac00",
    "\uc720\uc9c0",
}


@dataclass(slots=True)
class AutoOutcomeCandidate:
    variable_name: str
    measurement_level: str
    score: float
    reason: str


@dataclass(slots=True)
class AutoOutcomeAnalysisPlan:
    model_id: str
    dependent_variable: str
    analysis_plan: AnalysisPlan
    variable_map: VariableMap
    plan_result: AutoAnalysisPlanResult


@dataclass(slots=True)
class AutoMultiOutcomeAnalysisPlanResult:
    outcome_plans: list[AutoOutcomeAnalysisPlan]
    candidates: list[AutoOutcomeCandidate]
    warnings: list[str] = field(default_factory=list)


def _search_text(variable_map: VariableMap, variable_name: str) -> str:
    definition = variable_map.variables[variable_name]
    evidence_text = str(definition.evidence.get("auto_role_search_text", ""))
    return " ".join(
        [
            variable_name,
            definition.label,
            definition.korean_name,
            definition.question_text,
            evidence_text,
        ]
    ).lower()


def _looks_like_outcome(text: str) -> bool:
    return any(keyword in text for keyword in _OUTCOME_KEYWORDS)


def _looks_like_predictor_context(text: str) -> bool:
    return any(keyword in text for keyword in _PREDICTOR_CONTEXT_KEYWORDS)


def infer_auto_outcome_candidates(
    variable_map: VariableMap,
    *,
    max_outcomes: int | None = None,
) -> list[AutoOutcomeCandidate]:
    candidates: list[AutoOutcomeCandidate] = []
    for variable_name, definition in variable_map.variables.items():
        if definition.role in _SPECIAL_ROLES:
            continue
        if definition.measurement_level not in _ANALYZABLE_OUTCOME_LEVELS:
            continue

        text = _search_text(variable_map, variable_name)
        score = 0.0
        reasons: list[str] = []
        if definition.role == "dependent":
            score += 100.0
            reasons.append("currently inferred as dependent")
        if _looks_like_outcome(text):
            score += 70.0
            reasons.append("name or label suggests an outcome")
        if _looks_like_predictor_context(text):
            score -= 150.0
            reasons.append("name or label suggests a predictor or baseline covariate")
        if definition.measurement_level in {"continuous", "count", "proportion"}:
            score += 20.0
        elif definition.measurement_level in {"binary", "ordinal", "scale_item"}:
            score += 10.0
        confidence = definition.evidence.get("auto_role_confidence")
        if isinstance(confidence, int | float):
            score += float(confidence)

        if score <= 0.0:
            continue
        candidates.append(
            AutoOutcomeCandidate(
                variable_name=variable_name,
                measurement_level=definition.measurement_level,
                score=score,
                reason="; ".join(reasons) or "analyzable measurement level",
            )
        )

    candidates.sort(key=lambda item: -item.score)
    if max_outcomes is not None:
        return candidates[:max_outcomes]
    return candidates


def _variable_map_for_outcome(variable_map: VariableMap, dependent_variable: str) -> VariableMap:
    output = variable_map.model_copy(deep=True)
    for variable_name, definition in output.variables.items():
        if variable_name == dependent_variable:
            definition.role = "dependent"  # type: ignore[assignment]
            definition.evidence["multi_outcome_role"] = "dependent"
            continue
        if definition.role == "dependent":
            definition.role = "independent"  # type: ignore[assignment]
            definition.evidence["multi_outcome_displaced_dependent"] = True
    return output


def build_auto_multi_outcome_analysis_plans(
    variable_map: VariableMap,
    *,
    max_outcomes: int | None = None,
    model_id_prefix: str = "main_model",
    enable_robustness: bool = False,
) -> AutoMultiOutcomeAnalysisPlanResult:
    candidates = infer_auto_outcome_candidates(variable_map, max_outcomes=max_outcomes)
    warnings: list[str] = []
    if not candidates:
        return AutoMultiOutcomeAnalysisPlanResult(
            outcome_plans=[],
            candidates=[],
            warnings=["No analyzable outcome candidates were found."],
        )

    outcome_plans: list[AutoOutcomeAnalysisPlan] = []
    for index, candidate in enumerate(candidates, start=1):
        adjusted_map = _variable_map_for_outcome(variable_map, candidate.variable_name)
        plan_result = build_auto_analysis_plan(
            adjusted_map,
            enable_robustness=enable_robustness,
        )
        warnings.extend(plan_result.warnings)
        outcome_plans.append(
            AutoOutcomeAnalysisPlan(
                model_id=f"{model_id_prefix}_{index}",
                dependent_variable=candidate.variable_name,
                analysis_plan=plan_result.analysis_plan,
                variable_map=adjusted_map,
                plan_result=plan_result,
            )
        )
    return AutoMultiOutcomeAnalysisPlanResult(
        outcome_plans=outcome_plans,
        candidates=candidates,
        warnings=list(dict.fromkeys(warnings)),
    )


def auto_multi_outcome_candidates_to_dataframe(
    candidates: list[AutoOutcomeCandidate],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable_name": item.variable_name,
                "measurement_level": item.measurement_level,
                "score": item.score,
                "reason": item.reason,
            }
            for item in candidates
        ]
    )


def auto_multi_outcome_plans_to_dataframe(
    result: AutoMultiOutcomeAnalysisPlanResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": item.model_id,
                "dependent_variable": item.dependent_variable,
                "independent_variables": " | ".join(item.analysis_plan.variables.independent),
                "control_variables": " | ".join(item.analysis_plan.variables.controls),
                "regression_enabled": item.analysis_plan.analyses.regression.enabled,
                "regression_options": item.analysis_plan.analyses.regression.options,
            }
            for item in result.outcome_plans
        ]
    )


class AutoMultiOutcomeAnalysisPlanStep(PipelineStep):
    """Build one auto analysis plan per candidate outcome."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        max_outcomes: int | None = None,
        model_id_prefix: str = "main_model",
        enable_robustness: bool = False,
        order: int = 35,
    ) -> None:
        super().__init__(name="03b_auto_multi_outcome_plan", order=order, required=False)
        self.runtime = runtime
        self.max_outcomes = max_outcomes
        self.model_id_prefix = model_id_prefix
        self.enable_robustness = enable_robustness

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        try:
            variable_map = self.runtime.get_artifact("auto_variable_map")
        except KeyError:
            variable_map = None
        if not isinstance(variable_map, VariableMap):
            return StepResult(
                stage_name=self.name,
                success=False,
                warnings=["auto_variable_map artifact is required before multi-outcome planning."],
            )

        result = build_auto_multi_outcome_analysis_plans(
            variable_map,
            max_outcomes=self.max_outcomes,
            model_id_prefix=self.model_id_prefix,
            enable_robustness=self.enable_robustness,
        )
        self.runtime.set_artifact("auto_multi_outcome_plan_result", result)

        output_dir = working_directory / "result" / "03_auto_plan" / "multi_outcome"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = output_dir / "outcome_candidates.xlsx"
        plans_path = output_dir / "outcome_analysis_plans.xlsx"
        auto_multi_outcome_candidates_to_dataframe(result.candidates).to_excel(candidates_path, index=False)
        auto_multi_outcome_plans_to_dataframe(result).to_excel(plans_path, index=False)

        for item in result.outcome_plans:
            model_dir = output_dir / item.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            with (model_dir / "analysis_plan.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    item.analysis_plan.model_dump(mode="json"),
                    file,
                    allow_unicode=True,
                    sort_keys=False,
                )
            with (model_dir / "variable_map.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    item.variable_map.model_dump(mode="json"),
                    file,
                    allow_unicode=True,
                    sort_keys=False,
                )

        output_files = [str(candidates_path), str(plans_path)]
        output_files.extend(
            str(output_dir / item.model_id / filename)
            for item in result.outcome_plans
            for filename in ["analysis_plan.yaml", "variable_map.yaml"]
        )
        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=output_files,
            warnings=result.warnings,
            metadata={
                "candidate_count": len(result.candidates),
                "outcome_plan_count": len(result.outcome_plans),
                "model_ids": [item.model_id for item in result.outcome_plans],
            },
        )
