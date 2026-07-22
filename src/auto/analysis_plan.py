"""Automatic analysis-plan generation from inferred variable roles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config_models import AnalysisPlan, VariableDefinition, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult

_ANALYZABLE_DEPENDENT_LEVELS = {
    "binary",
    "continuous",
    "count",
    "nominal",
    "ordinal",
    "proportion",
    "scale_item",
}
_CONTINUOUS_LIKE_LEVELS = {"continuous", "proportion"}


@dataclass(slots=True)
class AutoAnalysisPlanDecision:
    item: str
    selected_value: str | bool | None
    confidence: float
    reason: str


@dataclass(slots=True)
class AutoAnalysisPlanResult:
    analysis_plan: AnalysisPlan
    decisions: list[AutoAnalysisPlanDecision]
    warnings: list[str] = field(default_factory=list)


def _names_by_role(variable_map: VariableMap, *roles: str) -> list[str]:
    role_set = set(roles)
    return [name for name, definition in variable_map.variables.items() if definition.role in role_set]


def _definition(variable_map: VariableMap, variable_name: str | None) -> VariableDefinition | None:
    if variable_name is None:
        return None
    return variable_map.variables.get(variable_name)


def _select_regression_estimator(
    variable_map: VariableMap,
    dependent_variable: str | None,
    *,
    entity_variable: str | None,
    time_variable: str | None,
    weight_variable: str | None,
) -> tuple[str | None, list[AutoAnalysisPlanDecision]]:
    decisions: list[AutoAnalysisPlanDecision] = []
    dependent_definition = _definition(variable_map, dependent_variable)
    dependent_level = dependent_definition.measurement_level if dependent_definition else "unknown"

    if entity_variable and time_variable and dependent_level in _CONTINUOUS_LIKE_LEVELS:
        decisions.append(
            AutoAnalysisPlanDecision(
                "regression_estimator",
                "panel_fe",
                0.8,
                "Entity and time roles support a first-pass panel fixed-effects model.",
            )
        )
        return "panel_fe", decisions

    if weight_variable and dependent_level == "continuous":
        decisions.append(
            AutoAnalysisPlanDecision(
                "regression_estimator",
                "wls",
                0.75,
                "A weight role was inferred for a continuous outcome.",
            )
        )
        return "wls", decisions

    decisions.append(
        AutoAnalysisPlanDecision(
            "regression_estimator",
            None,
            0.65,
            "No explicit estimator was selected; the regression builder will route by outcome measurement level.",
        )
    )
    return None, decisions


def build_auto_analysis_plan(
    variable_map: VariableMap,
    *,
    enable_robustness: bool = False,
) -> AutoAnalysisPlanResult:
    """Create a conservative AnalysisPlan from an auto-inferred VariableMap."""
    dependent_variables = _names_by_role(variable_map, "dependent")
    independent_variables = _names_by_role(variable_map, "independent")
    control_variables = _names_by_role(variable_map, "control")
    fixed_effects = _names_by_role(variable_map, "fixed_effect")
    weights = _names_by_role(variable_map, "weight")
    clusters = _names_by_role(variable_map, "cluster")
    id_variables = _names_by_role(variable_map, "id")
    time_variables = _names_by_role(variable_map, "time")

    warnings: list[str] = []
    decisions: list[AutoAnalysisPlanDecision] = []
    dependent_variable = dependent_variables[0] if len(dependent_variables) == 1 else None
    dependent_definition = _definition(variable_map, dependent_variable)
    dependent_level = dependent_definition.measurement_level if dependent_definition else "unknown"

    if not dependent_variables:
        warnings.append("No dependent variable was inferred; regression is disabled.")
    elif len(dependent_variables) > 1:
        warnings.append("Multiple dependent variables were inferred; regression is disabled until review.")
    elif dependent_level not in _ANALYZABLE_DEPENDENT_LEVELS:
        warnings.append("The inferred dependent variable has an unsupported measurement level.")

    predictor_count = len(independent_variables) + len(control_variables)
    if predictor_count == 0:
        warnings.append("No independent or control variables were inferred; regression is disabled.")

    regression_enabled = (
        dependent_variable is not None
        and dependent_level in _ANALYZABLE_DEPENDENT_LEVELS
        and predictor_count > 0
    )
    decisions.append(
        AutoAnalysisPlanDecision(
            "regression_enabled",
            regression_enabled,
            0.8 if regression_enabled else 0.6,
            "Exactly one analyzable dependent variable and at least one predictor were found."
            if regression_enabled
            else "Regression needs one analyzable dependent variable and at least one predictor.",
        )
    )

    entity_variable = id_variables[0] if id_variables else None
    time_variable = time_variables[0] if time_variables else None
    weight_variable = weights[0] if weights else None
    cluster_variable = clusters[0] if clusters else None
    panel_enabled = bool(entity_variable and time_variable and regression_enabled)

    estimator, estimator_decisions = _select_regression_estimator(
        variable_map,
        dependent_variable,
        entity_variable=entity_variable,
        time_variable=time_variable,
        weight_variable=weight_variable,
    )
    decisions.extend(estimator_decisions)
    decisions.append(
        AutoAnalysisPlanDecision(
            "panel_enabled",
            panel_enabled,
            0.75 if panel_enabled else 0.65,
            "Both entity and time roles were inferred."
            if panel_enabled
            else "Panel analysis requires both entity and time roles plus a valid regression setup.",
        )
    )

    regression_options: dict[str, Any] = {}
    if estimator is not None:
        regression_options["estimator"] = estimator
    if weight_variable and estimator == "wls":
        regression_options["weight_variable"] = weight_variable
    if cluster_variable:
        regression_options["cluster_variable"] = cluster_variable

    panel_options: dict[str, Any] = {}
    if entity_variable:
        panel_options["entity_variable"] = entity_variable
    if time_variable:
        panel_options["time_variable"] = time_variable

    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": dependent_variables,
                "independent": independent_variables,
                "controls": control_variables,
                "fixed_effects": fixed_effects,
                "weights": weights,
                "clusters": clusters,
            },
            "analyses": {
                "regression": {"enabled": regression_enabled, "options": regression_options},
                "panel": {"enabled": panel_enabled, "options": panel_options},
                "robustness": {"enabled": enable_robustness},
            },
            "review": {
                "required_roles": [
                    role
                    for role, present in {
                        "dependent": bool(dependent_variables),
                        "independent": bool(independent_variables or control_variables),
                    }.items()
                    if not present
                ]
            },
        }
    )
    return AutoAnalysisPlanResult(analysis_plan=plan, decisions=decisions, warnings=warnings)


def auto_analysis_plan_summary_to_dataframe(result: AutoAnalysisPlanResult) -> pd.DataFrame:
    plan = result.analysis_plan
    rows: list[dict[str, Any]] = [
        {
            "item": "dependent_variables",
            "value": " | ".join(plan.variables.dependent),
        },
        {
            "item": "independent_variables",
            "value": " | ".join(plan.variables.independent),
        },
        {
            "item": "control_variables",
            "value": " | ".join(plan.variables.controls),
        },
        {
            "item": "fixed_effects",
            "value": " | ".join(plan.variables.fixed_effects),
        },
        {
            "item": "weights",
            "value": " | ".join(plan.variables.weights),
        },
        {
            "item": "clusters",
            "value": " | ".join(plan.variables.clusters),
        },
        {
            "item": "regression_enabled",
            "value": plan.analyses.regression.enabled,
        },
        {
            "item": "regression_options",
            "value": plan.analyses.regression.options,
        },
        {
            "item": "panel_enabled",
            "value": plan.analyses.panel.enabled,
        },
        {
            "item": "panel_options",
            "value": plan.analyses.panel.options,
        },
        {
            "item": "robustness_enabled",
            "value": plan.analyses.robustness.enabled,
        },
    ]
    return pd.DataFrame(rows)


def auto_analysis_plan_decisions_to_dataframe(
    decisions: list[AutoAnalysisPlanDecision],
) -> pd.DataFrame:
    return pd.DataFrame([asdict(decision) for decision in decisions])


class AutoAnalysisPlanStep(PipelineStep):
    """Create an AnalysisPlan from the auto-inferred VariableMap artifact."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        enable_robustness: bool = False,
        order: int = 30,
    ) -> None:
        super().__init__(name="03_auto_analysis_plan", order=order, required=True)
        self.runtime = runtime
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
                warnings=["auto_variable_map artifact is required before auto analysis planning."],
            )

        result = build_auto_analysis_plan(
            variable_map,
            enable_robustness=self.enable_robustness,
        )
        self.runtime.set_artifact("auto_analysis_plan_result", result)
        self.runtime.set_artifact("auto_analysis_plan", result.analysis_plan)

        output_dir = working_directory / "result" / "03_auto_plan"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "analysis_plan_summary.xlsx"
        decisions_path = output_dir / "analysis_plan_decisions.xlsx"
        auto_analysis_plan_summary_to_dataframe(result).to_excel(summary_path, index=False)
        auto_analysis_plan_decisions_to_dataframe(result.decisions).to_excel(decisions_path, index=False)

        plan = result.analysis_plan
        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(summary_path), str(decisions_path)],
            warnings=result.warnings,
            metadata={
                "dependent_variable": plan.variables.dependent[0] if len(plan.variables.dependent) == 1 else None,
                "independent_count": len(plan.variables.independent),
                "control_count": len(plan.variables.controls),
                "regression_enabled": plan.analyses.regression.enabled,
                "regression_estimator": plan.analyses.regression.options.get("estimator"),
                "panel_enabled": plan.analyses.panel.enabled,
                "robustness_enabled": plan.analyses.robustness.enabled,
            },
        )
