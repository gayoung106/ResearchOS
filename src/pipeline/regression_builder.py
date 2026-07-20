"""?뚭?遺꾩꽍 愿???꾩껜 ?④퀎瑜?議곌굔遺 ?깅줉?섎뒗 鍮뚮뜑."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.config_models import (
    AnalysisPlan,
    VariableMap,
)
from src.pipeline.advanced_robustness_step import (
    AdvancedMixedEffectsRobustnessStep,
    AdvancedOLSRobustnessStep,
)
from src.pipeline.effect_size_step import (
    RegressionEffectSizeStep,
)
from src.pipeline.orchestrator import (
    ResearchOrchestrator,
)
from src.pipeline.regression_diagnostics_step import (
    RegressionDiagnosticsStep,
)
from src.pipeline.regression_reporting_step import (
    RegressionReportingStep,
)
from src.pipeline.regression_step import (
    RegressionAnalysisStep,
)
from src.pipeline.regression_visualization_step import (
    RegressionVisualizationStep,
)
from src.pipeline.research_audit_step import (
    ResearchAuditStep,
)
from src.pipeline.robustness_step import (
    GLMMRobustnessStep,
    MixedEffectsRobustnessStep,
    OLSRobustnessStep,
)
from src.pipeline.runtime import PipelineRuntime


@dataclass(slots=True)
class RegressionRegistration:
    """?뚭? ?뚯씠?꾨씪???깅줉 寃곌낵."""

    registered: bool
    model_id: str | None
    model_type: str | None
    measurement_level: str | None
    dependent_variable: str | None
    independent_variables: list[str]
    fixed_effects: list[str]
    group_variable: str | None
    diagnostics_registered: bool
    robustness_registered: bool
    advanced_robustness_registered: bool
    effect_size_registered: bool
    reporting_registered: bool
    visualization_registered: bool
    audit_registered: bool
    warnings: list[str]


def _resolve_dependent_measurement_level(
    dependent_variable: str,
    variable_map: VariableMap,
) -> str:
    definition = variable_map.variables.get(dependent_variable)

    return definition.measurement_level if definition is not None else "unknown"


def _collect_predictors(
    analysis_plan: AnalysisPlan,
) -> list[str]:
    """怨좎젙?④낵瑜??쒖쇅???쇰컲 ?ㅻ챸蹂?섎? ?섏쭛?쒕떎."""
    groups = analysis_plan.variables

    return list(
        dict.fromkeys(groups.independent + groups.mediators + groups.moderators + groups.controls)
    )


def _collect_fixed_effects(
    analysis_plan: AnalysisPlan,
) -> list[str]:
    """怨좎젙?④낵 蹂?섎? 以묐났 ?놁씠 ?섏쭛?쒕떎."""
    return list(dict.fromkeys(analysis_plan.variables.fixed_effects))


def _model_type_for_level(
    measurement_level: str,
) -> str | None:
    return {
        "continuous": "ols",
        "binary": "binary_logit",
        "ordinal": "ordered_logit",
        "scale_item": "ordered_logit",
        "count": "count_auto",
    }.get(measurement_level)


def _multilevel_options(
    analysis_plan: AnalysisPlan,
) -> dict[str, Any]:
    options = analysis_plan.analyses.multilevel.options
    return options if isinstance(options, dict) else {}


def _resolve_multilevel_group_variable(
    analysis_plan: AnalysisPlan,
    options: dict[str, Any],
) -> str | None:
    configured = options.get("group_variable")

    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    clusters = analysis_plan.variables.clusters
    return clusters[0] if clusters else None


def _robustness_options(
    analysis_plan: AnalysisPlan,
) -> dict[str, Any]:
    robustness = analysis_plan.analyses.robustness
    options = getattr(
        robustness,
        "options",
        None,
    )

    if isinstance(options, dict):
        return options

    return {}


def _resolve_cluster_variable(
    analysis_plan: AnalysisPlan,
    options: dict[str, Any],
) -> str | None:
    configured = options.get("cluster_variable")

    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    clusters = analysis_plan.variables.clusters
    return clusters[0] if clusters else None


def register_regression_pipeline(
    *,
    orchestrator: ResearchOrchestrator,
    runtime: PipelineRuntime,
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
    model_id: str = "main_model",
    regression_order: int = 90,
    diagnostics_order: int = 100,
    robustness_order: int = 110,
    advanced_robustness_order: int = 120,
    effect_size_order: int = 130,
    reporting_order: int = 140,
    visualization_order: int = 150,
    audit_order: int = 160,
) -> RegressionRegistration:
    """?ㅼ젙???곕씪 ?뚭?遺꾩꽍 愿???꾩껜 ?④퀎瑜??깅줉?쒕떎."""
    warnings: list[str] = []

    def not_registered(
        message: str,
        *,
        dependent_variable: (str | None) = None,
        independent_variables: (list[str] | None) = None,
        fixed_effects: (list[str] | None) = None,
        measurement_level: (str | None) = None,
    ) -> RegressionRegistration:
        return RegressionRegistration(
            registered=False,
            model_id=None,
            model_type=None,
            measurement_level=(measurement_level),
            dependent_variable=(dependent_variable),
            independent_variables=(independent_variables or []),
            fixed_effects=(fixed_effects or []),
            group_variable=None,
            diagnostics_registered=False,
            robustness_registered=False,
            advanced_robustness_registered=False,
            effect_size_registered=False,
            reporting_registered=False,
            visualization_registered=False,
            audit_registered=False,
            warnings=[message],
        )

    if not (analysis_plan.analyses.regression.enabled):
        return not_registered("?뚭?遺꾩꽍 ?ㅼ젙??鍮꾪솢?깊솕?섏뼱 ?덉뒿?덈떎.")

    dependent_variables = analysis_plan.variables.dependent

    if not dependent_variables:
        return not_registered("醫낆냽蹂?섍? 吏?뺣릺吏 ?딆븯?듬땲??")

    if len(dependent_variables) > 1:
        return not_registered("?꾩옱 湲곕낯 ?뚭? 鍮뚮뜑??醫낆냽蹂??1媛쒕쭔 吏?먰빀?덈떎.")

    dependent_variable = dependent_variables[0]
    independent_variables = _collect_predictors(analysis_plan)
    fixed_effects = _collect_fixed_effects(analysis_plan)

    if not independent_variables:
        return not_registered(
            "?뚭?遺꾩꽍???ъ슜???낅┰蹂?섍? ?놁뒿?덈떎.",
            dependent_variable=(dependent_variable),
            fixed_effects=fixed_effects,
        )

    duplicated = [variable for variable in fixed_effects if variable in independent_variables]
    if duplicated:
        return not_registered(
            "怨좎젙?④낵 蹂?섍? ?쇰컲 ?ㅻ챸蹂?섏뿉??以묐났 吏?뺣릺?덉뒿?덈떎: " + ", ".join(duplicated),
            dependent_variable=(dependent_variable),
            independent_variables=(independent_variables),
            fixed_effects=fixed_effects,
        )

    missing_fixed_effect_definitions = [
        variable for variable in fixed_effects if variable not in variable_map.variables
    ]
    if missing_fixed_effect_definitions:
        return not_registered(
            "怨좎젙?④낵 蹂?섏쓽 variable_map "
            "?뺤쓽媛 ?놁뒿?덈떎: " + ", ".join(missing_fixed_effect_definitions),
            dependent_variable=(dependent_variable),
            independent_variables=(independent_variables),
            fixed_effects=fixed_effects,
        )

    measurement_level = _resolve_dependent_measurement_level(
        dependent_variable,
        variable_map,
    )

    multilevel = analysis_plan.analyses.multilevel
    multilevel_options = _multilevel_options(analysis_plan)
    group_variable = None

    if multilevel.enabled:
        raw_random_slopes = multilevel_options.get("random_slope_variables")
        random_slope_variables = (
            [str(v).strip() for v in raw_random_slopes]
            if isinstance(raw_random_slopes, (list, tuple))
            else []
        )
        random_slope_variables = [v for v in random_slope_variables if v]
        random_slope_variable = str(multilevel_options.get("random_slope_variable", "")).strip()
        if not random_slope_variables and random_slope_variable:
            random_slope_variables = [random_slope_variable]
        level2_group = str(multilevel_options.get("level2_group", "")).strip()
        level3_group = str(multilevel_options.get("level3_group", "")).strip()
        count_distribution = (
            str(
                multilevel_options.get(
                    "count_distribution", multilevel_options.get("family", "poisson")
                )
            )
            .strip()
            .lower()
        )
        if count_distribution in {"nb", "nb2", "negative-binomial"}:
            count_distribution = "negative_binomial"
        if count_distribution not in {"poisson", "negative_binomial"}:
            return not_registered(
                "怨꾩닔???쇳빀紐⑦삎??count_distribution? poisson ?먮뒗 negative_binomial?댁뼱???⑸땲??",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if level2_group or level3_group:
            if not level2_group or not level3_group:
                return not_registered(
                    "3?섏? ?쇳빀?④낵 紐⑦삎?먮뒗 level2_group怨?level3_group??紐⑤몢 吏?뺥빐???⑸땲??",
                    dependent_variable=dependent_variable,
                    independent_variables=independent_variables,
                    fixed_effects=fixed_effects,
                    measurement_level=measurement_level,
                )
            model_type = (
                "mixed_binary_logit_three_level"
                if measurement_level == "binary"
                else "mixed_negative_binomial_three_level"
                if measurement_level == "count" and count_distribution == "negative_binomial"
                else "mixed_poisson_three_level"
                if measurement_level == "count"
                else "mixed_three_level"
            )
        else:
            model_type = (
                "mixed_binary_logit_random_slope"
                if measurement_level == "binary" and random_slope_variables
                else "mixed_binary_logit_random_intercept"
                if measurement_level == "binary"
                else "mixed_negative_binomial_random_slope"
                if measurement_level == "count"
                and count_distribution == "negative_binomial"
                and random_slope_variables
                else "mixed_poisson_random_slope"
                if measurement_level == "count" and random_slope_variables
                else "mixed_negative_binomial_random_intercept"
                if measurement_level == "count" and count_distribution == "negative_binomial"
                else "mixed_poisson_random_intercept"
                if measurement_level == "count"
                else "mixed_random_slope"
                if random_slope_variables
                else "mixed_random_intercept"
            )
        covariance_structure = (
            str(multilevel_options.get("random_effect_covariance", "correlated")).strip().lower()
        )
        if covariance_structure not in {"correlated", "diagonal"}:
            return not_registered(
                "random_effect_covariance??correlated ?먮뒗 diagonal?댁뼱???⑸땲??",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        group_variable = (
            level3_group
            if model_type
            in {"mixed_three_level", "mixed_binary_logit_three_level", "mixed_poisson_three_level", "mixed_negative_binomial_three_level"}
            else _resolve_multilevel_group_variable(analysis_plan, multilevel_options)
        )

        if measurement_level not in {"continuous", "binary", "count"}:
            return not_registered(
                "?쇳빀?④낵 紐⑦삎? ?곗냽?? ?댄빆 ?먮뒗 怨꾩닔??醫낆냽蹂?섎? 吏?먰빀?덈떎.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )

        if group_variable is None:
            return not_registered(
                "?쇳빀?④낵 紐⑦삎??洹몃９蹂?섍? 吏?뺣릺吏 ?딆븯?듬땲??",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )

        required_group_variables = (
            [level2_group, level3_group]
            if model_type
            in {"mixed_three_level", "mixed_binary_logit_three_level", "mixed_poisson_three_level", "mixed_negative_binomial_three_level"}
            else [group_variable]
        )
        missing_group_definitions = [
            variable
            for variable in required_group_variables
            if variable not in variable_map.variables
        ]
        if missing_group_definitions:
            return not_registered(
                "洹몃９蹂?섏쓽 variable_map ?뺤쓽媛 ?놁뒿?덈떎: " + ", ".join(missing_group_definitions),
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )

        if group_variable == dependent_variable or group_variable in independent_variables:
            return not_registered(
                "洹몃９蹂?섎뒗 醫낆냽蹂???먮뒗 ?쇰컲 ?ㅻ챸蹂?섏? 以묐났?????놁뒿?덈떎.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )

        missing_random_slopes = [
            v for v in random_slope_variables if v not in independent_variables
        ]
        if missing_random_slopes:
            return not_registered(
                "Random Slope 蹂?섎뒗 ?쇰컲 ?ㅻ챸蹂?섏뿉 ?ы븿?섏뼱???⑸땲?? "
                + ", ".join(missing_random_slopes),
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )

        if fixed_effects:
            return not_registered(
                "?꾩옱 ?쇳빀?④낵 鍮뚮뜑??蹂꾨룄 怨좎젙?④낵 蹂??吏?뺤쓣 吏?먰븯吏 ?딆뒿?덈떎.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
    else:
        model_type = _model_type_for_level(measurement_level)

    if model_type is None:
        return not_registered(
            f"吏?먮릺吏 ?딄굅??誘명솗?뺤씤 醫낆냽蹂??痢≪젙?섏??낅땲?? {measurement_level}",
            dependent_variable=(dependent_variable),
            independent_variables=(independent_variables),
            fixed_effects=fixed_effects,
            measurement_level=(measurement_level),
        )

    orchestrator.register(
        RegressionAnalysisStep(
            runtime,
            dependent_variable=(dependent_variable),
            independent_variables=(independent_variables),
            measurement_level=(measurement_level),
            fixed_effects=fixed_effects,
            model_id=model_id,
            model_type=model_type,
            group_variable=group_variable,
            mixed_effects_options=multilevel_options,
            order=regression_order,
        )
    )

    diagnostics_registered = False
    robustness_registered = False
    advanced_robustness_registered = False

    if model_type in {
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
        "mixed_poisson_random_intercept",
        "mixed_poisson_random_slope",
        "mixed_poisson_three_level",
        "mixed_negative_binomial_random_intercept",
        "mixed_negative_binomial_random_slope",
        "mixed_negative_binomial_three_level",
    }:
        orchestrator.register(
            RegressionDiagnosticsStep(runtime, model_id=model_id, order=diagnostics_order)
        )
        diagnostics_registered = True
        orchestrator.register(
            RegressionEffectSizeStep(runtime, model_id=model_id, order=effect_size_order)
        )
        robustness = analysis_plan.analyses.robustness
        if robustness.enabled:
            options = _robustness_options(analysis_plan)
            configured_optimizers = options.get(
                "mixed_glmm_optimizers",
                [multilevel_options.get("optimizer", "BFGS")],
            )
            optimizers = tuple(
                str(item).strip() for item in configured_optimizers if str(item).strip()
            )
            orchestrator.register(
                GLMMRobustnessStep(
                    runtime,
                    model_id=model_id,
                    optimizers=optimizers or ("BFGS",),
                    order=robustness_order,
                )
            )
            robustness_registered = True

        orchestrator.register(
            RegressionReportingStep(runtime, model_id=model_id, order=reporting_order)
        )
        orchestrator.register(
            RegressionVisualizationStep(runtime, model_id=model_id, order=visualization_order)
        )
        orchestrator.register(ResearchAuditStep(runtime, model_id=model_id, order=audit_order))
        return RegressionRegistration(
            registered=True,
            model_id=model_id,
            model_type=model_type,
            measurement_level=measurement_level,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            group_variable=group_variable,
            diagnostics_registered=diagnostics_registered,
            robustness_registered=robustness_registered,
            advanced_robustness_registered=False,
            effect_size_registered=True,
            reporting_registered=True,
            visualization_registered=True,
            audit_registered=True,
            warnings=warnings,
        )

    if model_type in {"mixed_random_intercept", "mixed_random_slope", "mixed_three_level"}:
        orchestrator.register(
            RegressionDiagnosticsStep(
                runtime,
                model_id=model_id,
                order=diagnostics_order,
            )
        )
        orchestrator.register(
            RegressionEffectSizeStep(
                runtime,
                model_id=model_id,
                order=effect_size_order,
            )
        )
        orchestrator.register(
            RegressionReportingStep(
                runtime,
                model_id=model_id,
                order=reporting_order,
            )
        )
        orchestrator.register(
            RegressionVisualizationStep(
                runtime,
                model_id=model_id,
                order=visualization_order,
            )
        )
        robustness = analysis_plan.analyses.robustness
        if robustness.enabled:
            options = _robustness_options(analysis_plan)
            configured_optimizers = options.get(
                "mixed_optimizers",
                ["lbfgs", "bfgs", "cg", "powell"],
            )
            optimizers = tuple(
                str(item).strip() for item in configured_optimizers if str(item).strip()
            )
            orchestrator.register(
                MixedEffectsRobustnessStep(
                    runtime,
                    model_id=model_id,
                    optimizers=optimizers,
                    order=robustness_order,
                )
            )
            robustness_registered = True
            run_advanced = bool(options.get("advanced_enabled", True))
            if run_advanced:
                orchestrator.register(
                    AdvancedMixedEffectsRobustnessStep(
                        runtime,
                        model_id=model_id,
                        bootstrap_replications=int(
                            options.get("mixed_bootstrap_replications", 500)
                        ),
                        run_leave_one_group_out=bool(
                            options.get("mixed_run_leave_one_group_out", True)
                        ),
                        order=advanced_robustness_order,
                    )
                )
                advanced_robustness_registered = True

        orchestrator.register(
            ResearchAuditStep(
                runtime,
                model_id=model_id,
                order=audit_order,
            )
        )

        return RegressionRegistration(
            registered=True,
            model_id=model_id,
            model_type=model_type,
            measurement_level=measurement_level,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            group_variable=group_variable,
            diagnostics_registered=True,
            robustness_registered=robustness_registered,
            advanced_robustness_registered=advanced_robustness_registered,
            effect_size_registered=True,
            reporting_registered=True,
            visualization_registered=True,
            audit_registered=True,
            warnings=warnings,
        )

    if model_type in {
        "ols",
        "binary_logit",
        "ordered_logit",
        "count_auto",
    }:
        orchestrator.register(
            RegressionDiagnosticsStep(
                runtime,
                model_id=model_id,
                order=diagnostics_order,
            )
        )
        diagnostics_registered = True
    else:
        warnings.append("?꾩옱 ?먮룞 吏꾨떒 ?④퀎媛 吏?먰븯吏 ?딅뒗 ?뚭?紐⑦삎?낅땲??")

    if model_type == "ols":
        robustness = analysis_plan.analyses.robustness

        if robustness.enabled:
            orchestrator.register(
                OLSRobustnessStep(
                    runtime,
                    model_id=model_id,
                    order=robustness_order,
                )
            )
            robustness_registered = True

            options = _robustness_options(analysis_plan)
            run_advanced = bool(
                options.get(
                    "advanced_enabled",
                    True,
                )
            )

            if run_advanced:
                bootstrap_replications = int(
                    options.get(
                        "bootstrap_replications",
                        2000,
                    )
                )
                run_jackknife = bool(
                    options.get(
                        "run_jackknife",
                        True,
                    )
                )
                cluster_variable = _resolve_cluster_variable(
                    analysis_plan,
                    options,
                )

                orchestrator.register(
                    AdvancedOLSRobustnessStep(
                        runtime,
                        model_id=model_id,
                        cluster_variable=(cluster_variable),
                        bootstrap_replications=(bootstrap_replications),
                        run_jackknife=(run_jackknife),
                        order=(advanced_robustness_order),
                    )
                )
                advanced_robustness_registered = True
            else:
                warnings.append("怨좉툒 媛뺢굔??遺꾩꽍??鍮꾪솢?깊솕?섏뼱 ?덉뒿?덈떎.")
        else:
            warnings.append("媛뺢굔??遺꾩꽍 ?ㅼ젙??鍮꾪솢?깊솕?섏뼱 ?덉뒿?덈떎.")
    else:
        warnings.append("?꾩옱 ?먮룞 媛뺢굔???④퀎??OLS 紐⑦삎留?吏?먰빀?덈떎.")

    orchestrator.register(
        RegressionEffectSizeStep(
            runtime,
            model_id=model_id,
            order=effect_size_order,
        )
    )
    orchestrator.register(
        RegressionReportingStep(
            runtime,
            model_id=model_id,
            order=reporting_order,
        )
    )
    orchestrator.register(
        RegressionVisualizationStep(
            runtime,
            model_id=model_id,
            order=visualization_order,
        )
    )
    orchestrator.register(
        ResearchAuditStep(
            runtime,
            model_id=model_id,
            order=audit_order,
        )
    )

    return RegressionRegistration(
        registered=True,
        model_id=model_id,
        model_type=model_type,
        measurement_level=(measurement_level),
        dependent_variable=(dependent_variable),
        independent_variables=(independent_variables),
        fixed_effects=fixed_effects,
        group_variable=group_variable,
        diagnostics_registered=(diagnostics_registered),
        robustness_registered=(robustness_registered),
        advanced_robustness_registered=(advanced_robustness_registered),
        effect_size_registered=True,
        reporting_registered=True,
        visualization_registered=True,
        audit_registered=True,
        warnings=warnings,
    )

