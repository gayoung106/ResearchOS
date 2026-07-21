"""?뚭?遺꾩꽍 愿???꾩껜 ?④퀎瑜?議곌굔遺 ?깅줉?섎뒗 鍮뚮뜑."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.config_models import (
    AnalysisPlan,
    VariableMap,
)
from src.pipeline.advanced_robustness_step import (
    AdvancedGLMMRobustnessStep,
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
        "nominal": "multinomial_logit",
        "count": "count_auto",
        "proportion": "fractional_logit",
    }.get(measurement_level)


def _regression_options(
    analysis_plan: AnalysisPlan,
) -> dict[str, Any]:
    options = analysis_plan.analyses.regression.options
    return options if isinstance(options, dict) else {}


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

    regression_options = _regression_options(analysis_plan)
    requested_model_type = str(regression_options.get("model_type", "")).strip().lower()
    requested_estimator = str(regression_options.get("estimator", "")).strip().lower()
    log_binomial_requested = requested_estimator in {
        "log_binomial",
        "log-binomial",
        "risk_ratio",
        "binomial_log",
    } or requested_model_type in {
        "log_binomial",
        "log-binomial",
        "risk_ratio",
        "binomial_log",
    }
    wls_requested = requested_estimator in {
        "wls",
        "weighted",
        "weighted_least_squares",
    } or requested_model_type in {
        "wls",
        "weighted",
        "weighted_least_squares",
    }
    cloglog_requested = requested_estimator in {
        "cloglog",
        "binary_cloglog",
        "complementary_log_log",
        "complementary-log-log",
    } or requested_model_type in {
        "cloglog",
        "binary_cloglog",
        "complementary_log_log",
        "complementary-log-log",
    }
    ordered_probit_requested = requested_estimator in {
        "ordered_probit",
        "ordinal_probit",
        "probit_ordinal",
    } or requested_model_type in {
        "ordered_probit",
        "ordinal_probit",
        "probit_ordinal",
    }
    probit_requested = requested_estimator in {
        "probit",
        "binary_probit",
        "probit_regression",
    } or requested_model_type in {
        "probit",
        "binary_probit",
        "probit_regression",
    }
    heckman_requested = requested_estimator in {
        "heckman",
        "heckman_selection",
        "sample_selection",
    } or requested_model_type in {
        "heckman",
        "heckman_selection",
        "sample_selection",
    }
    iv_requested = requested_estimator in {
        "iv",
        "2sls",
        "iv_2sls",
        "iv_2sls_regression",
    } or requested_model_type in {
        "iv",
        "2sls",
        "iv_2sls",
        "iv_2sls_regression",
    }
    inverse_gaussian_requested = requested_estimator in {
        "inverse_gaussian",
        "inverse-gaussian",
        "inverse_gaussian_regression",
    } or requested_model_type in {
        "inverse_gaussian",
        "inverse-gaussian",
        "inverse_gaussian_regression",
    }
    gamma_requested = requested_estimator in {"gamma", "gamma_regression"} or requested_model_type in {
        "gamma",
        "gamma_regression",
    }
    regularized_requested = requested_estimator in {
        "regularized",
        "regularized_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "elasticnet",
    } or requested_model_type in {
        "regularized",
        "regularized_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "elasticnet",
    }
    robust_requested = requested_estimator in {"robust", "rlm", "robust_regression"} or requested_model_type in {
        "robust",
        "rlm",
        "robust_regression",
    }
    tobit_requested = requested_estimator in {"tobit", "censored", "tobit_regression"} or requested_model_type in {
        "tobit",
        "censored",
        "tobit_regression",
    }
    panel_fe_requested = requested_estimator in {"panel_fe", "fixed_effects", "panel_fixed_effects"} or requested_model_type in {
        "panel_fe",
        "fixed_effects",
        "panel_fixed_effects",
    }
    beta_requested = requested_estimator in {"beta", "beta_regression"} or requested_model_type in {
        "beta",
        "beta_regression",
    }
    cox_requested = requested_estimator in {"cox", "cox_ph", "cox_proportional_hazards"} or requested_model_type in {
        "cox",
        "cox_ph",
        "cox_proportional_hazards",
    }
    aft_requested = requested_estimator in {"aft", "weibull_aft", "weibull-aft"} or requested_model_type in {
        "aft",
        "weibull_aft",
        "weibull-aft",
    }
    exponential_aft_requested = requested_estimator in {
        "exponential",
        "exponential_aft",
        "exponential-aft",
    } or requested_model_type in {
        "exponential",
        "exponential_aft",
        "exponential-aft",
    }
    lognormal_aft_requested = requested_estimator in {
        "lognormal_aft",
        "lognormal-aft",
        "log_normal_aft",
        "log-normal-aft",
    } or requested_model_type in {
        "lognormal_aft",
        "lognormal-aft",
        "log_normal_aft",
        "log-normal-aft",
    }
    loglogistic_aft_requested = requested_estimator in {
        "loglogistic_aft",
        "loglogistic-aft",
        "log_logistic_aft",
        "log-logistic-aft",
    } or requested_model_type in {
        "loglogistic_aft",
        "loglogistic-aft",
        "log_logistic_aft",
        "log-logistic-aft",
    }
    quantile_requested = requested_estimator in {"quantile", "quantile_regression"} or requested_model_type in {
        "quantile",
        "quantile_regression",
    }
    gee_requested = requested_estimator == "gee" or requested_model_type in {
        "gee",
        "gee_gaussian",
        "gee_logit",
        "gee_poisson",
    }
    multilevel = analysis_plan.analyses.multilevel
    multilevel_options = _multilevel_options(analysis_plan)
    group_variable = None

    if log_binomial_requested:
        if measurement_level != "binary":
            return not_registered(
                "Log-binomial regression supports binary dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "log_binomial"
        multilevel_options = {
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif wls_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Weighted least squares supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        weight_variable = str(regression_options.get("weight_variable", regression_options.get("weights", ""))).strip()
        if not weight_variable:
            return not_registered(
                "Weighted least squares requires regression.options.weight_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if weight_variable not in variable_map.variables:
            return not_registered(
                "WLS weight variable is missing from variable_map: " + weight_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "weighted_least_squares"
        multilevel_options = {
            "weight_variable": weight_variable,
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
        }
    elif cloglog_requested:
        if measurement_level != "binary":
            return not_registered(
                "Binary cloglog supports binary dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "binary_cloglog"
        multilevel_options = {
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif ordered_probit_requested:
        if measurement_level not in {"ordinal", "scale_item"}:
            return not_registered(
                "Ordered probit supports ordinal dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "ordered_probit"
        multilevel_options = {
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 200)
            ),
        }
    elif probit_requested:
        if measurement_level != "binary":
            return not_registered(
                "Binary probit supports binary dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "binary_probit"
        multilevel_options = {
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif heckman_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Heckman selection supports continuous observed outcomes.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        selection_variable = str(
            regression_options.get("selection_variable", regression_options.get("selected_variable", ""))
        ).strip()
        raw_selection_variables = regression_options.get(
            "selection_variables", regression_options.get("selection_predictors", [])
        )
        selection_variables = [str(value) for value in raw_selection_variables] if isinstance(raw_selection_variables, (list, tuple)) else [str(raw_selection_variables)] if raw_selection_variables else []
        if not selection_variable or not selection_variables:
            return not_registered(
                "Heckman selection requires selection_variable and selection_variables options.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        missing_selection_variables = [
            variable
            for variable in [selection_variable, *selection_variables]
            if variable not in variable_map.variables
        ]
        if missing_selection_variables:
            return not_registered(
                "Heckman variable is missing from variable_map: " + ", ".join(missing_selection_variables),
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "heckman_selection"
        multilevel_options = {
            "selection_variable": selection_variable,
            "selection_variables": selection_variables,
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif iv_requested:
        if measurement_level != "continuous":
            return not_registered(
                "IV 2SLS regression supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        endogenous = regression_options.get("endogenous_variables", regression_options.get("endogenous", []))
        instruments = regression_options.get("instrument_variables", regression_options.get("instruments", []))
        endogenous_variables = [str(value) for value in endogenous] if isinstance(endogenous, (list, tuple)) else [str(endogenous)] if endogenous else []
        instrument_variables = [str(value) for value in instruments] if isinstance(instruments, (list, tuple)) else [str(instruments)] if instruments else []
        if not endogenous_variables or not instrument_variables:
            return not_registered(
                "IV 2SLS requires endogenous_variables and instrument_variables options.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        missing_iv_variables = [
            variable
            for variable in [*endogenous_variables, *instrument_variables]
            if variable not in variable_map.variables
        ]
        if missing_iv_variables:
            return not_registered(
                "IV variable is missing from variable_map: " + ", ".join(missing_iv_variables),
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "iv_2sls_regression"
        multilevel_options = {
            "endogenous_variables": endogenous_variables,
            "instrument_variables": instrument_variables,
            "add_intercept": regression_options.get("add_intercept", True),
        }
    elif inverse_gaussian_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Inverse Gaussian regression supports strictly positive continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "inverse_gaussian_regression"
        multilevel_options = {
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif gamma_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Gamma regression supports strictly positive continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "gamma_regression"
        multilevel_options = {
            "covariance_type": regression_options.get("covariance_type", "HC3"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif regularized_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Regularized regression supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        penalty = regression_options.get("penalty", regression_options.get("regularization", requested_estimator or requested_model_type or "elastic_net"))
        if str(penalty).lower() in {"regularized", "regularized_regression"}:
            penalty = "elastic_net"
        model_type = "regularized_regression"
        multilevel_options = {
            "penalty": penalty,
            "alpha": regression_options.get("alpha", regression_options.get("lambda", 0.1)),
            "l1_ratio": regression_options.get("l1_ratio", 0.5),
            "add_intercept": regression_options.get("add_intercept", True),
            "standardize": regression_options.get("standardize", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 1000)
            ),
        }
    elif robust_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Robust regression supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "robust_regression"
        multilevel_options = {
            "norm": regression_options.get("norm", regression_options.get("robust_norm", "huber")),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif tobit_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Tobit regression supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        lower_limit = regression_options.get("lower_limit")
        upper_limit = regression_options.get("upper_limit")
        if lower_limit is None and upper_limit is None:
            return not_registered(
                "Tobit regression requires lower_limit, upper_limit, or both.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "tobit_regression"
        multilevel_options = {
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 300)
            ),
        }
    elif panel_fe_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Panel fixed effects supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        panel_options = analysis_plan.analyses.panel.options
        panel_options = panel_options if isinstance(panel_options, dict) else {}
        entity_variable = str(
            regression_options.get(
                "entity_variable",
                regression_options.get("id_variable", panel_options.get("entity_variable", panel_options.get("id_variable", ""))),
            )
        ).strip()
        time_variable = str(
            regression_options.get("time_variable", panel_options.get("time_variable", ""))
        ).strip()
        if not entity_variable:
            return not_registered(
                "Panel fixed effects requires entity_variable or id_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        missing_panel_variables = [
            variable for variable in [entity_variable, time_variable] if variable and variable not in variable_map.variables
        ]
        if missing_panel_variables:
            return not_registered(
                "Panel variable is missing from variable_map: " + ", ".join(missing_panel_variables),
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "panel_fixed_effects"
        multilevel_options = {
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "covariance_type": regression_options.get("covariance_type", "cluster_entity"),
        }
    elif beta_requested:
        if measurement_level != "proportion":
            return not_registered(
                "Beta regression supports proportion dependent variables strictly inside (0, 1).",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "beta_regression"
        multilevel_options = {
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }
    elif cox_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Cox proportional hazards supports continuous duration variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        event_variable = str(regression_options.get("event_variable", "")).strip()
        if not event_variable:
            return not_registered(
                "Cox proportional hazards requires regression.options.event_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if event_variable not in variable_map.variables:
            return not_registered(
                "Cox event variable is missing from variable_map: " + event_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "cox_proportional_hazards"
        multilevel_options = {
            "event_variable": event_variable,
            "ties": regression_options.get("ties", "breslow"),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 100)
            ),
        }




    elif exponential_aft_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Exponential AFT regression supports positive continuous duration outcomes.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        event_variable = str(regression_options.get("event_variable", "")).strip()
        if not event_variable:
            return not_registered(
                "Exponential AFT regression requires regression.options.event_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if event_variable not in variable_map.variables:
            return not_registered(
                "Exponential AFT event variable is missing from variable_map: " + event_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "exponential_aft"
        multilevel_options = {
            "event_variable": event_variable,
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 500)
            ),
        }
    elif loglogistic_aft_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Log-logistic AFT regression supports positive continuous duration outcomes.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        event_variable = str(regression_options.get("event_variable", "")).strip()
        if not event_variable:
            return not_registered(
                "Log-logistic AFT regression requires regression.options.event_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if event_variable not in variable_map.variables:
            return not_registered(
                "Log-logistic AFT event variable is missing from variable_map: " + event_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "loglogistic_aft"
        multilevel_options = {
            "event_variable": event_variable,
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 500)
            ),
        }
    elif lognormal_aft_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Log-normal AFT regression supports positive continuous duration outcomes.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        event_variable = str(regression_options.get("event_variable", "")).strip()
        if not event_variable:
            return not_registered(
                "Log-normal AFT regression requires regression.options.event_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if event_variable not in variable_map.variables:
            return not_registered(
                "Log-normal AFT event variable is missing from variable_map: " + event_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "lognormal_aft"
        multilevel_options = {
            "event_variable": event_variable,
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 500)
            ),
        }
    elif aft_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Weibull AFT regression supports positive continuous duration outcomes.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        event_variable = str(regression_options.get("event_variable", "")).strip()
        if not event_variable:
            return not_registered(
                "Weibull AFT regression requires regression.options.event_variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if event_variable not in variable_map.variables:
            return not_registered(
                "Weibull AFT event variable is missing from variable_map: " + event_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "weibull_aft"
        multilevel_options = {
            "event_variable": event_variable,
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 500)
            ),
        }
    elif quantile_requested:
        if measurement_level != "continuous":
            return not_registered(
                "Quantile regression supports continuous dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        model_type = "quantile_regression"
        multilevel_options = {
            "quantile": regression_options.get("quantile", 0.5),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get(
                "max_iterations", regression_options.get("maximum_iterations", 1000)
            ),
        }
    elif gee_requested:
        if requested_model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
            model_type = requested_model_type
        else:
            model_type = (
                "gee_logit"
                if measurement_level == "binary"
                else "gee_poisson"
                if measurement_level == "count"
                else "gee_gaussian"
                if measurement_level == "continuous"
                else None
            )
        if model_type is None:
            return not_registered(
                "GEE supports continuous, binary, or count dependent variables.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        group_variable = _resolve_cluster_variable(analysis_plan, regression_options)
        if group_variable is None:
            return not_registered(
                "GEE requires a cluster/group variable.",
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        if group_variable not in variable_map.variables:
            return not_registered(
                "GEE cluster variable is missing from variable_map: " + group_variable,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                measurement_level=measurement_level,
            )
        multilevel_options = {
            "group_variable": group_variable,
            "covariance_structure": regression_options.get("covariance_structure", "exchangeable"),
            "add_intercept": regression_options.get("add_intercept", True),
            "max_iterations": regression_options.get("max_iterations", regression_options.get("maximum_iterations", 100)),
        }
    elif multilevel.enabled:
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
            run_advanced = bool(options.get("advanced_enabled", True))
            if run_advanced:
                orchestrator.register(
                    AdvancedGLMMRobustnessStep(
                        runtime,
                        model_id=model_id,
                        bootstrap_replications=int(
                            options.get("glmm_bootstrap_replications", 200)
                        ),
                        run_leave_one_group_out=bool(
                            options.get("glmm_leave_one_group_out", True)
                        ),
                        optimizer=optimizers[0] if optimizers else "BFGS",
                        order=advanced_robustness_order,
                    )
                )
                advanced_robustness_registered = True

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
            advanced_robustness_registered=advanced_robustness_registered,
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
        "weighted_least_squares",
        "heckman_selection",
        "iv_2sls_regression",
        "inverse_gaussian_regression",
        "gamma_regression",
        "regularized_regression",
        "robust_regression",
        "tobit_regression",
        "panel_fixed_effects",
        "quantile_regression",
        "cox_proportional_hazards",
        "exponential_aft",
        "loglogistic_aft",
        "lognormal_aft",
        "weibull_aft",
        "fractional_logit",
        "beta_regression",
        "binary_logit",
        "log_binomial",
        "binary_probit",
        "binary_cloglog",
        "ordered_logit",
        "ordered_probit",
        "multinomial_logit",
        "count_auto",
        "gee_gaussian",
        "gee_logit",
        "gee_poisson",
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

