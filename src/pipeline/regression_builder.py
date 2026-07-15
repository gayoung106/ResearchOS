"""회귀분석 관련 전체 단계를 조건부 등록하는 빌더."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.advanced_robustness_step import (
    AdvancedOLSRobustnessStep,
)
from src.pipeline.effect_size_step import RegressionEffectSizeStep
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_diagnostics_step import (
    RegressionDiagnosticsStep,
)
from src.pipeline.regression_reporting_step import (
    RegressionReportingStep,
)
from src.pipeline.regression_step import RegressionAnalysisStep
from src.pipeline.regression_visualization_step import (
    RegressionVisualizationStep,
)
from src.pipeline.research_audit_step import ResearchAuditStep
from src.pipeline.robustness_step import OLSRobustnessStep
from src.pipeline.runtime import PipelineRuntime


@dataclass(slots=True)
class RegressionRegistration:
    """회귀 파이프라인 등록 결과."""

    registered: bool
    model_id: str | None
    model_type: str | None
    measurement_level: str | None
    dependent_variable: str | None
    independent_variables: list[str]
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
    groups = analysis_plan.variables

    return list(
        dict.fromkeys(
            groups.independent
            + groups.mediators
            + groups.moderators
            + groups.controls
            + groups.fixed_effects
        )
    )


def _model_type_for_level(
    measurement_level: str,
) -> str | None:
    return {
        "continuous": "ols",
        "binary": "binary_logit",
        "ordinal": "ordered_logit",
        "scale_item": "ordered_logit",
    }.get(measurement_level)


def _robustness_options(
    analysis_plan: AnalysisPlan,
) -> dict[str, Any]:
    robustness = analysis_plan.analyses.robustness
    options = getattr(robustness, "options", None)

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
    """
    설정에 따라 회귀분석 관련 전체 단계를 등록한다.

    공통:
    - 09 회귀분석
    - 13 효과크기
    - 14 논문용 보고서
    - 15 시각화
    - 16 연구 품질 감사

    OLS 추가:
    - 10 회귀진단
    - 11 HC0~HC3 강건성
    - 12 부트스트랩·잭나이프·군집강건
    """

    warnings: list[str] = []

    def not_registered(
        message: str,
        *,
        dependent_variable: str | None = None,
        independent_variables: list[str] | None = None,
        measurement_level: str | None = None,
    ) -> RegressionRegistration:
        return RegressionRegistration(
            registered=False,
            model_id=None,
            model_type=None,
            measurement_level=measurement_level,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables or [],
            diagnostics_registered=False,
            robustness_registered=False,
            advanced_robustness_registered=False,
            effect_size_registered=False,
            reporting_registered=False,
            visualization_registered=False,
            audit_registered=False,
            warnings=[message],
        )

    if not analysis_plan.analyses.regression.enabled:
        return not_registered("회귀분석 설정이 비활성화되어 있습니다.")

    dependent_variables = analysis_plan.variables.dependent

    if not dependent_variables:
        return not_registered("종속변수가 지정되지 않았습니다.")

    if len(dependent_variables) > 1:
        return not_registered("현재 기본 회귀 빌더는 종속변수 1개만 지원합니다.")

    dependent_variable = dependent_variables[0]
    independent_variables = _collect_predictors(analysis_plan)

    if not independent_variables:
        return not_registered(
            "회귀분석에 사용할 독립변수가 없습니다.",
            dependent_variable=dependent_variable,
        )

    measurement_level = _resolve_dependent_measurement_level(
        dependent_variable,
        variable_map,
    )
    model_type = _model_type_for_level(measurement_level)

    if model_type is None:
        return not_registered(
            f"지원되지 않거나 미확정인 종속변수 측정수준입니다: {measurement_level}",
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            measurement_level=measurement_level,
        )

    orchestrator.register(
        RegressionAnalysisStep(
            runtime,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            measurement_level=measurement_level,
            model_id=model_id,
            order=regression_order,
        )
    )

    diagnostics_registered = False
    robustness_registered = False
    advanced_robustness_registered = False

    if model_type == "ols":
        orchestrator.register(
            RegressionDiagnosticsStep(
                runtime,
                model_id=model_id,
                order=diagnostics_order,
            )
        )
        diagnostics_registered = True

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
            run_advanced = bool(options.get("advanced_enabled", True))

            if run_advanced:
                bootstrap_replications = int(
                    options.get(
                        "bootstrap_replications",
                        2000,
                    )
                )
                run_jackknife = bool(options.get("run_jackknife", True))
                cluster_variable = _resolve_cluster_variable(
                    analysis_plan,
                    options,
                )

                orchestrator.register(
                    AdvancedOLSRobustnessStep(
                        runtime,
                        model_id=model_id,
                        cluster_variable=cluster_variable,
                        bootstrap_replications=(bootstrap_replications),
                        run_jackknife=run_jackknife,
                        order=advanced_robustness_order,
                    )
                )
                advanced_robustness_registered = True
            else:
                warnings.append("고급 강건성 분석이 비활성화되어 있습니다.")
        else:
            warnings.append("강건성 분석 설정이 비활성화되어 있습니다.")
    else:
        warnings.append("현재 자동 진단·강건성 단계는 OLS 모형만 지원합니다.")

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
        measurement_level=measurement_level,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        diagnostics_registered=diagnostics_registered,
        robustness_registered=robustness_registered,
        advanced_robustness_registered=(advanced_robustness_registered),
        effect_size_registered=True,
        reporting_registered=True,
        visualization_registered=True,
        audit_registered=True,
        warnings=warnings,
    )
