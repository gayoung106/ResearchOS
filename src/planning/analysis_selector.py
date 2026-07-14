"""연구설계와 변수정보를 이용한 분석방법 추천기."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.common.config_models import AnalysisPlan, ResearchPlan, VariableMap
from src.planning.knowledge_base import (
    get_analysis_knowledge,
    get_methods_for_outcome,
)


@dataclass(slots=True)
class AnalysisRecommendation:
    """개별 분석방법 추천 결과."""

    method_id: str
    korean_name: str
    priority: str
    reason: str
    required_diagnostics: list[str] = field(default_factory=list)
    common_outputs: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    additional_requirements: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RecommendationResult:
    """전체 분석추천 결과."""

    dependent_variable: str | None
    measurement_level: str
    recommendations: list[AnalysisRecommendation]
    warnings: list[str]


def get_dependent_measurement_level(
    dependent_variable: str,
    variable_map: VariableMap,
) -> str:
    """variable_map에서 종속변수 측정수준을 조회한다."""
    definition = variable_map.variables.get(dependent_variable)

    if definition is None:
        return "unknown"

    return definition.measurement_level


def recommend_analysis_methods(
    research_plan: ResearchPlan,
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
) -> RecommendationResult:
    """
    종속변수 측정수준과 자료구조를 기준으로 분석방법을 추천한다.

    이 함수는 분석을 확정하지 않고 후보와 경고를 반환한다.
    """
    warnings: list[str] = []
    recommendations: list[AnalysisRecommendation] = []
    dependent_variables = analysis_plan.variables.dependent

    if not dependent_variables:
        return RecommendationResult(
            dependent_variable=None,
            measurement_level="unknown",
            recommendations=[],
            warnings=["종속변수가 지정되지 않았습니다."],
        )

    if len(dependent_variables) > 1:
        warnings.append("종속변수가 여러 개입니다. 현재 추천은 첫 번째 종속변수를 기준으로 합니다.")

    dependent_variable = dependent_variables[0]
    measurement_level = get_dependent_measurement_level(
        dependent_variable,
        variable_map,
    )

    if measurement_level == "unknown":
        warnings.append(f"{dependent_variable}의 측정수준이 확인되지 않았습니다.")

    for method_id in get_methods_for_outcome(measurement_level):
        knowledge = get_analysis_knowledge(method_id)
        recommendations.append(
            AnalysisRecommendation(
                method_id=knowledge.method_id,
                korean_name=knowledge.korean_name,
                priority="primary",
                reason=(
                    f"종속변수 {dependent_variable}의 측정수준이 "
                    f"{measurement_level}이므로 기본 후보로 추천합니다."
                ),
                required_diagnostics=list(knowledge.required_diagnostics),
                common_outputs=list(knowledge.common_outputs),
                cautions=list(knowledge.cautions),
            )
        )

    cluster_variable = research_plan.data.cluster_variable or (
        analysis_plan.variables.clusters[0] if analysis_plan.variables.clusters else None
    )

    if cluster_variable:
        for recommendation in recommendations:
            recommendation.additional_requirements.append(
                f"군집변수 {cluster_variable}를 고려한 군집강건표준오차 또는 다층모형 검토"
            )

    if research_plan.data.weight_variable or analysis_plan.variables.weights:
        for recommendation in recommendations:
            recommendation.additional_requirements.append(
                "표본가중치 적용 여부와 가중·비가중 결과 비교"
            )

    if research_plan.data.id_variable and research_plan.data.time_variable:
        warnings.append(
            "ID 변수와 시점 변수가 모두 지정되어 패널자료 가능성이 있습니다. "
            "일반 횡단면 모형 외에 패널모형을 검토해야 합니다."
        )

    if analysis_plan.variables.mediators:
        warnings.append(
            "매개변수가 지정되어 있습니다. 기본 회귀모형 외에 "
            "부트스트랩 간접효과 분석을 검토해야 합니다."
        )

    if analysis_plan.variables.moderators:
        warnings.append(
            "조절변수가 지정되어 있습니다. 상호작용항과 "
            "예측값 또는 단순기울기 분석을 검토해야 합니다."
        )

    if research_plan.design.causal_claim_allowed is False:
        for recommendation in recommendations:
            recommendation.cautions.append("현재 설정에서는 인과관계를 단정하지 않습니다.")

    return RecommendationResult(
        dependent_variable=dependent_variable,
        measurement_level=measurement_level,
        recommendations=recommendations,
        warnings=warnings,
    )


def recommendations_to_dataframe(
    result: RecommendationResult,
) -> pd.DataFrame:
    """추천 결과를 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for recommendation in result.recommendations:
        row = asdict(recommendation)
        row["dependent_variable"] = result.dependent_variable
        row["measurement_level"] = result.measurement_level
        row["required_diagnostics"] = " | ".join(recommendation.required_diagnostics)
        row["common_outputs"] = " | ".join(recommendation.common_outputs)
        row["cautions"] = " | ".join(recommendation.cautions)
        row["additional_requirements"] = " | ".join(recommendation.additional_requirements)
        rows.append(row)

    return pd.DataFrame(rows)


def recommendation_summary(
    result: RecommendationResult,
) -> dict[str, Any]:
    """추천 결과 요약을 반환한다."""
    return {
        "dependent_variable": result.dependent_variable,
        "measurement_level": result.measurement_level,
        "recommended_methods": [
            recommendation.method_id for recommendation in result.recommendations
        ],
        "warning_count": len(result.warnings),
        "warnings": result.warnings,
    }
