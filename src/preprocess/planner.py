"""확정된 변수정보를 바탕으로 전처리 계획을 생성하는 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.common.config_models import AnalysisPlan, VariableMap
from src.preprocess.evidence_resolver import ResolvedVariableLevel


@dataclass(slots=True)
class PreprocessingAction:
    """개별 전처리 작업 계획."""

    variable_name: str
    action_type: str
    status: str
    reason: str
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = True
    priority: str = "normal"


@dataclass(slots=True)
class PreprocessingPlan:
    """전체 전처리 계획."""

    actions: list[PreprocessingAction]
    warnings: list[str]
    blocked_variables: list[str]


def build_role_map(analysis_plan: AnalysisPlan) -> dict[str, str]:
    """분석계획의 변수 역할을 변수명 기준으로 매핑한다."""
    groups = analysis_plan.variables
    role_map: dict[str, str] = {}

    mapping = {
        "dependent": groups.dependent,
        "independent": groups.independent,
        "mediator": groups.mediators,
        "moderator": groups.moderators,
        "control": groups.controls,
        "fixed_effect": groups.fixed_effects,
        "weight": groups.weights,
        "cluster": groups.clusters,
    }

    for role, variables in mapping.items():
        for variable in variables:
            role_map[variable] = role

    return role_map


def plan_preprocessing(
    analysis_plan: AnalysisPlan,
    variable_map: VariableMap,
    resolved_levels: list[ResolvedVariableLevel],
) -> PreprocessingPlan:
    """
    변수 정의와 측정수준 판정을 바탕으로 전처리 계획을 생성한다.

    실제 데이터 변경은 수행하지 않는다.
    """
    actions: list[PreprocessingAction] = []
    warnings: list[str] = []
    blocked_variables: list[str] = []

    role_map = build_role_map(analysis_plan)
    resolved_map = {result.variable_name: result for result in resolved_levels}

    referenced_variables = list(role_map)

    for variable_name in referenced_variables:
        definition = variable_map.variables.get(variable_name)
        resolution = resolved_map.get(variable_name)
        role = role_map[variable_name]

        if definition is None:
            warnings.append(f"{variable_name}: variable_map 정의가 없습니다.")
            blocked_variables.append(variable_name)
            continue

        if resolution is None:
            warnings.append(f"{variable_name}: 측정수준 판정 결과가 없습니다.")
            blocked_variables.append(variable_name)
            continue

        if resolution.status == "conflict":
            warnings.append(
                f"{variable_name}: 측정수준 근거가 충돌하여 전처리 계획을 확정할 수 없습니다."
            )
            blocked_variables.append(variable_name)
            continue

        if definition.missing_values:
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="replace_missing_values",
                    status="planned",
                    reason="사용자 정의 결측값이 등록되어 있습니다.",
                    parameters={
                        "missing_values": definition.missing_values,
                    },
                    requires_confirmation=True,
                    priority="high",
                )
            )

        if definition.reverse_coded:
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="reverse_code",
                    status="planned",
                    reason="역문항으로 지정되어 있습니다.",
                    parameters={
                        "coding": definition.coding,
                        "scale_name": definition.scale_name,
                    },
                    requires_confirmation=True,
                    priority="high",
                )
            )

        resolved_level = resolution.resolved_level

        if resolved_level == "binary":
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="review_binary_recoding",
                    status="planned",
                    reason="이분형 분석을 위해 0/1 재코딩 필요 여부를 검토합니다.",
                    parameters={
                        "coding": definition.coding,
                        "role": role,
                    },
                    requires_confirmation=True,
                )
            )

        if resolved_level in {"nominal", "ordinal"}:
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="set_reference_category",
                    status="planned",
                    reason="범주형 변수의 기준범주를 명시해야 합니다.",
                    parameters={
                        "coding": definition.coding,
                        "role": role,
                    },
                    requires_confirmation=True,
                )
            )

        if (
            role in {"independent", "moderator", "mediator", "control"}
            and resolved_level == "continuous"
        ):
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="review_centering",
                    status="planned",
                    reason=(
                        "연속형 설명변수입니다. 상호작용 또는 해석 편의를 위한 "
                        "평균중심화 여부를 검토합니다."
                    ),
                    parameters={"role": role},
                    requires_confirmation=True,
                )
            )

        if role == "moderator" and resolved_level == "continuous":
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="mean_center",
                    status="planned",
                    reason="연속형 조절변수이므로 상호작용 해석을 위해 중심화를 권장합니다.",
                    parameters={"method": "mean"},
                    requires_confirmation=True,
                    priority="high",
                )
            )

        if resolved_level == "scale_item" and definition.scale_name:
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="assign_scale_item",
                    status="planned",
                    reason="척도 구성 문항으로 확인되었습니다.",
                    parameters={
                        "scale_name": definition.scale_name,
                        "reverse_coded": definition.reverse_coded,
                    },
                    requires_confirmation=False,
                )
            )

        for custom_rule in definition.preprocessing:
            actions.append(
                PreprocessingAction(
                    variable_name=variable_name,
                    action_type="custom_rule",
                    status="planned",
                    reason="variable_map에 사용자 정의 전처리 규칙이 등록되어 있습니다.",
                    parameters=custom_rule,
                    requires_confirmation=True,
                )
            )

    for rule in analysis_plan.preprocessing.recoding_rules:
        variable_name = str(rule.get("variable", "")).strip()
        if not variable_name:
            warnings.append("변수명이 없는 recoding_rules 항목이 있습니다.")
            continue

        actions.append(
            PreprocessingAction(
                variable_name=variable_name,
                action_type="configured_recoding",
                status="planned",
                reason="analysis_plan에 재코딩 규칙이 등록되어 있습니다.",
                parameters=rule,
                requires_confirmation=True,
                priority="high",
            )
        )

    for rule in analysis_plan.preprocessing.derived_variables:
        variable_name = str(rule.get("name", "")).strip() or "미지정 파생변수"
        actions.append(
            PreprocessingAction(
                variable_name=variable_name,
                action_type="create_derived_variable",
                status="planned",
                reason="analysis_plan에 파생변수 생성 규칙이 등록되어 있습니다.",
                parameters=rule,
                requires_confirmation=True,
            )
        )

    return PreprocessingPlan(
        actions=actions,
        warnings=warnings,
        blocked_variables=sorted(set(blocked_variables)),
    )


def preprocessing_plan_to_dataframe(
    plan: PreprocessingPlan,
) -> pd.DataFrame:
    """전처리 계획을 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for sequence, action in enumerate(plan.actions, start=1):
        row = asdict(action)
        row["sequence"] = sequence
        rows.append(row)

    columns = [
        "sequence",
        "variable_name",
        "action_type",
        "status",
        "priority",
        "requires_confirmation",
        "reason",
        "parameters",
    ]

    return pd.DataFrame(rows, columns=columns)


def preprocessing_plan_summary(
    plan: PreprocessingPlan,
) -> dict[str, Any]:
    """전처리 계획의 요약정보를 반환한다."""
    action_counts: dict[str, int] = {}
    confirmation_count = 0

    for action in plan.actions:
        action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
        if action.requires_confirmation:
            confirmation_count += 1

    return {
        "action_count": len(plan.actions),
        "action_type_counts": action_counts,
        "confirmation_required_count": confirmation_count,
        "warning_count": len(plan.warnings),
        "blocked_variable_count": len(plan.blocked_variables),
        "blocked_variables": plan.blocked_variables,
    }
