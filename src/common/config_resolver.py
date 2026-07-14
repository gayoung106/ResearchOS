"""설정 변수명과 실제 데이터 변수명을 비교하는 자동해결기."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from src.common.config_models import AnalysisPlan


@dataclass(slots=True)
class VariableCandidate:
    """변수명 후보."""

    requested_name: str
    candidate_name: str
    match_type: str
    similarity: float
    candidate_label: str | None = None


@dataclass(slots=True)
class VariableResolution:
    """변수명 해결 결과."""

    requested_name: str
    resolved_name: str | None
    status: str
    match_type: str | None
    similarity: float | None
    candidates: list[VariableCandidate]


def normalize_variable_name(value: str) -> str:
    """
    변수명 비교를 위한 정규화 문자열을 생성한다.

    공백, 하이픈, 밑줄 및 영숫자 외 문자를 제거하고 소문자로 변환한다.
    """
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s_-]+", "", normalized)
    normalized = re.sub(r"[^0-9a-z가-힣]", "", normalized)
    return normalized


def calculate_similarity(left: str, right: str) -> float:
    """두 문자열의 0~1 범위 유사도를 계산한다."""
    return SequenceMatcher(
        None,
        normalize_variable_name(left),
        normalize_variable_name(right),
    ).ratio()


def collect_requested_variables(analysis_plan: AnalysisPlan) -> list[str]:
    """분석계획에 등록된 모든 변수명을 중복 없이 반환한다."""
    groups = analysis_plan.variables

    values = (
        groups.dependent
        + groups.independent
        + groups.mediators
        + groups.moderators
        + groups.controls
        + groups.fixed_effects
        + groups.weights
        + groups.clusters
    )

    return list(dict.fromkeys(values))


def build_label_map(
    variable_metadata: pd.DataFrame | None,
) -> dict[str, str]:
    """
    변수 메타데이터에서 변수명-라벨 매핑을 생성한다.

    필요한 열:
    - variable_name
    - variable_label
    """
    if variable_metadata is None or variable_metadata.empty:
        return {}

    required_columns = {"variable_name", "variable_label"}
    if not required_columns.issubset(variable_metadata.columns):
        return {}

    label_map: dict[str, str] = {}

    for _, row in variable_metadata.iterrows():
        variable_name = str(row["variable_name"])
        label = row["variable_label"]

        if pd.notna(label) and str(label).strip():
            label_map[variable_name] = str(label).strip()

    return label_map


def resolve_variable_name(
    requested_name: str,
    available_columns: list[str],
    *,
    label_map: dict[str, str] | None = None,
    similarity_threshold: float = 0.72,
    max_candidates: int = 5,
) -> VariableResolution:
    """
    요청한 변수명을 실제 데이터 변수명과 비교한다.

    우선순위:
    1. 정확 일치
    2. 대소문자 무시 일치
    3. 정규화 일치
    4. 변수명 및 변수라벨 유사도 후보

    유사도 후보는 자동 확정하지 않는다.
    """
    label_map = label_map or {}

    if requested_name in available_columns:
        return VariableResolution(
            requested_name=requested_name,
            resolved_name=requested_name,
            status="resolved",
            match_type="exact",
            similarity=1.0,
            candidates=[],
        )

    lower_matches = [
        column for column in available_columns if column.lower() == requested_name.lower()
    ]
    if len(lower_matches) == 1:
        resolved = lower_matches[0]
        return VariableResolution(
            requested_name=requested_name,
            resolved_name=resolved,
            status="resolved",
            match_type="case_insensitive",
            similarity=1.0,
            candidates=[],
        )

    requested_normalized = normalize_variable_name(requested_name)
    normalized_matches = [
        column
        for column in available_columns
        if normalize_variable_name(column) == requested_normalized
    ]
    if len(normalized_matches) == 1:
        resolved = normalized_matches[0]
        return VariableResolution(
            requested_name=requested_name,
            resolved_name=resolved,
            status="resolved",
            match_type="normalized",
            similarity=1.0,
            candidates=[],
        )

    candidates: list[VariableCandidate] = []

    for column in available_columns:
        name_similarity = calculate_similarity(requested_name, column)
        label = label_map.get(column)
        label_similarity = calculate_similarity(requested_name, label) if label else 0.0

        similarity = max(name_similarity, label_similarity)
        if similarity < similarity_threshold:
            continue

        match_type = "label_similarity" if label_similarity > name_similarity else "name_similarity"

        candidates.append(
            VariableCandidate(
                requested_name=requested_name,
                candidate_name=column,
                match_type=match_type,
                similarity=round(similarity, 4),
                candidate_label=label,
            )
        )

    candidates.sort(
        key=lambda candidate: candidate.similarity,
        reverse=True,
    )
    candidates = candidates[:max_candidates]

    if candidates:
        return VariableResolution(
            requested_name=requested_name,
            resolved_name=None,
            status="review_required",
            match_type=None,
            similarity=None,
            candidates=candidates,
        )

    return VariableResolution(
        requested_name=requested_name,
        resolved_name=None,
        status="not_found",
        match_type=None,
        similarity=None,
        candidates=[],
    )


def resolve_analysis_variables(
    analysis_plan: AnalysisPlan,
    available_columns: list[str],
    *,
    variable_metadata: pd.DataFrame | None = None,
    similarity_threshold: float = 0.72,
) -> list[VariableResolution]:
    """분석계획에 포함된 모든 변수명을 해결한다."""
    label_map = build_label_map(variable_metadata)
    requested_variables = collect_requested_variables(analysis_plan)

    return [
        resolve_variable_name(
            requested_name=variable,
            available_columns=available_columns,
            label_map=label_map,
            similarity_threshold=similarity_threshold,
        )
        for variable in requested_variables
    ]


def resolutions_to_dataframe(
    resolutions: list[VariableResolution],
) -> pd.DataFrame:
    """변수 해결 결과를 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for resolution in resolutions:
        if resolution.candidates:
            for rank, candidate in enumerate(
                resolution.candidates,
                start=1,
            ):
                rows.append(
                    {
                        "requested_name": resolution.requested_name,
                        "status": resolution.status,
                        "resolved_name": resolution.resolved_name,
                        "match_type": candidate.match_type,
                        "similarity": candidate.similarity,
                        "candidate_rank": rank,
                        "candidate_name": candidate.candidate_name,
                        "candidate_label": candidate.candidate_label,
                    }
                )
        else:
            rows.append(
                {
                    "requested_name": resolution.requested_name,
                    "status": resolution.status,
                    "resolved_name": resolution.resolved_name,
                    "match_type": resolution.match_type,
                    "similarity": resolution.similarity,
                    "candidate_rank": None,
                    "candidate_name": None,
                    "candidate_label": None,
                }
            )

    return pd.DataFrame(rows)


def apply_confirmed_resolutions(
    analysis_plan: AnalysisPlan,
    confirmed_mapping: dict[str, str],
) -> AnalysisPlan:
    """
    연구자가 확정한 변수명 매핑을 분석계획 복사본에 적용한다.

    자동 후보는 적용하지 않으며, confirmed_mapping에 포함된 항목만 변경한다.
    """
    plan_data = analysis_plan.model_dump()
    variable_groups = plan_data["variables"]

    for group_name, variables in variable_groups.items():
        variable_groups[group_name] = [
            confirmed_mapping.get(variable, variable) for variable in variables
        ]

    return AnalysisPlan.model_validate(plan_data)


def resolution_summary(
    resolutions: list[VariableResolution],
) -> dict[str, int]:
    """해결 상태별 변수 개수를 반환한다."""
    summary = {
        "resolved": 0,
        "review_required": 0,
        "not_found": 0,
    }

    for resolution in resolutions:
        summary[resolution.status] += 1

    return summary


def serialize_resolutions(
    resolutions: list[VariableResolution],
) -> list[dict[str, Any]]:
    """변수 해결 결과를 JSON 직렬화 가능한 구조로 변환한다."""
    return [
        {
            **asdict(resolution),
            "candidates": [asdict(candidate) for candidate in resolution.candidates],
        }
        for resolution in resolutions
    ]


def validate_confirmed_mapping(
    confirmed_mapping: dict[str, str],
    available_columns: list[str],
) -> None:
    """확정 매핑의 대상 변수가 실제 데이터에 존재하는지 검사한다."""
    invalid_targets = sorted(
        target for target in confirmed_mapping.values() if target not in available_columns
    )

    if invalid_targets:
        raise ValueError(
            "확정 매핑의 대상 변수가 실제 데이터에 없습니다: " + ", ".join(invalid_targets)
        )
