"""자동 탐지 결과와 설문지·코드북 근거를 통합하는 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.preprocess.detector import VariableDetection

VALID_LEVELS = {
    "binary",
    "nominal",
    "ordinal",
    "continuous",
    "count",
    "proportion",
    "datetime",
    "string",
    "multi_response",
    "scale_item",
    "unknown",
}

COMPATIBLE_LEVEL_PAIRS = {
    frozenset({"ordinal", "scale_item"}),
}


@dataclass(slots=True)
class VariableEvidence:
    """설문지·코드북·값 라벨에서 확보한 측정수준 근거."""

    variable_name: str
    questionnaire_level: str | None = None
    codebook_level: str | None = None
    value_label_level: str | None = None
    questionnaire_text: str | None = None
    codebook_note: str | None = None
    value_labels: dict[Any, str] | None = None
    source_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResolvedVariableLevel:
    """근거 통합 후 최종 측정수준 판정."""

    variable_name: str
    detected_level: str
    resolved_level: str
    status: str
    confidence: float
    supporting_sources: list[str]
    conflicts: list[str]
    notes: list[str]


def _normalize_level(level: str | None) -> str | None:
    """측정수준 명칭을 내부 표준값으로 변환한다."""
    if level is None:
        return None

    value = level.strip().lower()

    aliases = {
        "dichotomous": "binary",
        "이분형": "binary",
        "명목형": "nominal",
        "순서형": "ordinal",
        "서열형": "ordinal",
        "연속형": "continuous",
        "척도형": "continuous",
        "횟수형": "count",
        "비율형": "proportion",
        "날짜형": "datetime",
        "문자형": "string",
        "다중응답": "multi_response",
        "리커트": "scale_item",
        "likert": "scale_item",
        "척도문항": "scale_item",
        "미확정": "unknown",
    }

    normalized = aliases.get(value, value)

    if normalized not in VALID_LEVELS:
        return None

    return normalized


def resolve_variable_evidence(
    detection: VariableDetection,
    evidence: VariableEvidence,
) -> ResolvedVariableLevel:
    """
    자동 탐지와 외부 근거를 통합해 측정수준을 판정한다.

    우선순위:
    1. 설문지
    2. 코드북
    3. 값 라벨
    4. 자동 탐지
    """
    questionnaire_level = _normalize_level(evidence.questionnaire_level)
    codebook_level = _normalize_level(evidence.codebook_level)
    value_label_level = _normalize_level(evidence.value_label_level)

    external_levels = [
        level
        for level in (
            questionnaire_level,
            codebook_level,
            value_label_level,
        )
        if level is not None
    ]

    supporting_sources: list[str] = []
    conflicts: list[str] = []
    notes: list[str] = []

    if questionnaire_level:
        supporting_sources.append("questionnaire")
    if codebook_level:
        supporting_sources.append("codebook")
    if value_label_level:
        supporting_sources.append("value_labels")

    if not external_levels:
        notes.append("설문지·코드북·값 라벨에서 측정수준 근거를 확인하지 못했습니다.")
        return ResolvedVariableLevel(
            variable_name=detection.variable_name,
            detected_level=detection.detected_level,
            resolved_level=detection.detected_level,
            status="review_required",
            confidence=min(detection.confidence, 0.8),
            supporting_sources=["automatic_detection"],
            conflicts=[],
            notes=notes,
        )

    unique_external_levels = set(external_levels)

    if len(unique_external_levels) > 1:
        conflicts.append(
            "설문지·코드북·값 라벨의 측정수준 정보가 서로 다릅니다: "
            + ", ".join(sorted(unique_external_levels))
        )
        return ResolvedVariableLevel(
            variable_name=detection.variable_name,
            detected_level=detection.detected_level,
            resolved_level="unknown",
            status="conflict",
            confidence=0.0,
            supporting_sources=supporting_sources,
            conflicts=conflicts,
            notes=notes,
        )

    resolved_level = external_levels[0]

    detected_and_resolved = frozenset(
        {
            detection.detected_level,
            resolved_level,
        }
    )

    is_compatible = detected_and_resolved in COMPATIBLE_LEVEL_PAIRS

    if (
        detection.detected_level != "unknown"
        and detection.detected_level != resolved_level
        and not is_compatible
    ):
        conflicts.append(
            f"자동 탐지({detection.detected_level})와 외부 근거({resolved_level})가 다릅니다."
        )

    if is_compatible:
        notes.append(
            "자동 탐지 결과와 외부 근거는 표현이 다르지만 순서형 리커트 문항으로 서로 호환됩니다."
        )

    if questionnaire_level and codebook_level:
        confidence = 0.98
    elif questionnaire_level:
        confidence = 0.95
    elif codebook_level:
        confidence = 0.92
    else:
        confidence = 0.85

    status = "confirmed" if not conflicts else "review_required"

    if status == "confirmed":
        notes.append("외부 자료의 측정수준 근거가 일관되게 확인되었습니다.")
    else:
        notes.append("외부 근거가 우선되지만 자동 탐지와 차이가 있어 검토가 필요합니다.")

    return ResolvedVariableLevel(
        variable_name=detection.variable_name,
        detected_level=detection.detected_level,
        resolved_level=resolved_level,
        status=status,
        confidence=confidence,
        supporting_sources=supporting_sources,
        conflicts=conflicts,
        notes=notes,
    )


def resolve_all_variable_evidence(
    detections: list[VariableDetection],
    evidences: list[VariableEvidence],
) -> list[ResolvedVariableLevel]:
    """전체 변수의 탐지 결과와 외부 근거를 통합한다."""
    evidence_map = {evidence.variable_name: evidence for evidence in evidences}

    results: list[ResolvedVariableLevel] = []

    for detection in detections:
        evidence = evidence_map.get(
            detection.variable_name,
            VariableEvidence(variable_name=detection.variable_name),
        )
        results.append(
            resolve_variable_evidence(
                detection,
                evidence,
            )
        )

    return results


def resolved_levels_to_dataframe(
    results: list[ResolvedVariableLevel],
) -> pd.DataFrame:
    """측정수준 판정 결과를 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for result in results:
        row = asdict(result)
        row["supporting_sources"] = " | ".join(result.supporting_sources)
        row["conflicts"] = " | ".join(result.conflicts)
        row["notes"] = " | ".join(result.notes)
        rows.append(row)

    return pd.DataFrame(rows)


def evidence_from_dataframe(
    dataframe: pd.DataFrame,
) -> list[VariableEvidence]:
    """
    표 형식의 근거 데이터를 VariableEvidence 목록으로 변환한다.

    지원 열:
    - variable_name
    - questionnaire_level
    - codebook_level
    - value_label_level
    - questionnaire_text
    - codebook_note
    - value_labels
    - source_files
    """
    if "variable_name" not in dataframe.columns:
        raise ValueError("근거 데이터에는 variable_name 열이 필요합니다.")

    evidences: list[VariableEvidence] = []

    for _, row in dataframe.iterrows():
        source_files = row.get("source_files", [])
        if isinstance(source_files, str):
            source_files = [item.strip() for item in source_files.split("|") if item.strip()]
        elif not isinstance(source_files, list):
            source_files = []

        value_labels = row.get("value_labels")
        if not isinstance(value_labels, dict):
            value_labels = None

        evidences.append(
            VariableEvidence(
                variable_name=str(row["variable_name"]),
                questionnaire_level=_optional_string(row.get("questionnaire_level")),
                codebook_level=_optional_string(row.get("codebook_level")),
                value_label_level=_optional_string(row.get("value_label_level")),
                questionnaire_text=_optional_string(row.get("questionnaire_text")),
                codebook_note=_optional_string(row.get("codebook_note")),
                value_labels=value_labels,
                source_files=source_files,
            )
        )

    return evidences


def resolution_summary(
    results: list[ResolvedVariableLevel],
) -> dict[str, Any]:
    """판정 상태 및 측정수준별 요약을 반환한다."""
    status_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}

    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        level_counts[result.resolved_level] = level_counts.get(result.resolved_level, 0) + 1

    return {
        "variable_count": len(results),
        "status_counts": status_counts,
        "resolved_level_counts": level_counts,
    }


def _optional_string(value: Any) -> str | None:
    """빈값과 NaN을 None으로 변환한다."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return text or None
