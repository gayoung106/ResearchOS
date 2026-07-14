"""변수의 측정수준 후보를 탐지하는 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DetectionEvidence:
    """측정수준 판단 근거."""

    dtype: str
    unique_count: int
    non_missing_count: int
    sample_values: list[Any] = field(default_factory=list)
    variable_label: str | None = None
    value_labels: dict[Any, str] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VariableDetection:
    """개별 변수의 측정수준 탐지 결과."""

    variable_name: str
    detected_level: str
    status: str
    confidence: float
    evidence: DetectionEvidence
    alternatives: list[str] = field(default_factory=list)


def _normalize_label(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _is_integer_like(series: pd.Series) -> bool:
    """결측 제외 값이 모두 정수형으로 표현 가능한지 확인한다."""
    non_null = series.dropna()

    if non_null.empty or not pd.api.types.is_numeric_dtype(non_null):
        return False

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        return False

    return bool(((numeric % 1) == 0).all())


def _detect_likert_candidate(
    series: pd.Series,
    *,
    variable_label: str | None = None,
    value_labels: dict[Any, str] | None = None,
) -> bool:
    """
    리커트형 문항 후보인지 탐지한다.

    값 범위만으로 확정하지 않고 라벨 근거를 함께 요구한다.
    """
    non_null = series.dropna()
    unique_values = sorted(non_null.unique().tolist())

    if not _is_integer_like(series):
        return False

    if len(unique_values) < 4 or len(unique_values) > 11:
        return False

    label_text = _normalize_label(variable_label)
    value_label_text = " ".join(
        _normalize_label(str(label)) for label in (value_labels or {}).values()
    )

    keywords = (
        "전혀",
        "매우",
        "그렇다",
        "동의",
        "만족",
        "빈도",
        "중요",
        "likert",
        "agree",
        "satisfied",
        "frequency",
    )

    return any(keyword in label_text or keyword in value_label_text for keyword in keywords)


def detect_variable_level(
    variable_name: str,
    series: pd.Series,
    *,
    variable_label: str | None = None,
    value_labels: dict[Any, str] | None = None,
) -> VariableDetection:
    """
    변수 하나의 측정수준 후보를 탐지한다.

    자동 탐지 결과는 후보이며, 설문지와 코드북 검토 전에는 확정하지 않는다.
    """
    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    sample_values = non_null.drop_duplicates().head(10).tolist()
    notes: list[str] = []

    evidence = DetectionEvidence(
        dtype=str(series.dtype),
        unique_count=unique_count,
        non_missing_count=int(non_null.shape[0]),
        sample_values=sample_values,
        variable_label=variable_label,
        value_labels=value_labels,
        notes=notes,
    )

    if non_null.empty:
        notes.append("유효 응답값이 없습니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="unknown",
            status="review_required",
            confidence=0.0,
            evidence=evidence,
        )

    if pd.api.types.is_datetime64_any_dtype(series):
        return VariableDetection(
            variable_name=variable_name,
            detected_level="datetime",
            status="detected",
            confidence=0.99,
            evidence=evidence,
        )

    if pd.api.types.is_bool_dtype(series):
        notes.append("불리언 자료형으로 확인되었습니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="binary",
            status="detected",
            confidence=0.98,
            evidence=evidence,
        )

    if not pd.api.types.is_numeric_dtype(series):
        if unique_count == 2:
            notes.append("문자형이면서 고유값이 2개이므로 이분형 후보입니다.")
            return VariableDetection(
                variable_name=variable_name,
                detected_level="binary",
                status="review_required",
                confidence=0.75,
                evidence=evidence,
                alternatives=["nominal"],
            )

        notes.append("문자형 변수이므로 명목형 후보입니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="nominal",
            status="review_required",
            confidence=0.7,
            evidence=evidence,
            alternatives=["string"],
        )

    if unique_count == 2:
        notes.append("숫자형이면서 고유값이 2개입니다. 값 라벨 확인이 필요합니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="binary",
            status="review_required",
            confidence=0.8,
            evidence=evidence,
            alternatives=["nominal"],
        )

    if _detect_likert_candidate(
        series,
        variable_label=variable_label,
        value_labels=value_labels,
    ):
        notes.append("정수형 범주와 라벨 표현이 리커트 문항과 일치합니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="scale_item",
            status="review_required",
            confidence=0.85,
            evidence=evidence,
            alternatives=["ordinal"],
        )

    if _is_integer_like(series) and 3 <= unique_count <= 12:
        notes.append("고유값 수가 적은 정수형 변수입니다. 순서형 또는 명목형일 수 있습니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="ordinal",
            status="review_required",
            confidence=0.65,
            evidence=evidence,
            alternatives=["nominal", "count"],
        )

    if _is_integer_like(series) and non_null.min() >= 0:
        notes.append("0 이상의 정수형 변수이므로 횟수형 후보입니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="count",
            status="review_required",
            confidence=0.7,
            evidence=evidence,
            alternatives=["continuous"],
        )

    if pd.api.types.is_numeric_dtype(series):
        notes.append("고유값이 충분한 숫자형 변수입니다.")
        return VariableDetection(
            variable_name=variable_name,
            detected_level="continuous",
            status="detected",
            confidence=0.9,
            evidence=evidence,
        )

    notes.append("자동 판별 근거가 부족합니다.")
    return VariableDetection(
        variable_name=variable_name,
        detected_level="unknown",
        status="review_required",
        confidence=0.2,
        evidence=evidence,
    )


def detect_dataframe_variables(
    dataframe: pd.DataFrame,
    *,
    variable_metadata: pd.DataFrame | None = None,
) -> list[VariableDetection]:
    """데이터프레임 전체 변수의 측정수준 후보를 탐지한다."""
    label_map: dict[str, str] = {}
    value_label_map: dict[str, dict[Any, str]] = {}

    if variable_metadata is not None and not variable_metadata.empty:
        if {
            "variable_name",
            "variable_label",
        }.issubset(variable_metadata.columns):
            for _, row in variable_metadata.iterrows():
                name = str(row["variable_name"])
                label = row["variable_label"]
                if pd.notna(label):
                    label_map[name] = str(label)

        if {
            "variable_name",
            "value_labels",
        }.issubset(variable_metadata.columns):
            for _, row in variable_metadata.iterrows():
                name = str(row["variable_name"])
                labels = row["value_labels"]
                if isinstance(labels, dict):
                    value_label_map[name] = labels

    return [
        detect_variable_level(
            variable_name=str(column),
            series=dataframe[column],
            variable_label=label_map.get(str(column)),
            value_labels=value_label_map.get(str(column)),
        )
        for column in dataframe.columns
    ]


def detections_to_dataframe(
    detections: list[VariableDetection],
) -> pd.DataFrame:
    """탐지 결과를 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for detection in detections:
        row = asdict(detection)
        evidence = row.pop("evidence")
        row.update(
            {
                "dtype": evidence["dtype"],
                "unique_count": evidence["unique_count"],
                "non_missing_count": evidence["non_missing_count"],
                "sample_values": evidence["sample_values"],
                "variable_label": evidence["variable_label"],
                "value_labels": evidence["value_labels"],
                "notes": " | ".join(evidence["notes"]),
                "alternatives": " | ".join(row["alternatives"]),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def detection_summary(
    detections: list[VariableDetection],
) -> dict[str, Any]:
    """탐지 결과 요약을 반환한다."""
    status_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}

    for detection in detections:
        status_counts[detection.status] = status_counts.get(detection.status, 0) + 1
        level_counts[detection.detected_level] = level_counts.get(detection.detected_level, 0) + 1

    return {
        "variable_count": len(detections),
        "status_counts": status_counts,
        "level_counts": level_counts,
    }
