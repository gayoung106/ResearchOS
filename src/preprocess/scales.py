"""복수 문항 척도 구성 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.common.config_models import VariableMap


@dataclass(slots=True)
class ScaleDefinition:
    """척도 구성 정의."""

    scale_name: str
    items: list[str]
    reverse_items: list[str] = field(default_factory=list)
    aggregation: str = "mean"
    minimum_valid_items: int | None = None
    output_name: str | None = None


@dataclass(slots=True)
class ScaleBuildRecord:
    """척도 생성 결과."""

    scale_name: str
    output_name: str
    item_count: int
    reverse_item_count: int
    aggregation: str
    minimum_valid_items: int
    valid_case_count: int
    missing_case_count: int


def collect_scale_definitions(
    variable_map: VariableMap,
) -> list[ScaleDefinition]:
    """variable_map에서 척도별 문항 구성을 수집한다."""
    grouped: dict[str, list[str]] = {}
    reverse_grouped: dict[str, list[str]] = {}

    for variable_name, definition in variable_map.variables.items():
        if not definition.scale_name:
            continue

        grouped.setdefault(definition.scale_name, []).append(variable_name)

        if definition.reverse_coded:
            reverse_grouped.setdefault(
                definition.scale_name,
                [],
            ).append(variable_name)

    return [
        ScaleDefinition(
            scale_name=scale_name,
            items=items,
            reverse_items=reverse_grouped.get(scale_name, []),
            aggregation="mean",
            minimum_valid_items=len(items),
            output_name=f"{scale_name}_mean",
        )
        for scale_name, items in grouped.items()
    ]


def build_scale(
    dataframe: pd.DataFrame,
    definition: ScaleDefinition,
) -> tuple[pd.Series, ScaleBuildRecord]:
    """정의된 문항으로 평균 또는 합산척도를 생성한다."""
    missing_columns = [item for item in definition.items if item not in dataframe.columns]

    if missing_columns:
        raise KeyError("척도 구성 문항이 데이터에 없습니다: " + ", ".join(missing_columns))

    if len(definition.items) < 2:
        raise ValueError(f"{definition.scale_name}: 척도는 최소 2개 문항이 필요합니다.")

    aggregation = definition.aggregation.lower()
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation은 mean 또는 sum이어야 합니다.")

    item_data = dataframe[definition.items].apply(
        pd.to_numeric,
        errors="coerce",
    )

    minimum_valid_items = (
        definition.minimum_valid_items
        if definition.minimum_valid_items is not None
        else len(definition.items)
    )

    if not 1 <= minimum_valid_items <= len(definition.items):
        raise ValueError("minimum_valid_items는 1 이상 문항 수 이하여야 합니다.")

    valid_item_count = item_data.notna().sum(axis=1)

    if aggregation == "mean":
        scale = item_data.mean(axis=1)
    else:
        scale = item_data.sum(axis=1, min_count=1)

    scale = scale.where(valid_item_count >= minimum_valid_items)

    output_name = definition.output_name or f"{definition.scale_name}_{aggregation}"
    scale.name = output_name

    record = ScaleBuildRecord(
        scale_name=definition.scale_name,
        output_name=output_name,
        item_count=len(definition.items),
        reverse_item_count=len(definition.reverse_items),
        aggregation=aggregation,
        minimum_valid_items=minimum_valid_items,
        valid_case_count=int(scale.notna().sum()),
        missing_case_count=int(scale.isna().sum()),
    )

    return scale, record


def build_all_scales(
    dataframe: pd.DataFrame,
    definitions: list[ScaleDefinition],
) -> tuple[pd.DataFrame, list[ScaleBuildRecord]]:
    """여러 척도를 데이터 복사본에 생성한다."""
    output = dataframe.copy(deep=True)
    records: list[ScaleBuildRecord] = []

    for definition in definitions:
        scale, record = build_scale(output, definition)

        if scale.name in output.columns:
            raise ValueError(f"척도 결과 변수명이 이미 존재합니다: {scale.name}")

        output[scale.name] = scale
        records.append(record)

    return output, records


def scale_records_to_dataframe(
    records: list[ScaleBuildRecord],
) -> pd.DataFrame:
    """척도 생성 기록을 데이터프레임으로 변환한다."""
    return pd.DataFrame(
        [
            {
                "scale_name": record.scale_name,
                "output_name": record.output_name,
                "item_count": record.item_count,
                "reverse_item_count": record.reverse_item_count,
                "aggregation": record.aggregation,
                "minimum_valid_items": record.minimum_valid_items,
                "valid_case_count": record.valid_case_count,
                "missing_case_count": record.missing_case_count,
            }
            for record in records
        ]
    )
