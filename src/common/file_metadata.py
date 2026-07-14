"""?곌뎄 ?곗씠?곗쓽 援ъ“? ?덉쭏??吏꾨떒?섎뒗 硫뷀??곗씠??紐⑤뱢."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.file_reader import ReadResult


@dataclass(slots=True)
class DatasetSummary:
    """?곗씠?곗뀑 ?섏? ?붿빟."""

    source_file: str
    file_type: str
    row_count: int
    column_count: int
    duplicate_row_count: int
    total_missing_count: int
    memory_usage_bytes: int


def summarize_dataset(read_result: ReadResult) -> DatasetSummary:
    """?곗씠?곗뀑??湲곕낯 援ъ“瑜??붿빟?쒕떎."""
    dataframe = read_result.dataframe

    return DatasetSummary(
        source_file=str(read_result.source_path),
        file_type=read_result.file_type,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_row_count=int(dataframe.duplicated().sum()),
        total_missing_count=int(dataframe.isna().sum().sum()),
        memory_usage_bytes=int(dataframe.memory_usage(index=True, deep=True).sum()),
    )


def build_variable_metadata(
    dataframe: pd.DataFrame,
    *,
    source_metadata: Any | None = None,
    max_unique_values: int = 20,
) -> pd.DataFrame:
    """蹂?섎퀎 ?먮즺?? 寃곗륫移? 怨좎쑀媛? ?덉떆媛?諛??쇰꺼 ?뺣낫瑜??앹꽦?쒕떎."""
    column_labels = (
        getattr(
            source_metadata,
            "column_names_to_labels",
            {},
        )
        or {}
    )
    value_labels = (
        getattr(
            source_metadata,
            "variable_value_labels",
            {},
        )
        or {}
    )

    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        unique_count = int(series.nunique(dropna=True))
        sample_values = non_null.astype(str).drop_duplicates().head(5).tolist()
        unique_values = (
            non_null.drop_duplicates().head(max_unique_values).tolist()
            if unique_count <= max_unique_values
            else []
        )

        rows.append(
            {
                "variable_name": str(column),
                "variable_label": column_labels.get(column),
                "dtype": str(series.dtype),
                "row_count": len(series),
                "non_missing_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "unique_count": unique_count,
                "is_constant": unique_count <= 1,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "is_datetime": bool(pd.api.types.is_datetime64_any_dtype(series)),
                "sample_values": " | ".join(sample_values),
                "unique_values": unique_values,
                "value_labels": value_labels.get(column),
            }
        )

    return pd.DataFrame(rows)


def build_frequency_tables(
    dataframe: pd.DataFrame,
    *,
    max_unique_values: int = 20,
) -> dict[str, pd.DataFrame]:
    """怨좎쑀媛??섍? ?곸? 蹂?섏쓽 鍮덈룄?쒕? ?앹꽦?쒕떎."""
    frequency_tables: dict[str, pd.DataFrame] = {}

    for column in dataframe.columns:
        series = dataframe[column]

        if series.nunique(dropna=False) > max_unique_values:
            continue

        table = series.value_counts(dropna=False).rename_axis("value").reset_index(name="frequency")
        table["percentage"] = table["frequency"] / len(series) * 100
        frequency_tables[str(column)] = table

    return frequency_tables


def summary_to_dataframe(summary: DatasetSummary) -> pd.DataFrame:
    """DatasetSummary瑜??몃줈???곗씠?고봽?덉엫?쇰줈 蹂?섑븳??"""
    summary_dict = asdict(summary)
    return pd.DataFrame(
        {
            "item": list(summary_dict.keys()),
            "value": list(summary_dict.values()),
        }
    )


def save_metadata_report(
    read_result: ReadResult,
    output_path: str | Path,
) -> Path:
    """?곗씠?곗뀑 ?붿빟, 蹂??硫뷀??곗씠?? 鍮덈룄?쒕? Excel濡???ν븳??"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset_summary = summarize_dataset(read_result)
    variable_metadata = build_variable_metadata(
        read_result.dataframe,
        source_metadata=read_result.metadata,
    )
    frequency_tables = build_frequency_tables(read_result.dataframe)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_to_dataframe(dataset_summary).to_excel(
            writer,
            sheet_name="dataset_summary",
            index=False,
        )
        variable_metadata.to_excel(
            writer,
            sheet_name="variable_metadata",
            index=False,
        )

        used_sheet_names = {"dataset_summary", "variable_metadata"}
        for index, (variable, table) in enumerate(
            frequency_tables.items(),
            start=1,
        ):
            safe_name = _make_safe_sheet_name(
                variable,
                used_sheet_names,
                fallback=f"frequency_{index}",
            )
            table.to_excel(writer, sheet_name=safe_name, index=False)
            used_sheet_names.add(safe_name)

    return output


def _make_safe_sheet_name(
    name: str,
    used_names: set[str],
    *,
    fallback: str,
) -> str:
    """Excel ?쒖빟??留욌뒗 以묐났 ?녿뒗 ?쒗듃紐낆쓣 ?앹꽦?쒕떎."""
    invalid_characters = set("[]:*?/\\")
    cleaned = "".join("_" if char in invalid_characters else char for char in name)
    cleaned = cleaned.strip()[:31] or fallback

    candidate = cleaned
    sequence = 1

    while candidate in used_names:
        suffix = f"_{sequence}"
        candidate = f"{cleaned[: 31 - len(suffix)]}{suffix}"
        sequence += 1

    return candidate
