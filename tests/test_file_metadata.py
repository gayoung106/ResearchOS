"""데이터셋 메타데이터 생성 및 저장 테스트."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.common.file_metadata import (
    DatasetSummary,
    _make_safe_sheet_name,
    build_frequency_tables,
    build_variable_metadata,
    save_metadata_report,
    summarize_dataset,
    summary_to_dataframe,
)
from src.common.file_reader import ReadResult


def test_summarize_dataset_counts_structure_and_quality(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 1, None, 3],
            "y": ["a", "a", "b", "c"],
        }
    )
    source_path = tmp_path / "sample.csv"

    read_result = ReadResult(
        dataframe=dataframe,
        source_path=source_path,
        file_type="csv",
    )

    summary = summarize_dataset(read_result)

    assert isinstance(summary, DatasetSummary)
    assert summary.source_file == str(source_path)
    assert summary.file_type == "csv"
    assert summary.row_count == 4
    assert summary.column_count == 2
    assert summary.duplicate_row_count == 1
    assert summary.total_missing_count == 1
    assert summary.memory_usage_bytes > 0


def test_build_variable_metadata_without_source_metadata() -> None:
    dataframe = pd.DataFrame(
        {
            "continuous": [1.0, 2.0, 3.0, None],
            "category": ["a", "b", "a", None],
            "constant": [1, 1, 1, 1],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    None,
                    "2026-01-04",
                ]
            ),
        }
    )

    metadata = build_variable_metadata(dataframe)

    assert list(metadata["variable_name"]) == [
        "continuous",
        "category",
        "constant",
        "date",
    ]

    continuous = metadata.loc[metadata["variable_name"] == "continuous"].iloc[0]
    category = metadata.loc[metadata["variable_name"] == "category"].iloc[0]
    constant = metadata.loc[metadata["variable_name"] == "constant"].iloc[0]
    date = metadata.loc[metadata["variable_name"] == "date"].iloc[0]

    assert continuous["row_count"] == 4
    assert continuous["non_missing_count"] == 3
    assert continuous["missing_count"] == 1
    assert continuous["missing_rate"] == 0.25
    assert continuous["unique_count"] == 3
    assert bool(continuous["is_numeric"]) is True
    assert bool(continuous["is_datetime"]) is False

    assert bool(category["is_numeric"]) is False
    assert category["sample_values"] == "a | b"
    assert category["unique_values"] == ["a", "b"]

    assert bool(constant["is_constant"]) is True
    assert constant["unique_count"] == 1

    assert bool(date["is_datetime"]) is True


def test_build_variable_metadata_uses_source_labels() -> None:
    dataframe = pd.DataFrame(
        {
            "gender": [1, 2, 1],
        }
    )
    source_metadata = SimpleNamespace(
        column_names_to_labels={
            "gender": "성별",
        },
        variable_value_labels={
            "gender": {
                1: "남성",
                2: "여성",
            },
        },
    )

    metadata = build_variable_metadata(
        dataframe,
        source_metadata=source_metadata,
    )

    row = metadata.iloc[0]

    assert row["variable_label"] == "성별"
    assert row["value_labels"] == {
        1: "남성",
        2: "여성",
    }


def test_build_variable_metadata_handles_missing_source_label_attributes() -> None:
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3],
        }
    )

    metadata = build_variable_metadata(
        dataframe,
        source_metadata=SimpleNamespace(),
    )

    row = metadata.iloc[0]

    assert row["variable_label"] is None
    assert row["value_labels"] is None


def test_build_variable_metadata_omits_unique_values_when_limit_exceeded() -> None:
    dataframe = pd.DataFrame(
        {
            "x": list(range(6)),
        }
    )

    metadata = build_variable_metadata(
        dataframe,
        max_unique_values=5,
    )

    row = metadata.iloc[0]

    assert row["unique_count"] == 6
    assert row["unique_values"] == []
    assert row["sample_values"] == "0 | 1 | 2 | 3 | 4"


def test_build_frequency_tables_includes_low_cardinality_variables() -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["a", "a", "b", None],
            "score": [1, 2, 3, 4],
        }
    )

    tables = build_frequency_tables(
        dataframe,
        max_unique_values=4,
    )

    assert set(tables) == {
        "group",
        "score",
    }

    group_table = tables["group"]

    assert list(group_table.columns) == [
        "value",
        "frequency",
        "percentage",
    ]
    assert group_table["frequency"].sum() == 4
    assert group_table["percentage"].sum() == 100


def test_build_frequency_tables_skips_high_cardinality_variables() -> None:
    dataframe = pd.DataFrame(
        {
            "id": list(range(10)),
            "group": ["a", "b"] * 5,
        }
    )

    tables = build_frequency_tables(
        dataframe,
        max_unique_values=3,
    )

    assert "id" not in tables
    assert "group" in tables


def test_summary_to_dataframe_converts_all_fields() -> None:
    summary = DatasetSummary(
        source_file="sample.csv",
        file_type="csv",
        row_count=10,
        column_count=3,
        duplicate_row_count=1,
        total_missing_count=2,
        memory_usage_bytes=500,
    )

    dataframe = summary_to_dataframe(summary)

    assert list(dataframe.columns) == [
        "item",
        "value",
    ]
    assert len(dataframe) == 7

    values = dict(
        zip(
            dataframe["item"],
            dataframe["value"],
            strict=True,
        )
    )

    assert values["source_file"] == "sample.csv"
    assert values["row_count"] == 10
    assert values["memory_usage_bytes"] == 500


def test_save_metadata_report_creates_expected_sheets(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["a", "a", "b"],
            "score": [1, 2, 3],
        }
    )
    read_result = ReadResult(
        dataframe=dataframe,
        source_path=tmp_path / "sample.csv",
        file_type="csv",
    )
    output_path = tmp_path / "reports" / "metadata.xlsx"

    result = save_metadata_report(
        read_result,
        output_path,
    )

    assert result == output_path
    assert result.exists()

    workbook = pd.ExcelFile(result)

    assert workbook.sheet_names == [
        "dataset_summary",
        "variable_metadata",
        "group",
        "score",
    ]


def test_save_metadata_report_sanitizes_duplicate_sheet_names(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "dataset_summary": ["a", "b"],
            "variable_metadata": [1, 2],
        }
    )
    read_result = ReadResult(
        dataframe=dataframe,
        source_path=tmp_path / "sample.csv",
        file_type="csv",
    )
    output_path = tmp_path / "metadata.xlsx"

    save_metadata_report(
        read_result,
        output_path,
    )

    workbook = pd.ExcelFile(output_path)

    assert workbook.sheet_names == [
        "dataset_summary",
        "variable_metadata",
        "dataset_summary_1",
        "variable_metadata_1",
    ]


def test_make_safe_sheet_name_replaces_invalid_characters() -> None:
    result = _make_safe_sheet_name(
        "분석[]:*?/\\결과",
        set(),
        fallback="frequency_1",
    )

    assert result == "분석_______결과"


def test_make_safe_sheet_name_uses_fallback_for_blank_name() -> None:
    result = _make_safe_sheet_name(
        "   ",
        set(),
        fallback="frequency_1",
    )

    assert result == "frequency_1"


def test_make_safe_sheet_name_limits_length() -> None:
    result = _make_safe_sheet_name(
        "가" * 40,
        set(),
        fallback="frequency_1",
    )

    assert len(result) == 31


def test_make_safe_sheet_name_avoids_duplicates() -> None:
    result = _make_safe_sheet_name(
        "group",
        {
            "group",
            "group_1",
        },
        fallback="frequency_1",
    )

    assert result == "group_2"


def test_make_safe_sheet_name_keeps_duplicate_within_limit() -> None:
    original = "가" * 31

    result = _make_safe_sheet_name(
        original,
        {
            original,
        },
        fallback="frequency_1",
    )

    assert len(result) == 31
    assert result.endswith("_1")
