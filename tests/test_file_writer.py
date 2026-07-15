"""데이터 파일 저장 유틸리티 테스트."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.common.file_writer import (
    _make_safe_sheet_name,
    write_data_file,
    write_excel_sheets,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """파일 저장 테스트용 데이터프레임을 제공한다."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["가", "나", "다"],
            "score": [10.5, 20.0, 30.5],
        }
    )


def test_write_csv(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "nested" / "sample.csv"

    result = write_data_file(
        sample_dataframe,
        output,
    )

    assert result == output.resolve()
    assert result.exists()

    loaded = pd.read_csv(
        result,
        encoding="utf-8-sig",
    )

    pd.testing.assert_frame_equal(
        loaded,
        sample_dataframe,
    )


def test_write_csv_with_index(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "sample_with_index.csv"

    result = write_data_file(
        sample_dataframe,
        output,
        index=True,
    )

    loaded = pd.read_csv(
        result,
        encoding="utf-8-sig",
    )

    assert "Unnamed: 0" in loaded.columns
    assert len(loaded) == len(sample_dataframe)


def test_write_xlsx(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "sample.xlsx"

    result = write_data_file(
        sample_dataframe,
        output,
    )

    assert result.exists()

    loaded = pd.read_excel(result)

    pd.testing.assert_frame_equal(
        loaded,
        sample_dataframe,
    )


def test_write_json(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "sample.json"

    result = write_data_file(
        sample_dataframe,
        output,
    )

    assert result.exists()

    loaded = pd.read_json(result)

    pd.testing.assert_frame_equal(
        loaded,
        sample_dataframe,
        check_dtype=False,
    )


def test_write_parquet(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "sample.parquet"

    result = write_data_file(
        sample_dataframe,
        output,
    )

    assert result.exists()

    loaded = pd.read_parquet(result)

    pd.testing.assert_frame_equal(
        loaded,
        sample_dataframe,
    )


def test_write_dta(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [10.5, 20.0, 30.5],
        }
    )
    output = tmp_path / "sample.dta"

    result = write_data_file(
        dataframe,
        output,
    )

    assert result.exists()

    loaded = pd.read_stata(result)

    pd.testing.assert_frame_equal(
        loaded,
        dataframe,
        check_dtype=False,
    )


def test_write_sav_passes_labels_to_pyreadstat(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "sample.sav"
    column_labels = {
        "id": "식별자",
        "name": "이름",
        "score": "점수",
    }
    variable_value_labels = {
        "id": {
            1: "첫 번째",
            2: "두 번째",
            3: "세 번째",
        }
    }

    with patch("src.common.file_writer.pyreadstat.write_sav") as write_sav:
        result = write_data_file(
            sample_dataframe,
            output,
            column_labels=column_labels,
            variable_value_labels=variable_value_labels,
        )

    assert result == output.resolve()

    write_sav.assert_called_once_with(
        sample_dataframe,
        output.resolve(),
        column_labels=column_labels,
        variable_value_labels=variable_value_labels,
    )


def test_write_data_file_rejects_unsupported_extension(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "sample.txt"

    with pytest.raises(ValueError):
        write_data_file(
            sample_dataframe,
            output,
        )


def test_write_excel_sheets(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "reports" / "multi_sheet.xlsx"

    result = write_excel_sheets(
        {
            "기술통계": sample_dataframe,
            "상관분석": sample_dataframe[["id", "score"]],
        },
        output,
    )

    assert result == output.resolve()
    assert result.exists()

    workbook = pd.ExcelFile(result)

    assert workbook.sheet_names == [
        "기술통계",
        "상관분석",
    ]


def test_write_excel_sheets_rejects_non_xlsx_extension(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "multi_sheet.csv"

    with pytest.raises(ValueError):
        write_excel_sheets(
            {
                "sheet": sample_dataframe,
            },
            output,
        )


def test_write_excel_sheets_sanitizes_invalid_names(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    output = tmp_path / "safe_names.xlsx"

    result = write_excel_sheets(
        {
            "분석/결과": sample_dataframe,
            "분석?결과": sample_dataframe,
        },
        output,
    )

    workbook = pd.ExcelFile(result)

    assert workbook.sheet_names == [
        "분석_결과",
        "분석_결과_1",
    ]


def test_make_safe_sheet_name_replaces_invalid_characters() -> None:
    result = _make_safe_sheet_name(
        "분석[]:*?/\\결과",
        set(),
        fallback="sheet_1",
    )

    assert result == "분석_______결과"


def test_make_safe_sheet_name_uses_fallback_for_empty_name() -> None:
    result = _make_safe_sheet_name(
        "   ",
        set(),
        fallback="sheet_1",
    )

    assert result == "sheet_1"


def test_make_safe_sheet_name_limits_length() -> None:
    result = _make_safe_sheet_name(
        "가" * 40,
        set(),
        fallback="sheet_1",
    )

    assert len(result) == 31


def test_make_safe_sheet_name_avoids_duplicate_names() -> None:
    result = _make_safe_sheet_name(
        "결과",
        {
            "결과",
            "결과_1",
        },
        fallback="sheet_1",
    )

    assert result == "결과_2"


def test_make_safe_sheet_name_limits_duplicate_name_length() -> None:
    original = "가" * 31

    result = _make_safe_sheet_name(
        original,
        {
            original,
        },
        fallback="sheet_1",
    )

    assert len(result) == 31
    assert result.endswith("_1")
