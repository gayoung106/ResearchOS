"""공통 파일 입출력 모듈 테스트."""

from pathlib import Path

import pandas as pd

from src.common.file_metadata import build_variable_metadata, summarize_dataset
from src.common.file_reader import read_data_file
from src.common.file_writer import write_data_file


def test_csv_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "sample.csv"
    expected = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "group": ["A", "B", "A"],
            "score": [3.2, 4.1, None],
        }
    )
    expected.to_csv(source, index=False, encoding="utf-8-sig")

    result = read_data_file(source)

    assert result.file_type == "csv"
    assert result.dataframe.shape == (3, 3)
    assert result.dataframe["score"].isna().sum() == 1


def test_metadata_generation(tmp_path: Path) -> None:
    source = tmp_path / "sample.parquet"
    dataframe = pd.DataFrame(
        {
            "binary": [0, 1, 1, None],
            "constant": ["A", "A", "A", "A"],
        }
    )
    write_data_file(dataframe, source)

    result = read_data_file(source)
    summary = summarize_dataset(result)
    metadata = build_variable_metadata(result.dataframe)

    assert summary.row_count == 4
    assert summary.column_count == 2
    assert bool(
        metadata.loc[
            metadata["variable_name"] == "constant",
            "is_constant",
        ].iloc[0]
    )


def test_excel_output(tmp_path: Path) -> None:
    output = tmp_path / "output.xlsx"
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    saved_path = write_data_file(dataframe, output)

    assert saved_path.exists()
