from pathlib import Path

import pandas as pd
import pytest

from src.common.file_reader import (
    SUPPORTED_EXTENSIONS,
    ReadResult,
    _validate_path,
    find_data_files,
    read_data_file,
)


def test_supported_extensions_contains_common_formats() -> None:
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".sav" in SUPPORTED_EXTENSIONS
    assert ".parquet" in SUPPORTED_EXTENSIONS


def test_validate_path_accepts_valid_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = _validate_path(path)

    assert resolved == path.resolve()


def test_validate_path_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _validate_path(tmp_path / "missing.csv")


def test_validate_path_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        _validate_path(tmp_path)


def test_validate_path_unknown_extension(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.xyz"
    path.write_text("abc")

    with pytest.raises(ValueError):
        _validate_path(path)


def test_read_csv(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        "x,y\n1,2\n3,4\n",
        encoding="utf-8",
    )

    result = read_data_file(path)

    assert isinstance(result, ReadResult)
    assert result.file_type == "csv"
    assert list(result.dataframe.columns) == ["x", "y"]
    assert len(result.dataframe) == 2


def test_read_json(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    path = tmp_path / "sample.json"
    df.to_json(path)

    result = read_data_file(path)

    assert result.file_type == "json"
    assert len(result.dataframe) == 2


def test_read_parquet(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2],
        }
    )

    path = tmp_path / "sample.parquet"
    df.to_parquet(path)

    result = read_data_file(path)

    assert result.file_type == "parquet"
    assert len(result.dataframe) == 2


def test_find_data_files_returns_supported_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.xlsx").write_text("")
    (tmp_path / "ignore.xyz").write_text("")
    (tmp_path / "~$temp.xlsx").write_text("")
    (tmp_path / ".hidden.csv").write_text("")

    files = find_data_files(tmp_path)

    names = [file.name for file in files]

    assert "a.csv" in names
    assert "b.xlsx" in names
    assert "ignore.xyz" not in names
    assert "~$temp.xlsx" not in names
    assert ".hidden.csv" not in names


def test_find_data_files_recursive(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()

    (child / "sample.csv").write_text("x\n1\n")

    files = find_data_files(tmp_path)

    assert len(files) == 1
    assert files[0].name == "sample.csv"


def test_find_data_files_missing_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        find_data_files(tmp_path / "missing")


def test_find_data_files_not_directory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n")

    with pytest.raises(ValueError):
        find_data_files(path)
