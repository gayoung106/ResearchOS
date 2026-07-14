"""다양한 연구 데이터 파일을 자동 판별하여 읽는 모듈."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".txt",
    ".xlsx",
    ".xls",
    ".sav",
    ".dta",
    ".sas7bdat",
    ".parquet",
    ".json",
}


@dataclass(slots=True)
class ReadResult:
    """데이터 파일 읽기 결과."""

    dataframe: pd.DataFrame
    source_path: Path
    file_type: str
    metadata: Any | None = None


def _validate_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if not path.is_file():
        raise ValueError(f"파일 경로가 아닙니다: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}. 지원 형식: {supported}")

    return path


def read_data_file(
    file_path: str | Path,
    *,
    encoding: str | None = None,
    sheet_name: str | int | None = 0,
    csv_kwargs: dict[str, Any] | None = None,
    excel_kwargs: dict[str, Any] | None = None,
) -> ReadResult:
    """확장자를 기준으로 연구 데이터 파일을 읽는다."""
    path = _validate_path(file_path)
    suffix = path.suffix.lower()
    csv_kwargs = csv_kwargs or {}
    excel_kwargs = excel_kwargs or {}

    if suffix in {".csv", ".txt"}:
        encodings = [encoding] if encoding else ["utf-8-sig", "cp949", "utf-8"]
        last_error: Exception | None = None

        for candidate in encodings:
            try:
                dataframe = pd.read_csv(path, encoding=candidate, **csv_kwargs)
                return ReadResult(
                    dataframe=dataframe,
                    source_path=path,
                    file_type=suffix.removeprefix("."),
                )
            except UnicodeDecodeError as exc:
                last_error = exc

        raise ValueError(
            f"지원 인코딩으로 파일을 읽지 못했습니다: {path}. 마지막 오류: {last_error}"
        )

    if suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(
            path,
            sheet_name=sheet_name,
            **excel_kwargs,
        )
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("여러 시트를 반환했습니다. sheet_name을 하나의 시트로 지정하세요.")
        return ReadResult(
            dataframe=dataframe,
            source_path=path,
            file_type=suffix.removeprefix("."),
        )

    if suffix == ".sav":
        dataframe, metadata = pyreadstat.read_sav(path)
        return ReadResult(dataframe, path, "sav", metadata)

    if suffix == ".dta":
        dataframe, metadata = pyreadstat.read_dta(path)
        return ReadResult(dataframe, path, "dta", metadata)

    if suffix == ".sas7bdat":
        dataframe, metadata = pyreadstat.read_sas7bdat(path)
        return ReadResult(dataframe, path, "sas7bdat", metadata)

    if suffix == ".parquet":
        return ReadResult(pd.read_parquet(path), path, "parquet")

    if suffix == ".json":
        return ReadResult(pd.read_json(path), path, "json")

    raise RuntimeError(f"처리되지 않은 파일 형식입니다: {suffix}")


def find_data_files(directory: str | Path) -> list[Path]:
    """디렉터리에서 지원되는 데이터 파일을 재귀적으로 찾는다."""
    root = Path(directory).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"디렉터리를 찾을 수 없습니다: {root}")

    if not root.is_dir():
        raise ValueError(f"디렉터리 경로가 아닙니다: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
        and not path.name.startswith("~$")
    ]

    return sorted(files)
