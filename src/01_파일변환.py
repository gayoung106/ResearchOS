"""rawdata 폴더의 원자료를 탐색하고 분석용 형식으로 변환한다."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.file_metadata import (
    build_variable_metadata,
    save_metadata_report,
    summarize_dataset,
)
from src.common.file_reader import find_data_files, read_data_file
from src.common.file_writer import write_data_file
from src.common.logger import setup_logger
from src.common.paths import RAW_DATA_DIR, get_result_dir

STEP_NAME = "01_conversion"
OUTPUT_DIR = get_result_dir(STEP_NAME)


def sanitize_filename(filename: str) -> str:
    """
    파일명에서 Windows 및 분석 작업에 부적절한 문자를 제거한다.

    Args:
        filename: 확장자를 제외한 원래 파일명

    Returns:
        정리된 파일명
    """
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", filename)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned)

    return cleaned or "dataset"


def make_unique_name(
    base_name: str,
    used_names: set[str],
) -> str:
    """중복되지 않는 출력용 데이터셋 이름을 생성한다."""
    candidate = base_name
    sequence = 2

    while candidate in used_names:
        candidate = f"{base_name}_{sequence:02d}"
        sequence += 1

    used_names.add(candidate)
    return candidate


def prepare_dataframe_for_sav(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    SAV 저장에 부적합할 수 있는 자료형을 안전한 형태로 변환한다.

    원본 데이터프레임은 변경하지 않는다.

    Returns:
        SAV 저장용 데이터프레임과 변환 경고 목록
    """
    converted = dataframe.copy()
    warnings: list[str] = []

    for column in converted.columns:
        series = converted[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            converted[column] = series.astype("string")
            warnings.append(f"{column}: 날짜형 변수를 문자열로 변환하여 SAV에 저장했습니다.")

        elif pd.api.types.is_timedelta64_dtype(series):
            converted[column] = series.astype("string")
            warnings.append(f"{column}: 시간간격형 변수를 문자열로 변환하여 SAV에 저장했습니다.")

        elif str(series.dtype) == "category":
            converted[column] = series.astype("string")
            warnings.append(f"{column}: 범주형 변수를 문자열로 변환하여 SAV에 저장했습니다.")

        elif pd.api.types.is_bool_dtype(series):
            converted[column] = series.astype("Int64")
            warnings.append(f"{column}: 불리언 변수를 0/1 정수형으로 변환하여 SAV에 저장했습니다.")

        elif pd.api.types.is_object_dtype(series):
            converted[column] = series.astype("string")

    return converted, warnings


def extract_labels(
    source_metadata: Any | None,
) -> tuple[dict[str, str] | None, dict[str, dict[Any, str]] | None]:
    """지원되는 원자료에서 변수 라벨과 값 라벨을 추출한다."""
    if source_metadata is None:
        return None, None

    column_labels = getattr(source_metadata, "column_names_to_labels", None) or None
    value_labels = getattr(source_metadata, "variable_value_labels", None) or None

    return column_labels, value_labels


def write_markdown_report(
    *,
    dataset_name: str,
    source_path: Path,
    dataframe: pd.DataFrame,
    output_files: dict[str, str],
    warnings: list[str],
) -> Path:
    """데이터셋별 변환 결과를 Markdown 보고서로 저장한다."""
    report_path = OUTPUT_DIR / f"{dataset_name}_변환보고서.md"

    missing_count = int(dataframe.isna().sum().sum())
    duplicate_count = int(dataframe.duplicated().sum())

    warning_lines = (
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 별도 경고 없음"
    )

    output_lines = "\n".join(f"- **{name}**: `{path}`" for name, path in output_files.items())

    content = f"""# 원자료 변환 보고서

## 1. 원본 파일

- 원본 경로: `{source_path}`
- 데이터셋 이름: `{dataset_name}`
- 변환 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 2. 데이터 구조

- 행 수: {len(dataframe):,}
- 열 수: {len(dataframe.columns):,}
- 전체 결측값 수: {missing_count:,}
- 완전 중복 행 수: {duplicate_count:,}

## 3. 생성 파일

{output_lines}

## 4. 변환 경고

{warning_lines}

## 5. 주의사항

- 원자료는 수정하지 않았습니다.
- 이 단계에서는 변수 재코딩이나 결측값 처리를 수행하지 않았습니다.
- 변수의 측정수준과 값 의미는 설문지 및 코드북과 함께 다음 단계에서 검토해야 합니다.
"""

    report_path.write_text(content, encoding="utf-8")
    return report_path


def convert_single_file(
    source_path: Path,
    dataset_name: str,
    logger: Any,
) -> dict[str, Any]:
    """원자료 파일 하나를 읽고 분석용 형식으로 변환한다."""
    logger.info("파일 읽기 시작: %s", source_path)

    read_result = read_data_file(source_path)
    dataframe = read_result.dataframe

    logger.info(
        "파일 읽기 완료: %s행 × %s열",
        len(dataframe),
        len(dataframe.columns),
    )

    dataset_dir = OUTPUT_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = dataset_dir / f"{dataset_name}.parquet"
    sav_path = dataset_dir / f"{dataset_name}.sav"
    metadata_path = dataset_dir / f"{dataset_name}_메타데이터.xlsx"
    variable_csv_path = dataset_dir / f"{dataset_name}_변수목록.csv"

    write_data_file(
        dataframe,
        parquet_path,
    )

    logger.info("Parquet 저장 완료: %s", parquet_path)

    column_labels, value_labels = extract_labels(read_result.metadata)
    sav_dataframe, sav_warnings = prepare_dataframe_for_sav(dataframe)

    write_data_file(
        sav_dataframe,
        sav_path,
        column_labels=column_labels,
        variable_value_labels=value_labels,
    )

    logger.info("SAV 저장 완료: %s", sav_path)

    save_metadata_report(
        read_result,
        metadata_path,
    )

    logger.info("메타데이터 보고서 저장 완료: %s", metadata_path)

    variable_metadata = build_variable_metadata(
        dataframe,
        source_metadata=read_result.metadata,
    )
    variable_metadata.to_csv(
        variable_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    output_files = {
        "Parquet": str(parquet_path.relative_to(OUTPUT_DIR.parent)),
        "SAV": str(sav_path.relative_to(OUTPUT_DIR.parent)),
        "메타데이터 Excel": str(metadata_path.relative_to(OUTPUT_DIR.parent)),
        "변수 목록 CSV": str(variable_csv_path.relative_to(OUTPUT_DIR.parent)),
    }

    report_path = write_markdown_report(
        dataset_name=dataset_name,
        source_path=source_path,
        dataframe=dataframe,
        output_files=output_files,
        warnings=sav_warnings,
    )

    summary = summarize_dataset(read_result)

    return {
        "dataset_name": dataset_name,
        "source_file": str(source_path),
        "source_type": read_result.file_type,
        **asdict(summary),
        "parquet_file": str(parquet_path),
        "sav_file": str(sav_path),
        "metadata_file": str(metadata_path),
        "report_file": str(report_path),
        "warning_count": len(sav_warnings),
        "status": "success",
    }


def save_conversion_summary(results: list[dict[str, Any]]) -> None:
    """전체 파일 변환 결과를 Excel, CSV, JSON으로 저장한다."""
    summary_dataframe = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "전체_변환결과.csv"
    excel_path = OUTPUT_DIR / "전체_변환결과.xlsx"
    json_path = OUTPUT_DIR / "전체_변환결과.json"

    summary_dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_dataframe.to_excel(
        excel_path,
        index=False,
        engine="xlsxwriter",
    )
    json_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """rawdata 폴더의 모든 지원 파일을 변환한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "01_file_conversion",
        "01_파일변환.log",
    )

    logger.info("원자료 변환을 시작합니다.")
    logger.info("원자료 폴더: %s", RAW_DATA_DIR)
    logger.info("결과 폴더: %s", OUTPUT_DIR)

    data_files = find_data_files(RAW_DATA_DIR)

    if not data_files:
        logger.warning(
            "rawdata 폴더에 지원되는 데이터 파일이 없습니다: %s",
            RAW_DATA_DIR,
        )
        return

    logger.info("발견한 원자료 파일 수: %s", len(data_files))

    results: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for source_path in data_files:
        base_name = sanitize_filename(source_path.stem)
        dataset_name = make_unique_name(base_name, used_names)

        try:
            result = convert_single_file(
                source_path=source_path,
                dataset_name=dataset_name,
                logger=logger,
            )
            results.append(result)

        except Exception as error:
            logger.exception(
                "파일 변환 실패: %s",
                source_path,
            )
            results.append(
                {
                    "dataset_name": dataset_name,
                    "source_file": str(source_path),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    save_conversion_summary(results)

    success_count = sum(result.get("status") == "success" for result in results)
    failure_count = len(results) - success_count

    logger.info(
        "원자료 변환 완료: 성공 %s건, 실패 %s건",
        success_count,
        failure_count,
    )


if __name__ == "__main__":
    main()
