"""기술통계 및 데이터 품질 리포트 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(slots=True)
class DataQualityWarning:
    """데이터 품질 경고."""

    variable_name: str
    warning_type: str
    severity: str
    message: str


@dataclass(slots=True)
class DescriptiveReport:
    """기술통계 및 품질점검 결과."""

    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    quality_warnings: list[DataQualityWarning]
    dataset_summary: dict[str, Any]


def summarize_numeric_variable(
    series: pd.Series,
) -> dict[str, Any]:
    """숫자형 변수의 기술통계를 계산한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    count = int(valid.shape[0])
    missing_count = int(numeric.isna().sum())

    if count == 0:
        return {
            "variable_name": str(series.name),
            "n": 0,
            "missing_count": missing_count,
            "missing_rate": 1.0 if len(series) else 0.0,
            "mean": np.nan,
            "standard_deviation": np.nan,
            "variance": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "range": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "coefficient_of_variation": np.nan,
            "confidence_interval_lower": np.nan,
            "confidence_interval_upper": np.nan,
        }

    mean_value = float(valid.mean())
    standard_deviation = float(valid.std(ddof=1)) if count > 1 else np.nan
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    standard_error = (
        standard_deviation / np.sqrt(count)
        if count > 1 and not np.isnan(standard_deviation)
        else np.nan
    )
    confidence_margin = (
        float(stats.t.ppf(0.975, df=count - 1)) * standard_error
        if count > 1 and not np.isnan(standard_error)
        else np.nan
    )

    return {
        "variable_name": str(series.name),
        "n": count,
        "missing_count": missing_count,
        "missing_rate": missing_count / len(series) if len(series) else 0.0,
        "mean": mean_value,
        "standard_deviation": standard_deviation,
        "variance": float(valid.var(ddof=1)) if count > 1 else np.nan,
        "median": float(valid.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "minimum": float(valid.min()),
        "maximum": float(valid.max()),
        "range": float(valid.max() - valid.min()),
        "skewness": float(valid.skew()) if count > 2 else np.nan,
        "kurtosis": float(valid.kurt()) if count > 3 else np.nan,
        "coefficient_of_variation": (
            standard_deviation / mean_value
            if mean_value != 0 and not np.isnan(standard_deviation)
            else np.nan
        ),
        "confidence_interval_lower": (
            mean_value - confidence_margin if not np.isnan(confidence_margin) else np.nan
        ),
        "confidence_interval_upper": (
            mean_value + confidence_margin if not np.isnan(confidence_margin) else np.nan
        ),
    }


def summarize_categorical_variable(
    series: pd.Series,
) -> pd.DataFrame:
    """범주형 변수의 빈도와 비율을 계산한다."""
    total_count = len(series)
    valid_count = int(series.notna().sum())

    table = series.value_counts(dropna=False).rename_axis("value").reset_index(name="frequency")
    table.insert(0, "variable_name", str(series.name))
    table["percent"] = table["frequency"] / total_count * 100 if total_count else 0.0
    table["valid_percent"] = table.apply(
        lambda row: (
            row["frequency"] / valid_count * 100
            if valid_count and pd.notna(row["value"])
            else np.nan
        ),
        axis=1,
    )

    valid_rows = table["value"].notna()
    table["cumulative_valid_percent"] = np.nan
    table.loc[
        valid_rows,
        "cumulative_valid_percent",
    ] = table.loc[valid_rows, "valid_percent"].cumsum()

    return table


def generate_quality_warnings(
    dataframe: pd.DataFrame,
) -> list[DataQualityWarning]:
    """변수별 데이터 품질 경고를 생성한다."""
    warnings: list[DataQualityWarning] = []
    row_count = len(dataframe)

    for column in dataframe.columns:
        series = dataframe[column]
        missing_rate = float(series.isna().mean()) if row_count else 0.0
        unique_count = int(series.nunique(dropna=True))

        if missing_rate >= 0.20:
            warnings.append(
                DataQualityWarning(
                    variable_name=str(column),
                    warning_type="high_missingness",
                    severity="high",
                    message=f"결측률이 {missing_rate:.1%}입니다.",
                )
            )

        if unique_count <= 1:
            warnings.append(
                DataQualityWarning(
                    variable_name=str(column),
                    warning_type="constant",
                    severity="high",
                    message="상수 변수이거나 유효한 값이 하나뿐입니다.",
                )
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()

            if len(numeric) >= 3:
                skewness = float(numeric.skew())
                if abs(skewness) >= 2:
                    warnings.append(
                        DataQualityWarning(
                            variable_name=str(column),
                            warning_type="extreme_skewness",
                            severity="normal",
                            message=f"왜도의 절대값이 큽니다: {skewness:.3f}",
                        )
                    )

            if np.isinf(pd.to_numeric(series, errors="coerce")).any():
                warnings.append(
                    DataQualityWarning(
                        variable_name=str(column),
                        warning_type="infinite_value",
                        severity="high",
                        message="무한대 값이 포함되어 있습니다.",
                    )
                )
        else:
            valid = series.dropna()
            if not valid.empty:
                dominant_rate = float(valid.value_counts(normalize=True).iloc[0])
                if dominant_rate >= 0.95:
                    warnings.append(
                        DataQualityWarning(
                            variable_name=str(column),
                            warning_type="dominant_category",
                            severity="normal",
                            message=(f"최빈 범주의 비율이 {dominant_rate:.1%}입니다."),
                        )
                    )

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        warnings.append(
            DataQualityWarning(
                variable_name="__dataset__",
                warning_type="duplicate_rows",
                severity="normal",
                message=f"완전 중복 행이 {duplicate_count}개 있습니다.",
            )
        )

    return warnings


def build_descriptive_report(
    dataframe: pd.DataFrame,
    *,
    categorical_max_unique: int = 20,
) -> DescriptiveReport:
    """전체 데이터의 기술통계 및 품질리포트를 생성한다."""
    numeric_rows: list[dict[str, Any]] = []
    categorical_tables: list[pd.DataFrame] = []

    for column in dataframe.columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(series):
            numeric_rows.append(summarize_numeric_variable(series))

        if not pd.api.types.is_numeric_dtype(series) or unique_count <= categorical_max_unique:
            categorical_tables.append(summarize_categorical_variable(series))

    numeric_summary = pd.DataFrame(numeric_rows)
    categorical_summary = (
        pd.concat(categorical_tables, ignore_index=True)
        if categorical_tables
        else pd.DataFrame(
            columns=[
                "variable_name",
                "value",
                "frequency",
                "percent",
                "valid_percent",
                "cumulative_valid_percent",
            ]
        )
    )

    warnings = generate_quality_warnings(dataframe)

    dataset_summary = {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "total_missing_count": int(dataframe.isna().sum().sum()),
        "numeric_variable_count": int(len(dataframe.select_dtypes(include=[np.number]).columns)),
        "non_numeric_variable_count": int(
            len(dataframe.select_dtypes(exclude=[np.number]).columns)
        ),
    }

    return DescriptiveReport(
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        quality_warnings=warnings,
        dataset_summary=dataset_summary,
    )


def quality_warnings_to_dataframe(
    warnings: list[DataQualityWarning],
) -> pd.DataFrame:
    """품질 경고를 데이터프레임으로 변환한다."""
    return pd.DataFrame(
        [asdict(warning) for warning in warnings],
        columns=[
            "variable_name",
            "warning_type",
            "severity",
            "message",
        ],
    )


def dataset_summary_to_dataframe(
    summary: dict[str, Any],
) -> pd.DataFrame:
    """데이터셋 요약을 세로형 표로 변환한다."""
    return pd.DataFrame(
        {
            "item": list(summary.keys()),
            "value": list(summary.values()),
        }
    )
