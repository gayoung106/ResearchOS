"""단변량 및 다변량 이상치 진단 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2


@dataclass(slots=True)
class OutlierVariableResult:
    """변수별 단변량 이상치 진단 결과."""

    variable_name: str
    method: str
    outlier_count: int
    outlier_rate: float
    lower_bound: float | None
    upper_bound: float | None
    outlier_indices: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MahalanobisResult:
    """다변량 Mahalanobis 거리 진단 결과."""

    variables: list[str]
    valid_case_count: int
    degrees_of_freedom: int
    significance_level: float
    cutoff: float
    outlier_count: int
    outlier_rate: float
    distances: pd.Series
    outlier_indices: list[Any]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OutlierReport:
    """전체 이상치 진단 결과."""

    univariate_results: list[OutlierVariableResult]
    mahalanobis_result: MahalanobisResult | None
    warnings: list[str]


def detect_zscore_outliers(
    series: pd.Series,
    *,
    threshold: float = 3.0,
) -> OutlierVariableResult:
    """절대 z-score 기준으로 단변량 이상치를 탐지한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    warnings: list[str] = []

    if threshold <= 0:
        raise ValueError("threshold는 0보다 커야 합니다.")

    if valid.empty:
        warnings.append("유효한 숫자값이 없습니다.")
        return OutlierVariableResult(
            variable_name=str(series.name),
            method="zscore",
            outlier_count=0,
            outlier_rate=0.0,
            lower_bound=None,
            upper_bound=None,
            outlier_indices=[],
            warnings=warnings,
        )

    standard_deviation = float(valid.std(ddof=0))

    if np.isclose(standard_deviation, 0.0):
        warnings.append("표준편차가 0이므로 이상치를 계산할 수 없습니다.")
        return OutlierVariableResult(
            variable_name=str(series.name),
            method="zscore",
            outlier_count=0,
            outlier_rate=0.0,
            lower_bound=float(valid.iloc[0]),
            upper_bound=float(valid.iloc[0]),
            outlier_indices=[],
            warnings=warnings,
        )

    mean_value = float(valid.mean())
    z_scores = (valid - mean_value) / standard_deviation
    outlier_mask = z_scores.abs() > threshold
    outlier_indices = valid.index[outlier_mask].tolist()

    return OutlierVariableResult(
        variable_name=str(series.name),
        method="zscore",
        outlier_count=len(outlier_indices),
        outlier_rate=len(outlier_indices) / len(valid),
        lower_bound=mean_value - threshold * standard_deviation,
        upper_bound=mean_value + threshold * standard_deviation,
        outlier_indices=outlier_indices,
        warnings=warnings,
    )


def detect_iqr_outliers(
    series: pd.Series,
    *,
    multiplier: float = 1.5,
) -> OutlierVariableResult:
    """IQR 기준으로 단변량 이상치를 탐지한다."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    warnings: list[str] = []

    if multiplier <= 0:
        raise ValueError("multiplier는 0보다 커야 합니다.")

    if valid.empty:
        warnings.append("유효한 숫자값이 없습니다.")
        return OutlierVariableResult(
            variable_name=str(series.name),
            method="iqr",
            outlier_count=0,
            outlier_rate=0.0,
            lower_bound=None,
            upper_bound=None,
            outlier_indices=[],
            warnings=warnings,
        )

    first_quartile = float(valid.quantile(0.25))
    third_quartile = float(valid.quantile(0.75))
    interquartile_range = third_quartile - first_quartile

    lower_bound = first_quartile - multiplier * interquartile_range
    upper_bound = third_quartile + multiplier * interquartile_range

    outlier_mask = (valid < lower_bound) | (valid > upper_bound)
    outlier_indices = valid.index[outlier_mask].tolist()

    if np.isclose(interquartile_range, 0.0):
        warnings.append("IQR이 0이므로 결과 해석에 주의가 필요합니다.")

    return OutlierVariableResult(
        variable_name=str(series.name),
        method="iqr",
        outlier_count=len(outlier_indices),
        outlier_rate=len(outlier_indices) / len(valid),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        outlier_indices=outlier_indices,
        warnings=warnings,
    )


def detect_univariate_outliers(
    dataframe: pd.DataFrame,
    *,
    variables: list[str] | None = None,
    methods: tuple[str, ...] = ("zscore", "iqr"),
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> list[OutlierVariableResult]:
    """여러 숫자형 변수의 단변량 이상치를 일괄 탐지한다."""
    selected_variables = variables or [
        str(column) for column in dataframe.select_dtypes(include=[np.number]).columns
    ]

    results: list[OutlierVariableResult] = []

    for variable in selected_variables:
        if variable not in dataframe.columns:
            raise KeyError(f"데이터에 변수가 없습니다: {variable}")

        for method in methods:
            if method == "zscore":
                results.append(
                    detect_zscore_outliers(
                        dataframe[variable],
                        threshold=z_threshold,
                    )
                )
            elif method == "iqr":
                results.append(
                    detect_iqr_outliers(
                        dataframe[variable],
                        multiplier=iqr_multiplier,
                    )
                )
            else:
                raise ValueError(f"지원하지 않는 단변량 이상치 방법입니다: {method}")

    return results


def detect_mahalanobis_outliers(
    dataframe: pd.DataFrame,
    variables: list[str],
    *,
    significance_level: float = 0.001,
) -> MahalanobisResult:
    """완전사례 기준 Mahalanobis 거리 이상치를 탐지한다."""
    if len(variables) < 2:
        raise ValueError("Mahalanobis 거리에는 최소 2개 변수가 필요합니다.")

    if not 0 < significance_level < 1:
        raise ValueError("significance_level은 0과 1 사이여야 합니다.")

    missing_variables = [variable for variable in variables if variable not in dataframe.columns]
    if missing_variables:
        raise KeyError("데이터에 변수가 없습니다: " + ", ".join(missing_variables))

    numeric = dataframe[variables].apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = numeric.dropna()
    warnings: list[str] = []

    if len(complete) <= len(variables):
        raise ValueError("완전사례 수가 변수 수보다 많아야 합니다.")

    centered = complete - complete.mean(axis=0)
    covariance = np.cov(
        complete.to_numpy(),
        rowvar=False,
        ddof=1,
    )

    rank = np.linalg.matrix_rank(covariance)
    if rank < len(variables):
        warnings.append("공분산행렬이 특이하거나 준특이하여 의사역행렬을 사용했습니다.")

    inverse_covariance = np.linalg.pinv(covariance)

    distances_array = np.einsum(
        "ij,jk,ik->i",
        centered.to_numpy(),
        inverse_covariance,
        centered.to_numpy(),
    )

    distances = pd.Series(
        distances_array,
        index=complete.index,
        name="mahalanobis_distance",
    )

    degrees_of_freedom = len(variables)
    cutoff = float(
        chi2.ppf(
            1 - significance_level,
            df=degrees_of_freedom,
        )
    )
    outlier_mask = distances > cutoff
    outlier_indices = distances.index[outlier_mask].tolist()

    return MahalanobisResult(
        variables=variables,
        valid_case_count=len(complete),
        degrees_of_freedom=degrees_of_freedom,
        significance_level=significance_level,
        cutoff=cutoff,
        outlier_count=len(outlier_indices),
        outlier_rate=len(outlier_indices) / len(complete),
        distances=distances,
        outlier_indices=outlier_indices,
        warnings=warnings,
    )


def build_outlier_report(
    dataframe: pd.DataFrame,
    *,
    univariate_variables: list[str] | None = None,
    mahalanobis_variables: list[str] | None = None,
) -> OutlierReport:
    """단변량 및 다변량 이상치 보고서를 생성한다."""
    warnings: list[str] = []

    univariate_results = detect_univariate_outliers(
        dataframe,
        variables=univariate_variables,
    )

    mahalanobis_result: MahalanobisResult | None = None

    if mahalanobis_variables:
        try:
            mahalanobis_result = detect_mahalanobis_outliers(
                dataframe,
                mahalanobis_variables,
            )
        except (KeyError, ValueError) as error:
            warnings.append(str(error))

    return OutlierReport(
        univariate_results=univariate_results,
        mahalanobis_result=mahalanobis_result,
        warnings=warnings,
    )


def univariate_results_to_dataframe(
    results: list[OutlierVariableResult],
) -> pd.DataFrame:
    """단변량 이상치 결과를 검토용 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for result in results:
        row = asdict(result)
        row["outlier_indices"] = " | ".join(str(index) for index in result.outlier_indices)
        row["warnings"] = " | ".join(result.warnings)
        rows.append(row)

    return pd.DataFrame(rows)


def mahalanobis_distances_to_dataframe(
    result: MahalanobisResult,
) -> pd.DataFrame:
    """Mahalanobis 거리와 이상치 여부를 데이터프레임으로 변환한다."""
    output = result.distances.rename_axis("row_index").reset_index()
    output["cutoff"] = result.cutoff
    output["is_outlier"] = output["mahalanobis_distance"] > result.cutoff
    return output


def outlier_report_summary(
    report: OutlierReport,
) -> dict[str, Any]:
    """전체 이상치 진단 결과를 요약한다."""
    univariate_flag_count = sum(result.outlier_count for result in report.univariate_results)

    return {
        "univariate_result_count": len(report.univariate_results),
        "univariate_flag_count": univariate_flag_count,
        "mahalanobis_available": (report.mahalanobis_result is not None),
        "mahalanobis_outlier_count": (
            report.mahalanobis_result.outlier_count if report.mahalanobis_result else 0
        ),
        "warning_count": len(report.warnings),
    }
