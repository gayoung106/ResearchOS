"""척도 신뢰도 분석 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ReliabilityResult:
    """척도 신뢰도 분석 결과."""

    scale_name: str
    item_count: int
    complete_case_count: int
    cronbach_alpha: float | None
    mean_inter_item_correlation: float | None
    warnings: list[str]


def cronbach_alpha(dataframe: pd.DataFrame) -> float:
    """완전사례 기준 Cronbach's alpha를 계산한다."""
    complete = dataframe.dropna()

    if complete.shape[1] < 2:
        raise ValueError("Cronbach alpha에는 최소 2개 문항이 필요합니다.")

    if complete.shape[0] < 2:
        raise ValueError("완전사례가 2개 미만입니다.")

    item_variances = complete.var(axis=0, ddof=1)
    total_score = complete.sum(axis=1)
    total_variance = total_score.var(ddof=1)

    if np.isclose(total_variance, 0.0):
        raise ValueError("총점 분산이 0이므로 alpha를 계산할 수 없습니다.")

    item_count = complete.shape[1]
    alpha = item_count / (item_count - 1) * (1 - item_variances.sum() / total_variance)

    return float(alpha)


def corrected_item_total_correlations(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """각 문항과 나머지 문항 총점 간 상관을 계산한다."""
    complete = dataframe.dropna()

    if complete.shape[1] < 2:
        raise ValueError("문항-총점 상관에는 최소 2개 문항이 필요합니다.")

    correlations: dict[str, float] = {}

    for column in complete.columns:
        other_total = complete.drop(columns=[column]).sum(axis=1)
        correlation = complete[column].corr(other_total)
        correlations[str(column)] = float(correlation)

    return pd.Series(
        correlations,
        name="corrected_item_total_correlation",
    )


def alpha_if_item_deleted(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """각 문항 삭제 시 Cronbach's alpha를 계산한다."""
    results: dict[str, float] = {}

    for column in dataframe.columns:
        reduced = dataframe.drop(columns=[column])

        if reduced.shape[1] < 2:
            results[str(column)] = np.nan
            continue

        try:
            results[str(column)] = cronbach_alpha(reduced)
        except ValueError:
            results[str(column)] = np.nan

    return pd.Series(
        results,
        name="alpha_if_item_deleted",
    )


def mean_inter_item_correlation(
    dataframe: pd.DataFrame,
) -> float:
    """문항 간 상관계수의 평균을 계산한다."""
    complete = dataframe.dropna()

    if complete.shape[1] < 2:
        raise ValueError("문항 간 상관에는 최소 2개 문항이 필요합니다.")

    correlation_matrix = complete.corr()
    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1,
        ).astype(bool)
    )

    values = upper_triangle.stack()

    if values.empty:
        raise ValueError("문항 간 상관을 계산할 수 없습니다.")

    return float(values.mean())


def run_reliability_analysis(
    dataframe: pd.DataFrame,
    *,
    scale_name: str,
) -> tuple[ReliabilityResult, pd.DataFrame]:
    """척도 신뢰도와 문항 수준 진단표를 계산한다."""
    numeric = dataframe.apply(
        pd.to_numeric,
        errors="coerce",
    )
    complete = numeric.dropna()
    warnings: list[str] = []

    if numeric.shape[1] < 2:
        raise ValueError("신뢰도 분석에는 최소 2개 문항이 필요합니다.")

    if complete.shape[0] < 2:
        raise ValueError("완전사례가 2개 미만입니다.")

    try:
        alpha = cronbach_alpha(numeric)
    except ValueError as error:
        alpha = None
        warnings.append(str(error))

    try:
        mean_correlation = mean_inter_item_correlation(numeric)
    except ValueError as error:
        mean_correlation = None
        warnings.append(str(error))

    item_total = corrected_item_total_correlations(numeric)
    deleted_alpha = alpha_if_item_deleted(numeric)

    item_table = pd.DataFrame(
        {
            "item": numeric.columns.astype(str),
            "mean": numeric.mean().to_numpy(),
            "standard_deviation": numeric.std(ddof=1).to_numpy(),
            "missing_count": numeric.isna().sum().to_numpy(),
            "corrected_item_total_correlation": item_total.reindex(
                numeric.columns.astype(str)
            ).to_numpy(),
            "alpha_if_item_deleted": deleted_alpha.reindex(numeric.columns.astype(str)).to_numpy(),
        }
    )

    result = ReliabilityResult(
        scale_name=scale_name,
        item_count=numeric.shape[1],
        complete_case_count=complete.shape[0],
        cronbach_alpha=alpha,
        mean_inter_item_correlation=mean_correlation,
        warnings=warnings,
    )

    return result, item_table


def reliability_result_to_dataframe(
    result: ReliabilityResult,
) -> pd.DataFrame:
    """신뢰도 결과를 세로형 데이터프레임으로 변환한다."""
    data = asdict(result)
    data["warnings"] = " | ".join(result.warnings)

    return pd.DataFrame(
        {
            "item": list(data.keys()),
            "value": list(data.values()),
        }
    )


def reliability_summary(
    result: ReliabilityResult,
) -> dict[str, Any]:
    """신뢰도 결과 요약을 반환한다."""
    return {
        "scale_name": result.scale_name,
        "item_count": result.item_count,
        "complete_case_count": result.complete_case_count,
        "cronbach_alpha": result.cronbach_alpha,
        "mean_inter_item_correlation": (result.mean_inter_item_correlation),
        "warning_count": len(result.warnings),
    }
