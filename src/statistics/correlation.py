"""상관분석 및 논문용 상관표 생성 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

SUPPORTED_METHODS = {"pearson", "spearman", "kendall"}


@dataclass(slots=True)
class CorrelationResult:
    """변수쌍별 상관분석 결과."""

    variable_1: str
    variable_2: str
    method: str
    coefficient: float | None
    p_value: float | None
    adjusted_p_value: float | None
    sample_size: int
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    significant: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CorrelationReport:
    """전체 상관분석 결과."""

    results: list[CorrelationResult]
    coefficient_matrix: pd.DataFrame
    p_value_matrix: pd.DataFrame
    sample_size_matrix: pd.DataFrame
    publication_table: pd.DataFrame
    warnings: list[str]


def _pairwise_complete(
    dataframe: pd.DataFrame,
    left: str,
    right: str,
) -> pd.DataFrame:
    """두 변수의 완전사례만 반환한다."""
    return dataframe[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()


def _pearson_confidence_interval(
    coefficient: float,
    sample_size: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Fisher z 변환으로 Pearson 상관의 신뢰구간을 계산한다."""
    if sample_size <= 3 or abs(coefficient) >= 1:
        return None, None

    fisher_z = np.arctanh(coefficient)
    standard_error = 1 / np.sqrt(sample_size - 3)
    critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    lower = np.tanh(fisher_z - critical * standard_error)
    upper = np.tanh(fisher_z + critical * standard_error)

    return float(lower), float(upper)


def correlate_pair(
    dataframe: pd.DataFrame,
    variable_1: str,
    variable_2: str,
    *,
    method: str = "pearson",
) -> CorrelationResult:
    """변수 두 개의 상관계수, p값, N을 계산한다."""
    method = method.lower()

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"지원하지 않는 상관방법입니다: {method}")

    missing = [
        variable for variable in (variable_1, variable_2) if variable not in dataframe.columns
    ]
    if missing:
        raise KeyError("데이터에 변수가 없습니다: " + ", ".join(missing))

    pair = _pairwise_complete(
        dataframe,
        variable_1,
        variable_2,
    )
    sample_size = len(pair)
    warnings: list[str] = []

    if sample_size < 3:
        warnings.append("유효 사례가 3개 미만입니다.")
        return CorrelationResult(
            variable_1=variable_1,
            variable_2=variable_2,
            method=method,
            coefficient=None,
            p_value=None,
            adjusted_p_value=None,
            sample_size=sample_size,
            warnings=warnings,
        )

    if pair[variable_1].nunique() <= 1:
        warnings.append(f"{variable_1}이 상수 변수입니다.")
    if pair[variable_2].nunique() <= 1:
        warnings.append(f"{variable_2}가 상수 변수입니다.")

    if warnings:
        return CorrelationResult(
            variable_1=variable_1,
            variable_2=variable_2,
            method=method,
            coefficient=None,
            p_value=None,
            adjusted_p_value=None,
            sample_size=sample_size,
            warnings=warnings,
        )

    if method == "pearson":
        coefficient, p_value = stats.pearsonr(
            pair[variable_1],
            pair[variable_2],
        )
        lower, upper = _pearson_confidence_interval(
            float(coefficient),
            sample_size,
        )
    elif method == "spearman":
        coefficient, p_value = stats.spearmanr(
            pair[variable_1],
            pair[variable_2],
        )
        lower, upper = None, None
    else:
        coefficient, p_value = stats.kendalltau(
            pair[variable_1],
            pair[variable_2],
        )
        lower, upper = None, None

    return CorrelationResult(
        variable_1=variable_1,
        variable_2=variable_2,
        method=method,
        coefficient=float(coefficient),
        p_value=float(p_value),
        adjusted_p_value=None,
        sample_size=sample_size,
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
    )


def run_correlation_analysis(
    dataframe: pd.DataFrame,
    variables: list[str],
    *,
    method: str = "pearson",
    p_adjust_method: str = "holm",
    alpha: float = 0.05,
    high_correlation_threshold: float = 0.80,
) -> CorrelationReport:
    """복수 변수의 상관분석과 다중검정 보정을 수행한다."""
    if len(variables) < 2:
        raise ValueError("상관분석에는 최소 2개 변수가 필요합니다.")

    results = [
        correlate_pair(
            dataframe,
            left,
            right,
            method=method,
        )
        for left, right in combinations(variables, 2)
    ]

    valid_indices = [index for index, result in enumerate(results) if result.p_value is not None]
    valid_p_values = [results[index].p_value for index in valid_indices]

    if valid_p_values:
        _, adjusted, _, _ = multipletests(
            valid_p_values,
            alpha=alpha,
            method=p_adjust_method,
        )

        for index, adjusted_p_value in zip(
            valid_indices,
            adjusted,
            strict=True,
        ):
            results[index].adjusted_p_value = float(adjusted_p_value)
            results[index].significant = adjusted_p_value < alpha

    coefficient_matrix = pd.DataFrame(
        np.eye(len(variables)),
        index=variables,
        columns=variables,
        dtype=float,
    )
    p_value_matrix = pd.DataFrame(
        np.nan,
        index=variables,
        columns=variables,
        dtype=float,
    )
    sample_size_matrix = pd.DataFrame(
        np.nan,
        index=variables,
        columns=variables,
        dtype=float,
    )

    warnings: list[str] = []

    for result in results:
        coefficient_matrix.loc[
            result.variable_1,
            result.variable_2,
        ] = result.coefficient
        coefficient_matrix.loc[
            result.variable_2,
            result.variable_1,
        ] = result.coefficient

        p_value_matrix.loc[
            result.variable_1,
            result.variable_2,
        ] = result.adjusted_p_value
        p_value_matrix.loc[
            result.variable_2,
            result.variable_1,
        ] = result.adjusted_p_value

        sample_size_matrix.loc[
            result.variable_1,
            result.variable_2,
        ] = result.sample_size
        sample_size_matrix.loc[
            result.variable_2,
            result.variable_1,
        ] = result.sample_size

        if result.coefficient is not None and abs(result.coefficient) >= high_correlation_threshold:
            warnings.append(
                f"{result.variable_1}와 {result.variable_2}의 "
                f"상관계수 절대값이 {abs(result.coefficient):.3f}입니다."
            )

        warnings.extend(
            f"{result.variable_1}-{result.variable_2}: {warning}" for warning in result.warnings
        )

    publication_table = build_publication_correlation_table(
        dataframe,
        variables,
        coefficient_matrix,
        p_value_matrix,
    )

    return CorrelationReport(
        results=results,
        coefficient_matrix=coefficient_matrix,
        p_value_matrix=p_value_matrix,
        sample_size_matrix=sample_size_matrix,
        publication_table=publication_table,
        warnings=warnings,
    )


def build_publication_correlation_table(
    dataframe: pd.DataFrame,
    variables: list[str],
    coefficient_matrix: pd.DataFrame,
    p_value_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """평균·표준편차와 하삼각 상관계수를 포함한 논문용 표를 생성한다."""
    numeric = dataframe[variables].apply(
        pd.to_numeric,
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []

    for row_index, variable in enumerate(variables, start=1):
        row: dict[str, Any] = {
            "번호": row_index,
            "변수": variable,
            "평균": float(numeric[variable].mean()),
            "표준편차": float(numeric[variable].std(ddof=1)),
        }

        for column_index, other in enumerate(
            variables,
            start=1,
        ):
            if column_index >= row_index:
                row[str(column_index)] = ""
                continue

            coefficient = coefficient_matrix.loc[
                variable,
                other,
            ]
            p_value = p_value_matrix.loc[
                variable,
                other,
            ]

            if pd.isna(coefficient):
                formatted = ""
            else:
                stars = ""
                if pd.notna(p_value):
                    if p_value < 0.001:
                        stars = "***"
                    elif p_value < 0.01:
                        stars = "**"
                    elif p_value < 0.05:
                        stars = "*"

                formatted = f"{coefficient:.3f}{stars}"

            row[str(column_index)] = formatted

        rows.append(row)

    return pd.DataFrame(rows)


def correlation_results_to_dataframe(
    report: CorrelationReport,
) -> pd.DataFrame:
    """변수쌍별 상관분석 결과를 데이터프레임으로 변환한다."""
    rows: list[dict[str, Any]] = []

    for result in report.results:
        row = asdict(result)
        row["warnings"] = " | ".join(result.warnings)
        rows.append(row)

    return pd.DataFrame(rows)
