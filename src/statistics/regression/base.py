"""회귀모형 공통 자료구조와 데이터 준비 기능."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ModelCoefficient:
    """회귀계수 한 행의 표준화된 결과."""

    term: str
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    exponentiated_estimate: float | None = None


@dataclass(slots=True)
class RegressionResult:
    """모든 회귀모형이 반환하는 공통 결과."""

    model_id: str
    model_type: str
    dependent_variable: str
    independent_variables: list[str]
    sample_size: int
    coefficients: list[ModelCoefficient]
    fit_statistics: dict[str, Any]
    converged: bool
    standard_error_type: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_result: Any | None = None


def validate_model_variables(
    dataframe: pd.DataFrame,
    dependent_variable: str,
    independent_variables: list[str],
) -> None:
    """회귀모형에 필요한 변수가 존재하는지 검사한다."""
    if not dependent_variable.strip():
        raise ValueError("종속변수는 비어 있을 수 없습니다.")

    if not independent_variables:
        raise ValueError("독립변수를 한 개 이상 지정해야 합니다.")

    requested = [dependent_variable, *independent_variables]
    duplicated = [
        variable for variable in dict.fromkeys(requested) if requested.count(variable) > 1
    ]
    if duplicated:
        raise ValueError(
            "종속변수와 독립변수 목록에 중복 변수가 있습니다: " + ", ".join(duplicated)
        )

    missing = [variable for variable in requested if variable not in dataframe.columns]
    if missing:
        raise KeyError("데이터에 변수가 없습니다: " + ", ".join(missing))


def prepare_model_data(
    dataframe: pd.DataFrame,
    dependent_variable: str,
    independent_variables: list[str],
) -> pd.DataFrame:
    """
    회귀모형 변수만 선택하고 숫자형 변환 후 완전사례를 반환한다.

    원본 데이터프레임은 수정하지 않는다.
    """
    validate_model_variables(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    selected = dataframe[[dependent_variable, *independent_variables]].copy()

    for column in selected.columns:
        selected[column] = pd.to_numeric(
            selected[column],
            errors="coerce",
        )

    complete = selected.dropna()

    if complete.empty:
        raise ValueError("회귀분석에 사용할 완전사례가 없습니다.")

    if complete[dependent_variable].nunique() <= 1:
        raise ValueError("종속변수가 상수이거나 유효 범주가 하나뿐입니다.")

    constant_predictors = [
        variable for variable in independent_variables if complete[variable].nunique() <= 1
    ]
    if constant_predictors:
        raise ValueError("상수 독립변수가 있습니다: " + ", ".join(constant_predictors))

    return complete


def coefficients_to_dataframe(
    result: RegressionResult,
) -> pd.DataFrame:
    """공통 회귀계수 결과를 데이터프레임으로 변환한다."""
    return pd.DataFrame([asdict(coefficient) for coefficient in result.coefficients])


def fit_statistics_to_dataframe(
    result: RegressionResult,
) -> pd.DataFrame:
    """모형 적합도 정보를 세로형 표로 변환한다."""
    values = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "dependent_variable": result.dependent_variable,
        "sample_size": result.sample_size,
        "converged": result.converged,
        "standard_error_type": result.standard_error_type,
        **result.fit_statistics,
    }

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )


def regression_result_summary(
    result: RegressionResult,
) -> dict[str, Any]:
    """회귀결과 핵심 요약을 반환한다."""
    significant_terms = [
        coefficient.term
        for coefficient in result.coefficients
        if coefficient.p_value < 0.05 and coefficient.term.lower() not in {"const", "intercept"}
    ]

    return {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "dependent_variable": result.dependent_variable,
        "sample_size": result.sample_size,
        "converged": result.converged,
        "significant_terms": significant_terms,
        "warning_count": len(result.warnings),
    }
