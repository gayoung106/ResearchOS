"""회귀모형 공통 설계행렬 생성 기능."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.statistics.regression.base import (
    validate_model_variables,
)


@dataclass(slots=True)
class RegressionDesignMatrix:
    """회귀모형에 사용할 종속변수와 설계행렬."""

    outcome: pd.Series
    predictors: pd.DataFrame
    fixed_effect_columns: list[str]
    metadata: dict[str, Any]


def prepare_regression_design_matrix(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_label: str = "회귀",
) -> RegressionDesignMatrix:
    """
    회귀분석 공통 설계행렬을 생성한다.

    일반 독립변수는 숫자형으로 변환한다. 고정효과 변수는 결정론적으로
    정렬한 첫 번째 범주를 기준범주로 사용하고 k-1개 더미변수로 변환한다.
    """
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))

    validate_model_variables(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    _validate_fixed_effects(
        dataframe,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
    )

    requested_columns = [
        dependent_variable,
        *independent_variables,
        *fixed_effects,
    ]
    selected = dataframe[requested_columns].copy()

    selected[dependent_variable] = pd.to_numeric(
        selected[dependent_variable],
        errors="coerce",
    )

    for variable in independent_variables:
        selected[variable] = pd.to_numeric(
            selected[variable],
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

    outcome = complete[dependent_variable].astype(float)
    predictors = complete[independent_variables].astype(float).copy()

    (
        predictors,
        fixed_effect_columns,
        reference_categories,
    ) = _encode_fixed_effects(
        complete,
        predictors=predictors,
        fixed_effects=fixed_effects,
    )

    if predictors.empty:
        raise ValueError(f"{model_label} 모형에 사용할 설명변수가 없습니다.")

    return RegressionDesignMatrix(
        outcome=outcome,
        predictors=predictors,
        fixed_effect_columns=fixed_effect_columns,
        metadata={
            "fixed_effects": fixed_effects,
            "fixed_effect_reference_categories": (reference_categories),
            "fixed_effect_columns": (fixed_effect_columns),
            "dropped_case_count": (len(dataframe) - len(complete)),
        },
    )


def _validate_fixed_effects(
    dataframe: pd.DataFrame,
    *,
    independent_variables: list[str],
    fixed_effects: list[str],
) -> None:
    """고정효과 변수의 중복과 존재 여부를 검사한다."""
    duplicated_fixed_effects = [
        variable for variable in fixed_effects if variable in independent_variables
    ]
    if duplicated_fixed_effects:
        raise ValueError(
            "고정효과 변수가 일반 독립변수에도 "
            "중복 지정되었습니다: " + ", ".join(duplicated_fixed_effects)
        )

    missing_fixed_effects = [
        variable for variable in fixed_effects if variable not in dataframe.columns
    ]
    if missing_fixed_effects:
        raise KeyError("데이터에 고정효과 변수가 없습니다: " + ", ".join(missing_fixed_effects))


def _encode_fixed_effects(
    complete: pd.DataFrame,
    *,
    predictors: pd.DataFrame,
    fixed_effects: list[str],
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, Any],
]:
    """고정효과 변수를 기준범주 제외 더미변수로 변환한다."""
    encoded_predictors = predictors.copy()
    encoded_columns: list[str] = []
    reference_categories: dict[str, Any] = {}

    for fixed_effect in fixed_effects:
        categories = _ordered_categories(complete[fixed_effect])

        if len(categories) <= 1:
            raise ValueError(f"고정효과 변수의 유효 범주가 하나뿐입니다: {fixed_effect}")

        reference_category = categories[0]
        reference_categories[fixed_effect] = reference_category

        categorical_values = pd.Categorical(
            complete[fixed_effect],
            categories=categories,
        )

        dummy_frame = pd.get_dummies(
            categorical_values,
            prefix=fixed_effect,
            prefix_sep="_",
            drop_first=True,
            dtype=float,
        )
        dummy_frame.index = complete.index

        collisions = [
            column for column in dummy_frame.columns if column in encoded_predictors.columns
        ]
        if collisions:
            raise ValueError(
                "고정효과 더미변수명이 기존 독립변수와 충돌합니다: " + ", ".join(collisions)
            )

        duplicate_dummy_columns = [
            column for column in dummy_frame.columns if column in encoded_columns
        ]
        if duplicate_dummy_columns:
            raise ValueError(
                "고정효과 더미변수명이 서로 충돌합니다: " + ", ".join(duplicate_dummy_columns)
            )

        encoded_predictors = pd.concat(
            [
                encoded_predictors,
                dummy_frame,
            ],
            axis=1,
        )
        encoded_columns.extend(dummy_frame.columns.tolist())

    return (
        encoded_predictors,
        encoded_columns,
        reference_categories,
    )


def _ordered_categories(
    series: pd.Series,
) -> list[Any]:
    """결측을 제외한 범주를 결정론적 순서로 반환한다."""
    categories = series.dropna().drop_duplicates().tolist()

    try:
        return sorted(categories)
    except TypeError:
        return sorted(
            categories,
            key=lambda value: str(value),
        )
