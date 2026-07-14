"""승인된 전처리 계획을 데이터 복사본에 적용하는 실행기."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.preprocess.planner import PreprocessingAction, PreprocessingPlan

APPROVED_STATUSES = {"approved", "planned"}


@dataclass(slots=True)
class ExecutionRecord:
    """개별 전처리 작업의 실행 결과."""

    variable_name: str
    action_type: str
    status: str
    message: str
    before_missing: int | None = None
    after_missing: int | None = None
    before_unique: int | None = None
    after_unique: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreprocessingExecutionResult:
    """전체 전처리 실행 결과."""

    dataframe: pd.DataFrame
    records: list[ExecutionRecord]
    warnings: list[str]


def execute_preprocessing_plan(
    dataframe: pd.DataFrame,
    plan: PreprocessingPlan,
    *,
    require_approval: bool = True,
) -> PreprocessingExecutionResult:
    """
    전처리 계획을 데이터프레임 복사본에 적용한다.

    원본 데이터프레임은 변경하지 않는다.
    """
    working = dataframe.copy(deep=True)
    records: list[ExecutionRecord] = []
    warnings = list(plan.warnings)

    for action in plan.actions:
        if require_approval and action.status != "approved":
            records.append(
                ExecutionRecord(
                    variable_name=action.variable_name,
                    action_type=action.action_type,
                    status="skipped",
                    message="승인되지 않은 작업이므로 실행하지 않았습니다.",
                )
            )
            continue

        try:
            record = _execute_action(working, action)
        except Exception as error:
            records.append(
                ExecutionRecord(
                    variable_name=action.variable_name,
                    action_type=action.action_type,
                    status="failed",
                    message=str(error),
                )
            )
            warnings.append(f"{action.variable_name}/{action.action_type}: {error}")
            continue

        records.append(record)

    return PreprocessingExecutionResult(
        dataframe=working,
        records=records,
        warnings=warnings,
    )


def _execute_action(
    dataframe: pd.DataFrame,
    action: PreprocessingAction,
) -> ExecutionRecord:
    """개별 전처리 작업을 실행한다."""
    if action.action_type == "replace_missing_values":
        return _replace_missing_values(dataframe, action)

    if action.action_type == "reverse_code":
        return _reverse_code(dataframe, action)

    if action.action_type in {
        "review_binary_recoding",
        "configured_recoding",
    }:
        return _recode_values(dataframe, action)

    if action.action_type == "mean_center":
        return _mean_center(dataframe, action)

    if action.action_type == "create_derived_variable":
        return _create_derived_variable(dataframe, action)

    if action.action_type in {
        "set_reference_category",
        "review_centering",
        "assign_scale_item",
    }:
        return ExecutionRecord(
            variable_name=action.variable_name,
            action_type=action.action_type,
            status="not_applicable",
            message="메타데이터 또는 후속 분석설정 작업이므로 데이터값은 변경하지 않았습니다.",
        )

    if action.action_type == "custom_rule":
        raise ValueError("custom_rule은 명시적인 실행 규칙이 없어 자동 실행할 수 없습니다.")

    raise ValueError(f"지원하지 않는 전처리 작업입니다: {action.action_type}")


def _validate_variable_exists(
    dataframe: pd.DataFrame,
    variable_name: str,
) -> None:
    if variable_name not in dataframe.columns:
        raise KeyError(f"데이터에 변수가 없습니다: {variable_name}")


def _replace_missing_values(
    dataframe: pd.DataFrame,
    action: PreprocessingAction,
) -> ExecutionRecord:
    _validate_variable_exists(dataframe, action.variable_name)

    series = dataframe[action.variable_name]
    before_missing = int(series.isna().sum())
    before_unique = int(series.nunique(dropna=True))
    missing_values = action.parameters.get("missing_values", [])

    if not isinstance(missing_values, list):
        raise ValueError("missing_values는 목록이어야 합니다.")

    dataframe[action.variable_name] = series.replace(
        missing_values,
        np.nan,
    )

    updated = dataframe[action.variable_name]

    return ExecutionRecord(
        variable_name=action.variable_name,
        action_type=action.action_type,
        status="completed",
        message="사용자 정의 결측값을 NaN으로 변환했습니다.",
        before_missing=before_missing,
        after_missing=int(updated.isna().sum()),
        before_unique=before_unique,
        after_unique=int(updated.nunique(dropna=True)),
        details={"missing_values": missing_values},
    )


def _reverse_code(
    dataframe: pd.DataFrame,
    action: PreprocessingAction,
) -> ExecutionRecord:
    _validate_variable_exists(dataframe, action.variable_name)

    coding = action.parameters.get("coding", {})
    minimum = coding.get("min")
    maximum = coding.get("max")

    if minimum is None or maximum is None:
        raise ValueError("역코딩에는 coding.min과 coding.max가 필요합니다.")

    series = dataframe[action.variable_name]
    numeric = pd.to_numeric(series, errors="coerce")

    invalid_count = int(series.notna().sum() - numeric.notna().sum())
    if invalid_count:
        raise ValueError(f"숫자로 변환할 수 없는 값이 {invalid_count}개 있습니다.")

    before_missing = int(series.isna().sum())
    before_unique = int(series.nunique(dropna=True))

    dataframe[action.variable_name] = float(minimum) + float(maximum) - numeric

    updated = dataframe[action.variable_name]

    return ExecutionRecord(
        variable_name=action.variable_name,
        action_type=action.action_type,
        status="completed",
        message="역문항 코딩을 적용했습니다.",
        before_missing=before_missing,
        after_missing=int(updated.isna().sum()),
        before_unique=before_unique,
        after_unique=int(updated.nunique(dropna=True)),
        details={
            "minimum": minimum,
            "maximum": maximum,
        },
    )


def _recode_values(
    dataframe: pd.DataFrame,
    action: PreprocessingAction,
) -> ExecutionRecord:
    _validate_variable_exists(dataframe, action.variable_name)

    mapping = action.parameters.get("mapping")
    if mapping is None:
        mapping = action.parameters.get("coding")

    if not isinstance(mapping, dict) or not mapping:
        return ExecutionRecord(
            variable_name=action.variable_name,
            action_type=action.action_type,
            status="skipped",
            message="재코딩 매핑이 없어 값을 변경하지 않았습니다.",
        )

    series = dataframe[action.variable_name]
    before_missing = int(series.isna().sum())
    before_unique = int(series.nunique(dropna=True))

    normalized_mapping = _normalize_mapping_keys(series, mapping)
    dataframe[action.variable_name] = series.replace(normalized_mapping)

    updated = dataframe[action.variable_name]

    return ExecutionRecord(
        variable_name=action.variable_name,
        action_type=action.action_type,
        status="completed",
        message="지정된 매핑에 따라 값을 재코딩했습니다.",
        before_missing=before_missing,
        after_missing=int(updated.isna().sum()),
        before_unique=before_unique,
        after_unique=int(updated.nunique(dropna=True)),
        details={"mapping": normalized_mapping},
    )


def _normalize_mapping_keys(
    series: pd.Series,
    mapping: dict[Any, Any],
) -> dict[Any, Any]:
    """
    YAML에서 문자열로 읽힌 숫자 키를 변수 자료형에 맞게 보정한다.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return mapping

    normalized: dict[Any, Any] = {}

    for key, value in mapping.items():
        if isinstance(key, str):
            try:
                numeric_key = float(key)
                if numeric_key.is_integer():
                    numeric_key = int(numeric_key)
                normalized[numeric_key] = value
                continue
            except ValueError:
                pass

        normalized[key] = value

    return normalized


def _mean_center(
    dataframe: pd.DataFrame,
    action: PreprocessingAction,
) -> ExecutionRecord:
    _validate_variable_exists(dataframe, action.variable_name)

    series = pd.to_numeric(
        dataframe[action.variable_name],
        errors="coerce",
    )

    invalid_count = int(dataframe[action.variable_name].notna().sum() - series.notna().sum())
    if invalid_count:
        raise ValueError(f"숫자로 변환할 수 없는 값이 {invalid_count}개 있습니다.")

    mean_value = float(series.mean())
    output_name = action.parameters.get(
        "output_name",
        f"{action.variable_name}_centered",
    )

    if output_name in dataframe.columns:
        raise ValueError(f"중심화 결과 변수명이 이미 존재합니다: {output_name}")

    dataframe[output_name] = series - mean_value

    return ExecutionRecord(
        variable_name=action.variable_name,
        action_type=action.action_type,
        status="completed",
        message=f"평균중심화 변수 {output_name}을 생성했습니다.",
        before_missing=int(series.isna().sum()),
        after_missing=int(dataframe[output_name].isna().sum()),
        before_unique=int(series.nunique(dropna=True)),
        after_unique=int(dataframe[output_name].nunique(dropna=True)),
        details={
            "mean": mean_value,
            "output_name": output_name,
        },
    )


def _create_derived_variable(
    dataframe: pd.DataFrame,
    action: PreprocessingAction,
) -> ExecutionRecord:
    parameters = action.parameters
    variable_name = str(parameters.get("name", action.variable_name))
    expression = parameters.get("expression")

    if not expression or not isinstance(expression, str):
        raise ValueError("파생변수 생성에는 문자열 expression이 필요합니다.")

    if variable_name in dataframe.columns:
        raise ValueError(f"파생변수명이 이미 존재합니다: {variable_name}")

    try:
        dataframe[variable_name] = dataframe.eval(
            expression,
            engine="python",
        )
    except Exception as error:
        raise ValueError(f"파생변수 수식을 실행할 수 없습니다: {expression}") from error

    created = dataframe[variable_name]

    return ExecutionRecord(
        variable_name=variable_name,
        action_type=action.action_type,
        status="completed",
        message="파생변수를 생성했습니다.",
        after_missing=int(created.isna().sum()),
        after_unique=int(created.nunique(dropna=True)),
        details={"expression": expression},
    )


def execution_records_to_dataframe(
    result: PreprocessingExecutionResult,
) -> pd.DataFrame:
    """실행 기록을 검토용 데이터프레임으로 변환한다."""
    return pd.DataFrame([asdict(record) for record in result.records])


def execution_summary(
    result: PreprocessingExecutionResult,
) -> dict[str, Any]:
    """전처리 실행 결과 요약을 반환한다."""
    status_counts: dict[str, int] = {}

    for record in result.records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1

    return {
        "record_count": len(result.records),
        "status_counts": status_counts,
        "warning_count": len(result.warnings),
        "output_row_count": len(result.dataframe),
        "output_column_count": len(result.dataframe.columns),
    }
