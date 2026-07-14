"""전처리 실행기 테스트."""

import pandas as pd

from src.preprocess.executor import (
    execute_preprocessing_plan,
    execution_summary,
)
from src.preprocess.planner import (
    PreprocessingAction,
    PreprocessingPlan,
)


def plan_with(
    *actions: PreprocessingAction,
) -> PreprocessingPlan:
    return PreprocessingPlan(
        actions=list(actions),
        warnings=[],
        blocked_variables=[],
    )


def approved_action(
    variable_name: str,
    action_type: str,
    parameters: dict | None = None,
) -> PreprocessingAction:
    return PreprocessingAction(
        variable_name=variable_name,
        action_type=action_type,
        status="approved",
        reason="테스트",
        parameters=parameters or {},
        requires_confirmation=True,
    )


def test_original_dataframe_is_not_modified() -> None:
    original = pd.DataFrame({"x": [1, 9, 2]})
    plan = plan_with(
        approved_action(
            "x",
            "replace_missing_values",
            {"missing_values": [9]},
        )
    )

    result = execute_preprocessing_plan(original, plan)

    assert original["x"].isna().sum() == 0
    assert result.dataframe["x"].isna().sum() == 1


def test_unapproved_action_is_skipped() -> None:
    dataframe = pd.DataFrame({"x": [1, 9, 2]})
    action = PreprocessingAction(
        variable_name="x",
        action_type="replace_missing_values",
        status="planned",
        reason="테스트",
        parameters={"missing_values": [9]},
    )

    result = execute_preprocessing_plan(
        dataframe,
        plan_with(action),
        require_approval=True,
    )

    assert result.dataframe["x"].isna().sum() == 0
    assert result.records[0].status == "skipped"


def test_reverse_coding() -> None:
    dataframe = pd.DataFrame({"q1": [1, 2, 5]})
    plan = plan_with(
        approved_action(
            "q1",
            "reverse_code",
            {
                "coding": {
                    "min": 1,
                    "max": 5,
                }
            },
        )
    )

    result = execute_preprocessing_plan(dataframe, plan)

    assert result.dataframe["q1"].tolist() == [5.0, 4.0, 1.0]
    assert result.records[0].status == "completed"


def test_numeric_string_mapping_is_normalized() -> None:
    dataframe = pd.DataFrame({"gender": [1, 2, 1]})
    plan = plan_with(
        approved_action(
            "gender",
            "configured_recoding",
            {
                "mapping": {
                    "1": 0,
                    "2": 1,
                }
            },
        )
    )

    result = execute_preprocessing_plan(dataframe, plan)

    assert result.dataframe["gender"].tolist() == [0, 1, 0]


def test_mean_center_creates_new_variable() -> None:
    dataframe = pd.DataFrame({"age": [20, 30, 40]})
    plan = plan_with(
        approved_action(
            "age",
            "mean_center",
            {"output_name": "age_c"},
        )
    )

    result = execute_preprocessing_plan(dataframe, plan)

    assert result.dataframe["age_c"].tolist() == [
        -10.0,
        0.0,
        10.0,
    ]


def test_derived_variable_creation() -> None:
    dataframe = pd.DataFrame({"age": [2, 3, 4]})
    plan = plan_with(
        approved_action(
            "age_squared",
            "create_derived_variable",
            {
                "name": "age_squared",
                "expression": "age ** 2",
            },
        )
    )

    result = execute_preprocessing_plan(dataframe, plan)

    assert result.dataframe["age_squared"].tolist() == [4, 9, 16]


def test_failed_action_is_recorded() -> None:
    dataframe = pd.DataFrame({"x": [1, 2, 3]})
    plan = plan_with(
        approved_action(
            "missing_variable",
            "replace_missing_values",
            {"missing_values": [9]},
        )
    )

    result = execute_preprocessing_plan(dataframe, plan)

    assert result.records[0].status == "failed"
    assert result.warnings


def test_execution_summary() -> None:
    dataframe = pd.DataFrame({"x": [1, 9, 2]})
    plan = plan_with(
        approved_action(
            "x",
            "replace_missing_values",
            {"missing_values": [9]},
        )
    )

    result = execute_preprocessing_plan(dataframe, plan)
    summary = execution_summary(result)

    assert summary["record_count"] == 1
    assert summary["status_counts"]["completed"] == 1
    assert summary["output_row_count"] == 3
