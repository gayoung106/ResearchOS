from pathlib import Path

import pandas as pd

from src.auto.variable_inference import (
    AutoVariableInferenceStep,
    build_auto_variable_map,
    role_inferences_to_dataframe,
    variable_map_to_dataframe,
)
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime


def _analysis_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "respondent_id": [101, 102, 103, 104, 105, 106],
            "wave": [1, 1, 2, 2, 3, 3],
            "cluster_id": [1, 1, 2, 2, 3, 3],
            "sample_weight": [1.1, 0.9, 1.0, 1.2, 0.8, 1.3],
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2],
            "age": [21, 35, 44, 51, 39, 28],
            "gender": [0, 1, 1, 0, 1, 0],
        }
    )


def test_build_auto_variable_map_infers_roles_and_levels() -> None:
    result = build_auto_variable_map(_analysis_dataframe())
    roles = {item.variable_name: item.role for item in result.role_inferences}
    levels = {name: definition.measurement_level for name, definition in result.variable_map.variables.items()}

    assert roles["respondent_id"] == "id"
    assert roles["wave"] == "time"
    assert roles["cluster_id"] == "cluster"
    assert roles["sample_weight"] == "weight"
    assert roles["outcome_score"] == "dependent"
    assert roles["age"] == "independent"
    assert levels["outcome_score"] == "continuous"
    assert result.variable_map.variables["outcome_score"].review_status == "auto_inferred"


def test_auto_variable_inference_prefers_group_name_for_cluster() -> None:
    data = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "y": [1, 2, 3, 4, 5, 6],
            "x": [2, 3, 4, 5, 6, 7],
        }
    )

    result = build_auto_variable_map(data)
    roles = {item.variable_name: item.role for item in result.role_inferences}

    assert roles["group"] == "cluster"
    assert roles["y"] == "dependent"
    assert roles["x"] == "independent"


def test_variable_inference_dataframes_are_exportable() -> None:
    result = build_auto_variable_map(_analysis_dataframe())

    roles = role_inferences_to_dataframe(result.role_inferences)
    variable_map = variable_map_to_dataframe(result.variable_map)

    assert {"variable_name", "role", "measurement_level", "confidence"}.issubset(roles.columns)
    assert {"variable_name", "role", "measurement_level", "reason"}.issubset(variable_map.columns)
    assert variable_map.shape[0] == len(_analysis_dataframe().columns)


def test_auto_variable_inference_step_populates_runtime_and_outputs(tmp_path: Path) -> None:
    runtime = PipelineRuntime(dataframe=_analysis_dataframe())

    step_result = AutoVariableInferenceStep(runtime).run(
        ResearchContext(project_name="auto variable inference"),
        tmp_path,
    )

    assert step_result.success is True
    assert len(runtime.detections) == len(_analysis_dataframe().columns)
    assert runtime.get_artifact("auto_variable_map").variables["outcome_score"].role == "dependent"
    assert runtime.get_artifact("auto_variable_inference_result").role_inferences
    assert {Path(path).name for path in step_result.output_files} == {
        "variable_detections.xlsx",
        "variable_role_inference.xlsx",
        "inferred_variable_map.xlsx",
    }


def test_auto_variable_inference_uses_korean_labels_for_roles() -> None:
    data = pd.DataFrame(
        {
            "q2": [3.1, 3.4, 3.7, 4.0, 4.1, 4.3, 4.4, 4.7],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "school_code": [1, 1, 2, 2, 3, 3, 4, 4],
            "q1": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
        }
    )
    metadata = pd.DataFrame(
        {
            "variable_name": ["q2", "age", "school_code", "q1"],
            "variable_label": [
                "\uc0c1\uc0ac \uc9c0\uc6d0 \uc778\uc2dd",
                "\uc5f0\ub839",
                "\uc18c\uc18d \ud559\uad50",
                "\uc9c1\ubb34 \ub9cc\uc871\ub3c4 \ucd1d\uc810",
            ],
            "question_text": [
                "\uc0c1\uc0ac\uac00 \uc5bc\ub9c8\ub098 \uc9c0\uc6d0\ud569\ub2c8\uae4c?",
                "\ub9cc \ub098\uc774",
                "\ud604\uc7ac \uc18c\uc18d\ub41c \ud559\uad50",
                "\uc804\ubc18\uc801\uc778 \uc9c1\ubb34 \ub9cc\uc871\ub3c4 \uc810\uc218",
            ],
        }
    )

    result = build_auto_variable_map(data, variable_metadata=metadata)
    roles = {item.variable_name: item.role for item in result.role_inferences}
    variable_map = variable_map_to_dataframe(result.variable_map)

    assert roles["q1"] == "dependent"
    assert roles["q2"] == "independent"
    assert roles["school_code"] == "cluster"
    assert result.variable_map.variables["q1"].label == "\uc9c1\ubb34 \ub9cc\uc871\ub3c4 \ucd1d\uc810"
    assert result.variable_map.variables["q1"].question_text == (
        "\uc804\ubc18\uc801\uc778 \uc9c1\ubb34 \ub9cc\uc871\ub3c4 \uc810\uc218"
    )
    assert {"label", "question_text"}.issubset(variable_map.columns)
