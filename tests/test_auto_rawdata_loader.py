from pathlib import Path

import pandas as pd

from src.auto.rawdata_loader import (
    AutoRawDataLoadingStep,
    discover_metadata_files,
    discover_rawdata_files,
    load_rawdata_project,
)
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime


def test_discover_rawdata_files_finds_supported_files(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    (rawdata / "notes.md").write_text("ignore", encoding="utf-8")
    pd.DataFrame({"y": [1, 2], "x": [3, 4]}).to_csv(rawdata / "data.csv", index=False)

    files = discover_rawdata_files(tmp_path)

    assert [path.name for path in files] == ["data.csv"]


def test_load_rawdata_project_selects_largest_usable_dataset(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    pd.DataFrame({"y": [1], "x": [2]}).to_csv(rawdata / "small.csv", index=False)
    pd.DataFrame({"y": [1, 2, 3, 4], "x": [2, 3, 4, 5], "z": [5, 6, 7, 8]}).to_csv(
        rawdata / "large.csv",
        index=False,
    )

    result = load_rawdata_project(tmp_path)

    assert result.selected_candidate.source_path.name == "large.csv"
    assert result.dataframe.shape == (4, 3)
    assert result.variable_metadata.shape[0] == 3
    assert len(result.candidates) == 2


def test_load_rawdata_project_scores_excel_sheets(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    path = rawdata / "workbook.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="tiny", index=False)
        pd.DataFrame({"y": [1, 2, 3], "x": [2, 4, 6]}).to_excel(
            writer,
            sheet_name="analysis",
            index=False,
        )

    result = load_rawdata_project(tmp_path)

    assert result.selected_candidate.source_path.name == "workbook.xlsx"
    assert result.selected_candidate.sheet_name == "analysis"
    assert result.dataframe.shape == (3, 2)
    assert len(result.candidates) == 2


def test_auto_rawdata_loading_step_populates_runtime_and_outputs(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    pd.DataFrame({"y": [1, 2, 3], "x": [3, 4, 5]}).to_csv(rawdata / "data.csv", index=False)
    runtime = PipelineRuntime()

    step_result = AutoRawDataLoadingStep(runtime).run(
        ResearchContext(project_name="auto rawdata"),
        tmp_path,
    )

    assert step_result.success is True
    assert runtime.dataframe is not None
    assert runtime.dataframe.shape == (3, 2)
    assert runtime.variable_metadata is not None
    assert runtime.get_artifact("auto_rawdata_load_result").selected_candidate.source_path.name == "data.csv"
    assert {Path(path).name for path in step_result.output_files} == {
        "analysis_base.parquet",
        "variable_metadata.xlsx",
        "rawdata_candidates.xlsx",
    }


def test_load_rawdata_project_enriches_metadata_from_codebook(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    pd.DataFrame(
        {
            "q1": [1, 2, 3, 4],
            "age": [21, 35, 44, 51],
        }
    ).to_csv(rawdata / "survey.csv", index=False)
    codebook = tmp_path / "codebook"
    codebook.mkdir()
    pd.DataFrame(
        {
            "variable_name": ["q1", "age"],
            "variable_label": ["job satisfaction outcome score", "respondent age"],
            "question_text": ["Overall, how satisfied are you with your job?", "Age in years"],
            "role": ["dependent", "control"],
        }
    ).to_csv(codebook / "survey_codebook.csv", index=False)

    result = load_rawdata_project(tmp_path)
    metadata = result.variable_metadata.set_index("variable_name")

    assert [path.name for path in discover_metadata_files(tmp_path)] == ["survey_codebook.csv"]
    assert result.metadata_files == [codebook / "survey_codebook.csv"]
    assert metadata.loc["q1", "variable_label"] == "job satisfaction outcome score"
    assert metadata.loc["q1", "question_text"] == "Overall, how satisfied are you with your job?"
    assert metadata.loc["q1", "role_hint"] == "dependent"
    assert "survey_codebook.csv" in metadata.loc["q1", "metadata_source_files"]


def test_load_rawdata_project_auto_merges_files_with_unique_shared_id(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "outcome_score": [2.0, 2.4, 3.1, 3.3],
        }
    ).to_csv(rawdata / "outcomes.csv", index=False)
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "age": [21, 35, 44, 51],
            "gender": [0, 1, 1, 0],
        }
    ).to_csv(rawdata / "demographics.csv", index=False)

    result = load_rawdata_project(tmp_path)

    assert result.merge_key == "person_id"
    assert result.merged_candidate_labels
    assert result.dataframe.shape == (4, 4)
    assert set(result.dataframe.columns) == {"person_id", "outcome_score", "age", "gender"}
    assert result.selected_candidate.column_count == 4


def test_load_rawdata_project_can_disable_auto_merge(tmp_path: Path) -> None:
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "outcome_score": [2.0, 2.4, 3.1, 3.3],
        }
    ).to_csv(rawdata / "outcomes.csv", index=False)
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "age": [21, 35, 44, 51],
        }
    ).to_csv(rawdata / "demographics.csv", index=False)

    result = load_rawdata_project(tmp_path, auto_merge=False)

    assert result.merge_key is None
    assert result.merged_candidate_labels == []
    assert result.dataframe.shape[1] == 2
