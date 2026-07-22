from pathlib import Path

import pandas as pd

from src.auto.runner import run_auto_rawdata_analysis
from src.auto.validation import validate_auto_run_outputs


def _write_rawdata(root: Path) -> None:
    rawdata_dir = root / "rawdata"
    rawdata_dir.mkdir()
    pd.DataFrame(
        {
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "gender": [0, 1, 1, 0, 1, 0, 1, 0],
        }
    ).to_csv(rawdata_dir / "survey.csv", index=False)


def test_validate_auto_run_outputs_passes_for_plan_only_run(tmp_path: Path) -> None:
    _write_rawdata(tmp_path)
    result = run_auto_rawdata_analysis(tmp_path, run_analysis=False)

    report = validate_auto_run_outputs(
        runtime=result.runtime,
        output_files=result.output_files,
    )

    assert report.passed is True
    assert report.warnings == []
    assert {item.item for item in report.items} >= {
        "artifact:auto_rawdata_load_result",
        "artifact:auto_variable_map",
        "artifact:auto_analysis_plan",
        "file:auto_run_report.md",
        "file:auto_final_report.md",
        "yaml:auto_analysis_plan",
        "yaml:auto_variable_map",
    }


def test_validate_auto_run_outputs_reports_missing_required_file(tmp_path: Path) -> None:
    _write_rawdata(tmp_path)
    result = run_auto_rawdata_analysis(tmp_path, run_analysis=False)
    output_files = [path for path in result.output_files if Path(path).name != "auto_run_report.md"]

    report = validate_auto_run_outputs(
        runtime=result.runtime,
        output_files=output_files,
    )

    assert report.passed is False
    assert any("file:auto_run_report.md" in warning for warning in report.warnings)


def test_validate_auto_run_outputs_can_require_model_outputs(tmp_path: Path) -> None:
    _write_rawdata(tmp_path)
    result = run_auto_rawdata_analysis(tmp_path, run_analysis=False)

    report = validate_auto_run_outputs(
        runtime=result.runtime,
        output_files=result.output_files,
        require_model_outputs=True,
    )

    assert report.passed is False
    assert any("coefficients.xlsx" in warning for warning in report.warnings)
    assert any("fit_statistics.xlsx" in warning for warning in report.warnings)
