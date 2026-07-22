from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.auto import cli


def test_auto_cli_plan_only_passes_arguments(monkeypatch, tmp_path, capsys) -> None:
    calls = {}

    def fake_run_auto_rawdata_analysis(working_directory, **kwargs):
        calls["working_directory"] = working_directory
        calls.update(kwargs)
        registration = SimpleNamespace(
            model_type="ols",
            dependent_variable="outcome_score",
            independent_variables=["age", "gender"],
        )
        return SimpleNamespace(
            success=True,
            failed_stage=None,
            pipeline_build_result=SimpleNamespace(registration=registration),
            output_files=[str(tmp_path / "result" / "auto_run_summary.xlsx")],
            warnings=[],
        )

    monkeypatch.setattr(cli, "run_auto_rawdata_analysis", fake_run_auto_rawdata_analysis)

    exit_code = cli.main(
        [
            "--working-directory",
            str(tmp_path),
            "--rawdata-dir",
            "data",
            "--source-file",
            "data/survey.csv",
            "--no-auto-merge",
            "--codebook-dir",
            "codebooks",
            "--questionnaire-dir",
            "questionnaires",
            "--project-name",
            "cli study",
            "--model-id",
            "model_a",
            "--enable-robustness",
            "--plan-only",
            "--multi-outcome",
            "--max-outcomes",
            "2",
            "--dependent-variable",
            "final_score",
            "--independent-variables",
            "baseline_score",
            "age",
            "--control-variables",
            "gender",
            "--cluster-variable",
            "site",
            "--weight-variable",
            "sample_weight",
            "--id-variable",
            "person_id",
            "--time-variable",
            "wave",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls["working_directory"] == tmp_path
    assert calls["rawdata_dir"] == "data"
    assert calls["source_file"] == "data/survey.csv"
    assert calls["auto_merge"] is False
    assert calls["codebook_dir"] == "codebooks"
    assert calls["questionnaire_dir"] == "questionnaires"
    assert calls["project_name"] == "cli study"
    assert calls["model_id"] == "model_a"
    assert calls["enable_robustness"] is True
    assert calls["run_analysis"] is False
    assert calls["enable_multi_outcome"] is True
    assert calls["max_outcomes"] == 2
    assert calls["dependent_variable"] == "final_score"
    assert calls["independent_variables"] == ["baseline_score", "age"]
    assert calls["control_variables"] == ["gender"]
    assert calls["cluster_variable"] == "site"
    assert calls["weight_variable"] == "sample_weight"
    assert calls["id_variable"] == "person_id"
    assert calls["time_variable"] == "wave"
    assert "Auto rawdata analysis completed." in captured.out
    assert "Model type: ols" in captured.out


def test_auto_cli_returns_nonzero_on_failure(monkeypatch, capsys) -> None:
    def fake_run_auto_rawdata_analysis(working_directory, **kwargs):
        return SimpleNamespace(
            success=False,
            failed_stage="01_auto_rawdata_loading",
            pipeline_build_result=None,
            output_files=[],
            warnings=["missing rawdata"],
        )

    monkeypatch.setattr(cli, "run_auto_rawdata_analysis", fake_run_auto_rawdata_analysis)

    exit_code = cli.main(["--plan-only"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Auto rawdata analysis failed." in captured.out
    assert "Failed stage: 01_auto_rawdata_loading" in captured.out
    assert "missing rawdata" in captured.out


def test_auto_cli_smoke_plan_only_creates_core_outputs(tmp_path: Path) -> None:
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()
    pd.DataFrame(
        {
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "gender": [0, 1, 1, 0, 1, 0, 1, 0],
        }
    ).to_csv(rawdata_dir / "survey.csv", index=False)

    exit_code = cli.main(
        [
            "--working-directory",
            str(tmp_path),
            "--project-name",
            "cli smoke",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "result" / "00_auto_run" / "auto_run_report.md").exists()
    assert (tmp_path / "result" / "00_auto_run" / "auto_final_report.md").exists()
    assert (tmp_path / "result" / "03_auto_plan" / "auto_analysis_plan.yaml").exists()
    assert (tmp_path / "result" / "03_auto_plan" / "auto_variable_map.yaml").exists()
