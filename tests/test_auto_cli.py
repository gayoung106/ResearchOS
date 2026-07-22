from types import SimpleNamespace

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
            "--project-name",
            "cli study",
            "--model-id",
            "model_a",
            "--enable-robustness",
            "--plan-only",
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
    assert calls["project_name"] == "cli study"
    assert calls["model_id"] == "model_a"
    assert calls["enable_robustness"] is True
    assert calls["run_analysis"] is False
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
