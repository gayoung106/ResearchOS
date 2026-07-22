from pathlib import Path

import pandas as pd

from src.auto.runner import run_auto_rawdata_analysis
from src.pipeline.orchestrator import OrchestratorResult, ResearchOrchestrator


def _write_rawdata(root: Path) -> Path:
    rawdata_dir = root / "rawdata"
    rawdata_dir.mkdir()
    path = rawdata_dir / "survey.csv"
    pd.DataFrame(
        {
            "outcome_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "gender": [0, 1, 1, 0, 1, 0, 1, 0],
        }
    ).to_csv(path, index=False)
    return path


def test_run_auto_rawdata_analysis_prepares_and_registers_pipeline_without_execution(tmp_path: Path) -> None:
    _write_rawdata(tmp_path)

    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata setup",
        run_analysis=False,
    )

    assert result.success is True
    assert result.orchestrator_result is None
    assert result.pipeline_build_result is not None
    assert result.pipeline_build_result.success is True
    assert result.pipeline_build_result.registration is not None
    assert result.pipeline_build_result.registration.model_type == "ols"
    assert result.context.dependent_variables == ["outcome_score"]
    assert result.context.independent_variables == ["age", "gender"]
    assert result.runtime.get_artifact("auto_rawdata_load_result").selected_candidate.row_count == 8
    assert result.runtime.get_artifact("auto_analysis_plan").analyses.regression.enabled is True
    assert {Path(path).name for path in result.output_files} >= {
        "analysis_base.parquet",
        "variable_role_inference.xlsx",
        "analysis_plan_summary.xlsx",
        "auto_analysis_plan.yaml",
        "auto_variable_map.yaml",
        "auto_run_summary.xlsx",
    }


def test_run_auto_rawdata_analysis_executes_registered_pipeline_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_rawdata(tmp_path)
    calls: dict[str, list[str]] = {}

    def fake_run(self: ResearchOrchestrator, **kwargs) -> OrchestratorResult:
        calls["registered"] = self.registry.names()
        self.context.add_generated_file(self.working_directory / "result" / "fake_regression_output.xlsx")
        return OrchestratorResult(success=True, completed_stages=["09_regression_analysis"])

    monkeypatch.setattr(ResearchOrchestrator, "run", fake_run)

    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata execution",
    )

    assert result.success is True
    assert result.orchestrator_result is not None
    assert result.orchestrator_result.completed_stages == ["09_regression_analysis"]
    assert calls["registered"] == [
        "09_regression_analysis",
        "10_regression_diagnostics",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    ]
    assert "fake_regression_output.xlsx" in {Path(path).name for path in result.output_files}


def test_run_auto_rawdata_analysis_reports_setup_failure(tmp_path: Path) -> None:
    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata missing",
        run_analysis=False,
    )

    assert result.success is False
    assert result.failed_stage == "01_auto_rawdata_loading"
    assert result.pipeline_build_result is None
    assert Path(result.output_files[-1]).name == "auto_run_summary.xlsx"
