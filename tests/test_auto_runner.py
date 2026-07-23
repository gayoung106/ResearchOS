from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.auto.runner import run_auto_rawdata_analysis
from src.pipeline.orchestrator import OrchestratorResult, ResearchOrchestrator
from src.statistics.regression.base import ModelCoefficient, RegressionResult


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
    validation = result.runtime.get_artifact("auto_run_validation_report")
    assert validation.passed is True
    assert {Path(path).name for path in result.output_files} >= {
        "analysis_base.parquet",
        "variable_role_inference.xlsx",
        "analysis_plan_summary.xlsx",
        "auto_analysis_plan.yaml",
        "auto_variable_map.yaml",
        "auto_run_summary.xlsx",
        "auto_run_report.md",
        "auto_final_report.md",
    }
    report_path = next(Path(path) for path in result.output_files if Path(path).name == "auto_run_report.md")
    report_text = report_path.read_text(encoding="utf-8")
    assert "# \uc790\ub3d9 \ubd84\uc11d \uc2e4\ud589 \uc694\uc57d" in report_text
    assert "outcome_score" in report_text
    assert "ols" in report_text
    final_report_path = next(Path(path) for path in result.output_files if Path(path).name == "auto_final_report.md")
    final_report_text = final_report_path.read_text(encoding="utf-8")
    assert "Main model" in final_report_text
    assert "outcome_score" in final_report_text


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



def test_run_auto_rawdata_analysis_applies_variable_role_overrides(tmp_path: Path) -> None:
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()
    pd.DataFrame(
        {
            "baseline_score": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
            "final_score": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
            "site": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    ).to_csv(rawdata_dir / "survey.csv", index=False)

    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata overrides",
        run_analysis=False,
        dependent_variable="final_score",
        independent_variables=["baseline_score"],
        control_variables=["age"],
        cluster_variable="site",
    )

    plan = result.runtime.get_artifact("auto_analysis_plan")
    variable_map = result.runtime.get_artifact("auto_variable_map")

    assert result.success is True
    assert plan.variables.dependent == ["final_score"]
    assert plan.variables.independent == ["baseline_score"]
    assert plan.variables.controls == ["age"]
    assert plan.variables.clusters == ["site"]
    assert variable_map.variables["final_score"].review_status == "user_overridden"
    assert result.pipeline_build_result is not None
    assert result.pipeline_build_result.registration is not None
    assert result.pipeline_build_result.registration.dependent_variable == "final_score"
    assert result.pipeline_build_result.registration.independent_variables == ["baseline_score", "age"]
    assert "overridden_variable_map.xlsx" in {Path(path).name for path in result.output_files}


def test_run_auto_rawdata_analysis_reports_missing_override_variable(tmp_path: Path) -> None:
    _write_rawdata(tmp_path)

    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata bad override",
        run_analysis=False,
        dependent_variable="missing_y",
    )

    assert result.success is False
    assert result.failed_stage == "02_auto_variable_role_overrides"
    assert any("missing_y" in warning for warning in result.warnings)


def test_run_auto_rawdata_analysis_reports_setup_failure(tmp_path: Path) -> None:
    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata missing",
        run_analysis=False,
    )

    assert result.success is False
    assert result.failed_stage == "01_auto_rawdata_loading"
    assert result.pipeline_build_result is None
    assert {Path(path).name for path in result.output_files} >= {
        "auto_run_summary.xlsx",
        "auto_run_report.md",
        "auto_final_report.md",
    }


def test_run_auto_rawdata_analysis_runs_multi_outcome_pipelines_when_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()
    pd.DataFrame(
        {
            "satisfaction_outcome": [2.0, 2.4, 3.1, 3.3, 4.0, 4.2, 4.7, 5.1],
            "performance_outcome": [10.0, 11.2, 12.1, 13.0, 14.5, 15.1, 15.8, 16.2],
            "baseline_score": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
            "age": [21, 35, 44, 51, 39, 28, 46, 57],
        }
    ).to_csv(rawdata_dir / "survey.csv", index=False)
    calls: list[str] = []

    def fake_run(self: ResearchOrchestrator, **kwargs) -> OrchestratorResult:
        calls.append(self.context.project_name)
        self.context.add_generated_file(self.working_directory / "result" / f"{len(calls)}_fake_output.xlsx")
        return OrchestratorResult(success=True, completed_stages=["09_regression_analysis"])

    monkeypatch.setattr(ResearchOrchestrator, "run", fake_run)

    result = run_auto_rawdata_analysis(
        tmp_path,
        project_name="auto rawdata multi outcome",
        enable_multi_outcome=True,
        max_outcomes=2,
    )

    assert result.success is True
    assert result.multi_outcome_pipeline_build_result is not None
    assert result.multi_outcome_pipeline_build_result.success is True
    assert result.multi_outcome_pipeline_run_result is not None
    assert result.multi_outcome_pipeline_run_result.success is True
    assert len(result.multi_outcome_pipeline_run_result.completed_models) == 2
    assert len(calls) == 3
    assert result.runtime.get_artifact("auto_multi_outcome_pipeline_run_result").success is True
    assert "outcome_analysis_plans.xlsx" in {Path(path).name for path in result.output_files}
    assert any(Path(path).name.endswith("fake_output.xlsx") for path in result.output_files)
    final_report_path = next(Path(path) for path in result.output_files if Path(path).name == "auto_final_report.md")
    final_report_text = final_report_path.read_text(encoding="utf-8")
    assert "Multi-outcome models" in final_report_text
    assert "satisfaction_outcome" in final_report_text
    assert "performance_outcome" in final_report_text


def test_auto_final_report_summarizes_regression_artifacts(monkeypatch, tmp_path: Path) -> None:
    _write_rawdata(tmp_path)

    def fake_run(self: ResearchOrchestrator, **kwargs) -> OrchestratorResult:
        result = RegressionResult(
            model_id="main_model",
            model_type="ols",
            dependent_variable="outcome_score",
            independent_variables=["age", "gender"],
            sample_size=8,
            coefficients=[
                ModelCoefficient("const", 0.1, 0.1, 1.0, 0.40, -0.1, 0.3),
                ModelCoefficient("age", 0.25, 0.05, 5.0, 0.001, 0.15, 0.35),
                ModelCoefficient("gender", -0.2, 0.08, -2.5, 0.030, -0.4, -0.05),
            ],
            fit_statistics={"r_squared": 0.72, "aic": 12.5},
            converged=True,
            standard_error_type="HC3",
            warnings=["diagnostic warning"],
        )
        self.registry.names()
        step_runtime = next(step.runtime for step in self.registry.ordered_steps() if hasattr(step, "runtime"))
        step_runtime.set_artifact("regression_result:main_model", result)
        step_runtime.set_artifact(
            "effect_size_report:main_model",
            SimpleNamespace(
                effects=[
                    SimpleNamespace(
                        term="age",
                        effect_type="standardized_beta",
                        estimate=0.55,
                        p_value=0.001,
                        magnitude="large",
                    )
                ]
            ),
        )
        step_runtime.set_artifact(
            "regression_publication_report:main_model",
            SimpleNamespace(narrative="Age was positively associated with outcome_score."),
        )
        return OrchestratorResult(success=True, completed_stages=["09_regression_analysis"])

    monkeypatch.setattr(ResearchOrchestrator, "run", fake_run)

    result = run_auto_rawdata_analysis(tmp_path, project_name="auto final result summary")

    final_report_path = next(Path(path) for path in result.output_files if Path(path).name == "auto_final_report.md")
    final_report_text = final_report_path.read_text(encoding="utf-8")

    assert "Main model results" in final_report_text
    assert "| age | 0.250 | 0.001 |" in final_report_text
    assert "| r_squared | 0.720 |" in final_report_text
    assert "standardized_beta" in final_report_text
    assert "Age was positively associated with outcome_score." in final_report_text
    assert "diagnostic warning" in final_report_text
