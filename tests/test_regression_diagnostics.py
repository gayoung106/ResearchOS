"""OLS Regression Diagnostics 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.context import ResearchContext
from src.pipeline.regression_diagnostics_step import RegressionDiagnosticsStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.diagnostics.ols import (
    build_ols_diagnostics,
    calculate_multicollinearity,
    calculate_residuals_and_influence,
    run_diagnostic_tests,
)
from src.statistics.regression.ols import fit_ols


def make_ols_result():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    error = rng.normal(scale=0.8, size=200)
    y = 1.5 + 2.0 * x1 - 0.7 * x2 + error

    return fit_ols(
        pd.DataFrame({"y": y, "x1": x1, "x2": x2}),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        model_id="main_model",
        covariance_type="HC3",
    )


def test_multicollinearity_contains_vif_and_tolerance() -> None:
    diagnostics = calculate_multicollinearity(make_ols_result())

    assert len(diagnostics) == 2
    assert all(item.vif is not None for item in diagnostics)
    assert all(item.tolerance is not None for item in diagnostics)
    assert all(item.status == "PASS" for item in diagnostics)


def test_diagnostic_tests_include_required_tests() -> None:
    tests = run_diagnostic_tests(make_ols_result())
    names = {item.test_name for item in tests}

    assert "Breusch-Pagan LM" in names
    assert "White LM" in names
    assert "Ramsey RESET" in names
    assert "Jarque-Bera" in names


def test_influence_tables_are_created() -> None:
    result = make_ols_result()
    residuals, influence, thresholds = calculate_residuals_and_influence(result)

    assert len(residuals) == result.sample_size
    assert len(influence) == result.sample_size
    assert thresholds.cooks_distance > 0
    assert "any_influence_flag" in influence.columns


def test_build_ols_diagnostics() -> None:
    report = build_ols_diagnostics(make_ols_result())

    assert report.model_id == "main_model"
    assert report.sample_size == 200
    assert report.parameter_count == 3
    assert report.summary["model_id"] == "main_model"


def test_high_multicollinearity_is_flagged() -> None:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=150)
    x2 = x1 + rng.normal(scale=0.001, size=150)
    y = 2 + x1 + rng.normal(size=150)

    result = fit_ols(
        pd.DataFrame({"y": y, "x1": x1, "x2": x2}),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
    )

    diagnostics = calculate_multicollinearity(result)

    assert any(item.status in {"WARNING", "FAIL"} for item in diagnostics)


def test_pipeline_step_outputs_files(tmp_path: Path) -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact(
        "regression_result:main_model",
        make_ols_result(),
    )

    step_result = RegressionDiagnosticsStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert all(Path(path).exists() for path in step_result.output_files)
    assert runtime.get_artifact("regression_diagnostics:main_model").sample_size == 200


def test_non_ols_model_is_skipped(tmp_path: Path) -> None:
    result = make_ols_result()
    result.model_type = "binary_logit"

    runtime = PipelineRuntime()
    runtime.set_artifact("regression_result:main_model", result)

    step_result = RegressionDiagnosticsStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert step_result.success is True
    assert step_result.output_files == []
    assert step_result.warnings
