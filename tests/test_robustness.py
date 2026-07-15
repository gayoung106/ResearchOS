"""OLS Robustness Engine 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.context import ResearchContext
from src.pipeline.robustness_step import OLSRobustnessStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.regression.ols import fit_ols
from src.statistics.robustness.ols import (
    build_ols_robustness_report,
    coefficient_comparison_to_dataframe,
    stability_summary_to_dataframe,
)


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=300)
    x2 = rng.normal(size=300)
    scale = 0.5 + 0.8 * np.abs(x1)
    error = rng.normal(scale=scale)
    y = 1.0 + 1.5 * x1 - 0.6 * x2 + error

    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )


def test_robustness_report_contains_all_covariance_types() -> None:
    report = build_ols_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
    )

    assert report.covariance_types == [
        "nonrobust",
        "HC0",
        "HC1",
        "HC2",
        "HC3",
    ]
    assert len(report.model_statistics) == 5


def test_estimates_are_identical_across_covariance_types() -> None:
    report = build_ols_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
    )
    frame = coefficient_comparison_to_dataframe(report)

    for _, group in frame.groupby("term"):
        assert group["estimate"].nunique() == 1


def test_standard_errors_can_differ() -> None:
    report = build_ols_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
    )
    frame = coefficient_comparison_to_dataframe(report)
    x1 = frame.loc[frame["term"] == "x1"]

    assert x1["standard_error"].nunique() > 1


def test_stability_summary_is_created() -> None:
    report = build_ols_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
    )
    summary = stability_summary_to_dataframe(report)

    assert set(summary["term"]) >= {"const", "x1", "x2"}
    assert set(summary["status"]).issubset({"STABLE", "PARTIALLY_STABLE", "UNSTABLE"})


def test_pipeline_step_outputs_files(
    tmp_path: Path,
) -> None:
    dataframe = make_dataframe()
    regression_result = fit_ols(
        dataframe,
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        model_id="main_model",
    )

    runtime = PipelineRuntime(dataframe=dataframe)
    runtime.set_artifact(
        "regression_result:main_model",
        regression_result,
    )

    step = OLSRobustnessStep(
        runtime,
        model_id="main_model",
    )
    result = step.run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert result.success is True
    assert len(result.output_files) == 4
    assert all(Path(path).exists() for path in result.output_files)
    assert runtime.get_artifact("robustness_report:main_model").model_id == "main_model"


def test_non_ols_model_does_not_run() -> None:
    dataframe = make_dataframe()
    regression_result = fit_ols(
        dataframe,
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        model_id="main_model",
    )
    regression_result.model_type = "binary_logit"

    runtime = PipelineRuntime(dataframe=dataframe)
    runtime.set_artifact(
        "regression_result:main_model",
        regression_result,
    )

    step = OLSRobustnessStep(
        runtime,
        model_id="main_model",
    )

    assert step.should_run(ResearchContext(project_name="테스트")) is False
