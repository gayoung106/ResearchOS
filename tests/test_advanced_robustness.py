"""고급 OLS 강건성 분석 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd
from src.statistics.robustness.resampling import (
    bootstrap_ols,
    fit_cluster_robust_ols,
    jackknife_ols,
)

from src.pipeline.advanced_robustness_step import (
    AdvancedOLSRobustnessStep,
)
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.statistics.regression.ols import fit_ols


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(44)
    cluster = np.repeat(np.arange(20), 15)
    x1 = rng.normal(size=len(cluster))
    x2 = rng.normal(size=len(cluster))
    cluster_effect = rng.normal(
        scale=0.7,
        size=20,
    )[cluster]
    y = 1.2 + 1.5 * x1 - 0.4 * x2 + cluster_effect + rng.normal(size=len(cluster))

    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "cluster": cluster,
        }
    )


def test_cluster_robust_ols() -> None:
    report = fit_cluster_robust_ols(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        cluster_variable="cluster",
    )

    assert report.method == "cluster_robust"
    assert report.metadata["cluster_count"] == 20
    assert len(report.coefficients) == 3


def test_bootstrap_ols() -> None:
    report = bootstrap_ols(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        replications=200,
        random_seed=7,
    )

    assert report.method == "bootstrap"
    assert report.metadata["successful_replications"] > 0
    assert len(report.coefficients) == 3
    assert all(item.standard_error > 0 for item in report.coefficients)


def test_bootstrap_is_reproducible() -> None:
    first = bootstrap_ols(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        replications=150,
        random_seed=99,
    )
    second = bootstrap_ols(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        replications=150,
        random_seed=99,
    )

    assert first.coefficients[1].standard_error == second.coefficients[1].standard_error


def test_jackknife_ols() -> None:
    report = jackknife_ols(
        make_dataframe().head(80),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
    )

    assert report.method == "jackknife"
    assert report.metadata["replications"] == 80
    assert len(report.coefficients) == 3


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

    step = AdvancedOLSRobustnessStep(
        runtime,
        model_id="main_model",
        cluster_variable="cluster",
        bootstrap_replications=120,
        run_jackknife=True,
    )

    result = step.run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert result.success is True
    assert len(result.output_files) == 6
    assert all(Path(path).exists() for path in result.output_files)
    assert set(result.metadata["completed_methods"]) == {
        "bootstrap",
        "jackknife",
        "cluster_robust",
    }


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

    step = AdvancedOLSRobustnessStep(
        runtime,
        model_id="main_model",
    )

    assert step.should_run(ResearchContext(project_name="테스트")) is False
