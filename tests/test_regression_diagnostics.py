"""OLS 및 Binary Logit 회귀진단 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.context import ResearchContext
from src.pipeline.regression_diagnostics_step import (
    RegressionDiagnosticsStep,
)
from src.pipeline.runtime import PipelineRuntime
from src.statistics.diagnostics.binary_logit import (
    build_binary_logit_diagnostics,
    calculate_binary_classification_metrics,
    calculate_binary_multicollinearity,
)
from src.statistics.diagnostics.ols import (
    build_ols_diagnostics,
    calculate_multicollinearity,
    calculate_residuals_and_influence,
    run_diagnostic_tests,
)
from src.statistics.diagnostics.ordered_logit import (
    build_ordered_logit_diagnostics,
    calculate_ordered_classification_metrics,
    calculate_ordered_multicollinearity,
    calculate_ordered_thresholds,
)
from src.statistics.regression.binary_logit import (
    fit_binary_logit,
)
from src.statistics.regression.ols import fit_ols
from src.statistics.regression.ordered_logit import fit_ordered_logit


def make_ols_result():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    error = rng.normal(
        scale=0.8,
        size=200,
    )
    y = 1.5 + 2.0 * x1 - 0.7 * x2 + error

    return fit_ols(
        pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        ),
        dependent_variable="y",
        independent_variables=[
            "x1",
            "x2",
        ],
        model_id="main_model",
        covariance_type="HC3",
    )


def make_binary_logit_result():
    rng = np.random.default_rng(21)
    x1 = rng.normal(size=300)
    x2 = rng.normal(size=300)
    linear_predictor = -0.3 + 1.1 * x1 - 0.8 * x2
    probability = 1 / (1 + np.exp(-linear_predictor))
    y = rng.binomial(
        1,
        probability,
    )

    return fit_binary_logit(
        pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        ),
        dependent_variable="y",
        independent_variables=[
            "x1",
            "x2",
        ],
        model_id="main_model",
        covariance_type="HC3",
    )


def make_ordered_logit_result():
    rng = np.random.default_rng(31)
    x1 = rng.normal(size=360)
    x2 = rng.normal(size=360)
    latent = 0.9 * x1 - 0.6 * x2 + rng.logistic(size=360)
    y = np.digitize(latent, bins=[-0.8, 0.0, 0.9]) + 1

    return fit_ordered_logit(
        pd.DataFrame({"y": y, "x1": x1, "x2": x2}),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        model_id="main_model",
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
    (
        residuals,
        influence,
        thresholds,
    ) = calculate_residuals_and_influence(result)

    assert len(residuals) == (result.sample_size)
    assert len(influence) == (result.sample_size)
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
    x2 = x1 + rng.normal(
        scale=0.001,
        size=150,
    )
    y = 2 + x1 + rng.normal(size=150)

    result = fit_ols(
        pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        ),
        dependent_variable="y",
        independent_variables=[
            "x1",
            "x2",
        ],
    )

    diagnostics = calculate_multicollinearity(result)

    assert any(
        item.status
        in {
            "WARNING",
            "FAIL",
        }
        for item in diagnostics
    )


def test_binary_logit_multicollinearity() -> None:
    diagnostics = calculate_binary_multicollinearity(make_binary_logit_result())

    assert len(diagnostics) == 2
    assert all(item.vif is not None for item in diagnostics)


def test_binary_logit_classification_metrics() -> None:
    result = make_binary_logit_result()
    metrics, predictions = calculate_binary_classification_metrics(result)

    assert 0 <= metrics.accuracy <= 1
    assert metrics.roc_auc is not None
    assert 0 <= metrics.roc_auc <= 1
    assert 0 <= metrics.brier_score <= 1
    assert len(predictions) == (result.sample_size)
    assert {
        "actual",
        "predicted_probability",
        "predicted_class",
        "classification_correct",
    }.issubset(predictions.columns)


def test_build_binary_logit_diagnostics() -> None:
    report = build_binary_logit_diagnostics(make_binary_logit_result())

    assert report.model_id == "main_model"
    assert report.sample_size == 300
    assert report.parameter_count == 3
    assert report.event_count > 0
    assert report.non_event_count > 0
    assert report.summary["roc_auc"] is not None


def test_invalid_classification_threshold_raises() -> None:
    result = make_binary_logit_result()

    try:
        calculate_binary_classification_metrics(
            result,
            threshold=1.0,
        )
    except ValueError as error:
        assert "0과 1 사이" in str(error)
    else:
        raise AssertionError("ValueError가 발생해야 합니다.")


def test_ordered_logit_multicollinearity() -> None:
    diagnostics = calculate_ordered_multicollinearity(make_ordered_logit_result())

    assert len(diagnostics) == 2
    assert all(item.vif is not None for item in diagnostics)


def test_ordered_logit_classification_metrics() -> None:
    result = make_ordered_logit_result()
    metrics, predictions, confusion_matrix = calculate_ordered_classification_metrics(result)

    assert 0 <= metrics.accuracy <= 1
    assert metrics.mean_absolute_category_error >= 0
    assert 0 <= metrics.ranked_probability_score <= 1
    assert 0 <= metrics.mean_maximum_predicted_probability <= 1
    assert len(predictions) == result.sample_size
    assert confusion_matrix.shape == (4, 4)


def test_ordered_logit_thresholds_are_increasing() -> None:
    thresholds = calculate_ordered_thresholds(make_ordered_logit_result())

    assert len(thresholds) == 3
    assert thresholds["strictly_increasing"].all()


def test_build_ordered_logit_diagnostics() -> None:
    report = build_ordered_logit_diagnostics(make_ordered_logit_result())

    assert report.model_id == "main_model"
    assert report.sample_size == 360
    assert report.category_count == 4
    assert report.summary["thresholds_strictly_increasing"] is True


def test_pipeline_step_outputs_ols_files(
    tmp_path: Path,
) -> None:
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


def test_pipeline_step_outputs_binary_logit_files(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact(
        "regression_result:main_model",
        make_binary_logit_result(),
    )

    step_result = RegressionDiagnosticsStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert all(Path(path).exists() for path in step_result.output_files)

    report = runtime.get_artifact("regression_diagnostics:main_model")
    assert report.sample_size == 300
    assert report.classification_metrics.roc_auc is not None


def test_pipeline_step_outputs_ordered_logit_files(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact(
        "regression_result:main_model",
        make_ordered_logit_result(),
    )

    step_result = RegressionDiagnosticsStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert step_result.success is True
    assert len(step_result.output_files) == 6
    assert all(Path(path).exists() for path in step_result.output_files)

    report = runtime.get_artifact("regression_diagnostics:main_model")
    assert report.sample_size == 360
    assert report.category_count == 4


def test_unsupported_model_is_skipped(
    tmp_path: Path,
) -> None:
    result = make_ols_result()
    result.model_type = "unsupported"

    runtime = PipelineRuntime()
    runtime.set_artifact(
        "regression_result:main_model",
        result,
    )

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
    assert step_result.metadata["skipped"] is True
