"""Tweedie regression integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.research import build_research_audit_report
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.regression_diagnostics_step import RegressionDiagnosticsStep
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import build_regression_publication_report
from src.statistics.diagnostics.tweedie import (
    build_tweedie_diagnostics,
    tweedie_diagnostic_summary_to_dataframe,
    tweedie_observations_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.selector import fit_regression_by_level
from src.statistics.regression.tweedie import fit_tweedie_regression
from src.visualization.regression import build_regression_visualizations
from tests.support.assertions import assert_registry_matches
from tests.support.builders import build_regression_pipeline
from tests.support.expected_pipeline import regression_pipeline


def _tweedie_dataframe(*, seed: int = 1207, size: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    z = rng.normal(size=size)
    mean = np.exp(0.25 + 0.55 * x - 0.2 * z)
    zero = rng.binomial(1, 0.25, size=size).astype(bool)
    positive = rng.gamma(shape=2.0, scale=mean / 2.0)
    y = np.where(zero, 0.0, positive)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_tweedie_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _tweedie_dataframe()
    result = fit_tweedie_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
        variance_power=1.5,
    )
    diagnostics = build_tweedie_diagnostics(result)
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    runtime.set_artifact("regression_diagnostics:main_model", diagnostics)
    runtime.set_artifact("effect_size_report:main_model", effects)
    runtime.set_artifact("regression_publication_report:main_model", report)
    runtime.set_artifact("regression_visualization:main_model", visual)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "tweedie_regression"
    assert result.fit_statistics["zero_count"] > 0
    assert result.metadata["variance_power"] == 1.5
    assert diagnostics.summary["variance_power"] == 1.5
    assert tweedie_observations_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "root_mean_squared_error" in set(tweedie_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert any(effect.effect_type == "mean_ratio" for effect in effects.effects)
    assert "Tweedie" in report.narrative
    assert any(path.endswith("tweedie_observed_vs_predicted.png") for path in visual.output_files)
    assert audit.metadata["model_type"] == "tweedie_regression"
    assert audit.metadata["variance_power"] == 1.5


def test_selector_routes_explicit_tweedie_regression() -> None:
    result = fit_regression_by_level(
        _tweedie_dataframe(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="tweedie_regression",
        model_id="main_model",
        mixed_effects_options={"variance_power": 1.4},
    )

    assert result.model_type == "tweedie_regression"
    assert result.model_id == "main_model"
    assert result.metadata["variance_power"] == 1.4


def test_tweedie_diagnostics_pipeline_step(tmp_path: Path) -> None:
    data = _tweedie_dataframe()
    result = fit_tweedie_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="tweedie diagnostics"),
        tmp_path,
    )

    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "tweedie_regression"


def test_builder_registers_explicit_tweedie_pipeline(tmp_path: Path) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x", "z"]},
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"model_type": "tweedie_regression", "variance_power": 1.5},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {"role": "dependent", "measurement_level": "continuous"},
                "x": {"role": "independent", "measurement_level": "continuous"},
                "z": {"role": "independent", "measurement_level": "continuous"},
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        project_name="tweedie builder",
    )

    assert registration.registered is True
    assert registration.model_type == "tweedie_regression"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
    assert registration.robustness_registered is False
    assert_registry_matches(
        orchestrator,
        regression_pipeline(diagnostics=True, robustness=False, advanced_robustness=False),
    )
