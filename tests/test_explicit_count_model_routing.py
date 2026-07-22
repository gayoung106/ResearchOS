"""Explicit count model routing integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.research import build_research_audit_report
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import build_regression_publication_report
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations
from tests.support.assertions import assert_registry_matches
from tests.support.builders import build_regression_pipeline
from tests.support.expected_pipeline import count_pipeline


def _poisson_dataframe(*, seed: int = 440, size: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    z = rng.normal(size=size)
    mean = np.exp(0.25 + 0.45 * x - 0.2 * z)
    y = rng.poisson(mean)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _overdispersed_dataframe(*, seed: int = 441, size: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    z = rng.normal(size=size)
    mean = np.exp(0.3 + 0.5 * x - 0.15 * z)
    alpha = 1.1
    multiplier = rng.gamma(shape=1 / alpha, scale=alpha, size=size)
    y = rng.poisson(mean * multiplier)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_explicit_poisson_integrates_outputs(tmp_path: Path) -> None:
    data = _poisson_dataframe()
    result = fit_regression_by_level(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="count",
        model_type="poisson",
        model_id="main_model",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    runtime.set_artifact("effect_size_report:main_model", effects)
    runtime.set_artifact("regression_publication_report:main_model", report)
    runtime.set_artifact("regression_visualization:main_model", visual)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "poisson"
    assert result.model_id == "main_model"
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in effects.effects)
    assert report.model_type == "poisson"
    assert any(path.endswith("count_observed_vs_predicted.png") for path in visual.output_files)
    assert audit.metadata["model_type"] == "poisson"


def test_selector_routes_explicit_negative_binomial() -> None:
    result = fit_regression_by_level(
        _overdispersed_dataframe(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="count",
        model_type="negative_binomial",
        model_id="main_model",
    )

    assert result.model_type == "negative_binomial"
    assert result.model_id == "main_model"
    assert result.fit_statistics["alpha"] > 0


def test_builder_routes_explicit_negative_binomial(tmp_path: Path) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x", "z"]},
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"model_type": "negative_binomial"},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {"role": "dependent", "measurement_level": "count"},
                "x": {"role": "independent", "measurement_level": "continuous"},
                "z": {"role": "independent", "measurement_level": "continuous"},
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        project_name="explicit count routing",
    )

    assert registration.registered is True
    assert registration.model_type == "negative_binomial"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
    assert registration.robustness_registered is False
    assert_registry_matches(orchestrator, count_pipeline())
