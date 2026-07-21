"""Explicit zero-inflated count regression pipeline integration tests."""

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
from src.statistics.diagnostics.count import build_count_diagnostics
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations
from tests.support.builders import build_regression_pipeline


def _zip_dataframe(*, seed: int = 20260721, size: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    z = rng.normal(size=size)
    mean = np.exp(0.25 + 0.45 * x - 0.20 * z)
    y = rng.poisson(mean)
    y[rng.random(size) < 0.42] = 0
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _zinb_dataframe(*, seed: int = 20260722, size: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    z = rng.normal(size=size)
    mean = np.exp(0.30 + 0.45 * x - 0.20 * z)
    alpha = 1.1
    multiplier = rng.gamma(shape=1 / alpha, scale=alpha, size=size)
    y = rng.poisson(mean * multiplier)
    y[rng.random(size) < 0.35] = 0
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _count_variable_map() -> VariableMap:
    return VariableMap.model_validate(
        {
            "variables": {
                "y": {"role": "dependent", "measurement_level": "count"},
                "x": {"role": "independent", "measurement_level": "continuous"},
                "z": {"role": "control", "measurement_level": "continuous"},
            }
        }
    )


def test_explicit_zero_inflated_poisson_integrates_outputs(tmp_path: Path) -> None:
    data = _zip_dataframe()
    result = fit_regression_by_level(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="count",
        model_type="zero_inflated_poisson",
        model_id="main_model",
        mixed_effects_options={"max_iterations": 300},
    )
    diagnostics = build_count_diagnostics(result)
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    runtime.set_artifact("effect_size_report:main_model", effects)
    runtime.set_artifact("regression_publication_report:main_model", report)
    runtime.set_artifact("regression_visualization:main_model", visual)
    audit = build_research_audit_report(runtime, model_id="main_model")
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        context=ResearchContext(project_name="zero inflated count"),
        working_directory=tmp_path,
    )

    assert result.model_type == "zero_inflated_poisson"
    assert diagnostics.model_type == "zero_inflated_poisson"
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in effects.effects)
    assert all(not effect.term.startswith("inflate_") for effect in effects.effects)
    assert report.model_type == "zero_inflated_poisson"
    assert report.narrative
    assert len(visual.output_files) >= 2
    assert visual.metadata["figure_count"] == len(visual.output_files)
    assert audit.metadata["model_type"] == "zero_inflated_poisson"
    assert audit.metadata["inflation_model"] == "logit_intercept_only"
    assert step_result.success is True
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "zero_inflated_poisson"


def test_builder_and_selector_route_explicit_zero_inflated_negative_binomial(
    tmp_path: Path,
) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x"], "controls": ["z"]},
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"model_type": "zero_inflated_negative_binomial"},
                },
                "robustness": {"enabled": True},
            },
        }
    )
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=plan,
        variable_map=_count_variable_map(),
    )
    selected = fit_regression_by_level(
        _zinb_dataframe(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="count",
        model_type="zero_inflated_negative_binomial",
        model_id="main_model",
        mixed_effects_options={"max_iterations": 500},
    )

    assert registration.registered is True
    assert registration.model_type == "zero_inflated_negative_binomial"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
    assert registration.robustness_registered is False
    assert any("OLS" in warning for warning in registration.warnings)
    assert "10_regression_diagnostics" in orchestrator.registry.names()
    assert selected.model_type == "zero_inflated_negative_binomial"
    assert selected.fit_statistics["alpha"] > 0
