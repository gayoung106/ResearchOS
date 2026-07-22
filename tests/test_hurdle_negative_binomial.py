"""Hurdle negative binomial regression integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.research import build_research_audit_report
from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import build_regression_publication_report
from src.statistics.diagnostics.count import build_count_diagnostics
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.hurdle_negative_binomial import fit_hurdle_negative_binomial
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations
from tests.support.assertions import assert_registry_matches
from tests.support.builders import build_regression_pipeline
from tests.support.expected_pipeline import count_pipeline


def _hurdle_nb_dataframe(*, seed: int = 884, size: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    z = rng.normal(size=size)
    hurdle_probability = 1 / (1 + np.exp(-(-0.35 + 0.75 * x - 0.25 * z)))
    positive = rng.binomial(1, hurdle_probability, size=size)
    mean = np.exp(0.15 + 0.55 * x + 0.2 * z)
    alpha = 1.4
    multiplier = rng.gamma(shape=1 / alpha, scale=alpha, size=size)
    y = rng.poisson(mean * multiplier) + 1
    y = y * positive
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_hurdle_negative_binomial_integrates_outputs(tmp_path: Path) -> None:
    data = _hurdle_nb_dataframe()
    result = fit_hurdle_negative_binomial(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_count_diagnostics(result)
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

    assert result.model_type == "hurdle_negative_binomial"
    assert result.fit_statistics["zero_count"] > 0
    assert result.fit_statistics["positive_count"] > 0
    assert result.fit_statistics["alpha"] > 0
    assert diagnostics.summary["alpha"] == result.fit_statistics["alpha"]
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in effects.effects)
    assert "Hurdle negative binomial" in report.narrative
    assert any(path.endswith("count_observed_vs_predicted.png") for path in visual.output_files)
    assert audit.metadata["model_type"] == "hurdle_negative_binomial"
    assert audit.metadata["hurdle_model"] == "logit"


def test_selector_routes_explicit_hurdle_negative_binomial() -> None:
    result = fit_regression_by_level(
        _hurdle_nb_dataframe(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="count",
        model_type="hurdle_negative_binomial",
        model_id="main_model",
    )

    assert result.model_type == "hurdle_negative_binomial"
    assert result.model_id == "main_model"
    assert result.metadata["count_model"] == "zero_truncated_negative_binomial_mean"


def test_builder_registers_explicit_hurdle_negative_binomial(tmp_path: Path) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x", "z"]},
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"model_type": "hurdle_negative_binomial"},
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
        project_name="hurdle negative binomial builder",
    )

    assert registration.registered is True
    assert registration.model_type == "hurdle_negative_binomial"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
    assert registration.robustness_registered is False
    assert_registry_matches(orchestrator, count_pipeline())
