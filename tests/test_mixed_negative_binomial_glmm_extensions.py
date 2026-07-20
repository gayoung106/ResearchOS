from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit.research import build_research_audit_report
from src.common.config_models import AnalysisPlan, VariableDefinition, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import register_regression_pipeline
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import build_regression_publication_report
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_negative_binomial import (
    fit_mixed_negative_binomial_random_slope,
    fit_mixed_negative_binomial_three_level,
)
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _negative_binomial_counts(rng: np.random.Generator, mu: np.ndarray, alpha: float) -> np.ndarray:
    shape = 1.0 / alpha
    rate = shape / mu
    latent_mean = rng.gamma(shape, 1.0 / rate)
    return rng.poisson(latent_mean)


def _random_slope_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260726)
    groups = np.repeat(np.arange(6), 14)
    x = rng.normal(size=len(groups))
    intercepts = rng.normal(0, 0.25, 6)
    slopes = rng.normal(0, 0.18, 6)
    mu = np.exp(0.25 + intercepts[groups] + (0.35 + slopes[groups]) * x)
    y = _negative_binomial_counts(rng, mu, alpha=0.55)
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def _three_level_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260727)
    region = np.repeat(np.arange(3), 36)
    cluster = np.repeat(np.arange(9), 12)
    x = rng.normal(size=len(cluster))
    region_effect = rng.normal(0, 0.18, 3)
    cluster_effect = rng.normal(0, 0.24, 9)
    mu = np.exp(0.2 + 0.3 * x + region_effect[region] + cluster_effect[cluster])
    y = _negative_binomial_counts(rng, mu, alpha=0.6)
    return pd.DataFrame({"y": y, "x": x, "cluster": cluster, "region": region})


def _count_variable_map(*, include_group: bool = True) -> VariableMap:
    variables = {
        "y": VariableDefinition(original_name="y", role="dependent", measurement_level="count"),
        "x": VariableDefinition(original_name="x", role="independent", measurement_level="continuous"),
    }
    if include_group:
        variables["group"] = VariableDefinition(
            original_name="group", role="cluster", measurement_level="nominal"
        )
    return VariableMap(variables=variables)


def test_fit_mixed_negative_binomial_random_slope_integrates_outputs(tmp_path: Path) -> None:
    result = fit_mixed_negative_binomial_random_slope(
        _random_slope_data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        random_slope_variable="x",
        quadrature_points=5,
        max_iterations=90,
    )
    effect_report = build_regression_effect_size_report(result)
    publication_report = build_regression_publication_report(result, effect_report)
    visualization_report = build_regression_visualizations(result, output_directory=tmp_path)

    runtime = PipelineRuntime()
    runtime.set_artifact("regression_result:main_model", result)
    runtime.set_artifact("effect_size_report:main_model", effect_report)
    runtime.set_artifact("regression_publication_report:main_model", publication_report)
    runtime.set_artifact("regression_visualization:main_model", visualization_report)
    audit_report = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "mixed_negative_binomial_random_slope"
    assert result.fit_statistics["group_count"] == 6
    assert result.fit_statistics["dispersion_alpha"] > 0
    assert result.fit_statistics["random_slope_variance"] >= 0
    assert len(result.metadata["random_intercepts"]) == 6
    assert len(result.metadata["random_slopes"]) == 6
    assert any(item.effect_type == "incidence_rate_ratio" for item in effect_report.effects)
    assert publication_report.model_type == result.model_type
    assert {Path(path).name for path in visualization_report.output_files} == {
        "coefficient_forest.png",
        "random_intercepts.png",
    }
    assert audit_report.metadata["model_type"] == result.model_type


def test_fit_and_select_mixed_negative_binomial_three_level() -> None:
    data = _three_level_data()
    result = fit_mixed_negative_binomial_three_level(
        data,
        dependent_variable="y",
        independent_variables=["x"],
        level2_group="cluster",
        level3_group="region",
        quadrature_points=5,
        max_iterations=70,
    )
    selected = fit_regression_by_level(
        data,
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="mixed_negative_binomial_three_level",
        mixed_effects_options={
            "level2_group": "cluster",
            "level3_group": "region",
            "quadrature_points": 5,
            "max_iterations": 70,
        },
    )

    assert result.model_type == "mixed_negative_binomial_three_level"
    assert selected.model_type == "mixed_negative_binomial_three_level"
    assert result.fit_statistics["level2_group_count"] == 9
    assert result.fit_statistics["level3_group_count"] == 3
    assert result.fit_statistics["level2_vpc"] + result.fit_statistics["level3_vpc"] == pytest.approx(1.0)
    assert len(result.metadata["level2_random_effects"]) == 9
    assert len(result.metadata["level3_random_effects"]) == 3


def test_builder_registers_mixed_negative_binomial_random_slope_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["group"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {
                    "enabled": True,
                    "options": {
                        "count_distribution": "negative_binomial",
                        "random_slope_variable": "x",
                        "quadrature_points": 5,
                    },
                },
            },
        }
    )
    runtime = PipelineRuntime()
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="nb glmm"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
        analysis_plan=plan,
        variable_map=_count_variable_map(),
    )

    assert registration.registered is True
    assert registration.model_type == "mixed_negative_binomial_random_slope"
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
