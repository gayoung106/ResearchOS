from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.research import build_research_audit_report
from src.common.config_models import AnalysisPlan, VariableDefinition, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.regression_builder import register_regression_pipeline
from src.pipeline.regression_diagnostics_step import RegressionDiagnosticsStep
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import build_regression_publication_report
from src.statistics.diagnostics.gee import (
    build_gee_diagnostics,
    gee_cluster_diagnostics_to_dataframe,
    gee_diagnostic_summary_to_dataframe,
    gee_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.gee import fit_gee
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _gaussian_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260730)
    clusters = np.repeat(np.arange(8), 10)
    x = rng.normal(size=len(clusters))
    cluster_effect = rng.normal(0.0, 0.4, size=8)
    y = 1.0 + 0.8 * x + cluster_effect[clusters] + rng.normal(0.0, 0.3, len(clusters))
    return pd.DataFrame({"y": y, "x": x, "cluster": clusters})


def _poisson_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260731)
    clusters = np.repeat(np.arange(6), 12)
    x = rng.normal(size=len(clusters))
    cluster_effect = rng.normal(0.0, 0.25, size=6)
    y = rng.poisson(np.exp(0.2 + 0.45 * x + cluster_effect[clusters]))
    return pd.DataFrame({"y": y, "x": x, "cluster": clusters})


def _negative_binomial_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260923)
    clusters = np.repeat(np.arange(7), 9)
    x = rng.normal(size=len(clusters))
    cluster_effect = rng.normal(0.0, 0.35, size=7)
    mean = np.exp(0.25 + 0.4 * x + cluster_effect[clusters])
    alpha = 0.8
    size = 1.0 / alpha
    probability = size / (size + mean)
    y = rng.negative_binomial(size, probability)
    return pd.DataFrame({"y": y, "x": x, "cluster": clusters})


def test_fit_gee_gaussian_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    result = fit_gee(
        _gaussian_data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="cluster",
        model_type="gee_gaussian",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=_gaussian_data())
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "gee_gaussian"
    assert result.fit_statistics["cluster_count"] == 8
    assert result.metadata["group_variable"] == "cluster"
    assert "qic_warning_count" in result.metadata
    assert any(effect.effect_type == "standardized_beta" for effect in effects.effects)
    assert "GEE accounted for 8 clusters defined by cluster." in report.narrative
    assert {Path(path).name for path in visual.output_files} == {"coefficient_forest.png"}
    assert audit.metadata["cluster_count"] == 8


def test_selector_routes_gee_poisson_and_effects_are_irr() -> None:
    result = fit_regression_by_level(
        _poisson_data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="gee_poisson",
        group_variable="cluster",
        mixed_effects_options={"covariance_structure": "exchangeable"},
    )
    effects = build_regression_effect_size_report(result)

    assert result.model_type == "gee_poisson"
    assert result.fit_statistics["cluster_count"] == 6
    assert all(coefficient.exponentiated_estimate is not None for coefficient in result.coefficients)
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in effects.effects)


def test_selector_routes_gee_negative_binomial_and_effects_are_irr() -> None:
    result = fit_regression_by_level(
        _negative_binomial_data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="gee_negative_binomial",
        group_variable="cluster",
        mixed_effects_options={"covariance_structure": "exchangeable"},
    )
    effects = build_regression_effect_size_report(result)

    assert result.model_type == "gee_negative_binomial"
    assert result.fit_statistics["cluster_count"] == 7
    assert result.fit_statistics["negative_binomial_alpha"] == 1.0
    assert all(coefficient.exponentiated_estimate is not None for coefficient in result.coefficients)
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in effects.effects)


def test_builder_registers_explicit_gee_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["cluster"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {
                        "estimator": "gee",
                        "covariance_structure": "exchangeable",
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="continuous"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "cluster": VariableDefinition(role="cluster", measurement_level="nominal"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="gee builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "gee_gaussian"
    assert registration.group_variable == "cluster"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True



def test_gee_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    result = fit_gee(
        _gaussian_data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="cluster",
        model_type="gee_gaussian",
    )
    report = build_gee_diagnostics(result)
    runtime = PipelineRuntime(dataframe=_gaussian_data())
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="gee diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert report.cluster_count == 8
    assert gee_cluster_diagnostics_to_dataframe(report).shape[0] == 8
    assert gee_residuals_to_dataframe(report).shape[0] == result.sample_size
    assert "max_abs_cluster_mean_pearson_residual" in set(
        gee_diagnostic_summary_to_dataframe(report)["item"]
    )
    assert step_result.success is True
    assert len(step_result.output_files) == 3
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "gee_gaussian"
    assert any("GEE diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_gee_negative_binomial_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["cluster"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {
                        "estimator": "gee_negative_binomial",
                        "covariance_structure": "exchangeable",
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="count"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "cluster": VariableDefinition(role="cluster", measurement_level="nominal"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="gee negative binomial builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "gee_negative_binomial"
    assert registration.group_variable == "cluster"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_gee_negative_binomial_diagnostics_reporting_and_audit(tmp_path: Path) -> None:
    result = fit_gee(
        _negative_binomial_data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="cluster",
        model_type="gee_negative_binomial",
    )
    diagnostics = build_gee_diagnostics(result)
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=_negative_binomial_data())
    runtime.set_artifact("regression_result:main_model", result)
    runtime.set_artifact("effect_size_report:main_model", effects)
    runtime.set_artifact("regression_publication_report:main_model", report)
    runtime.set_artifact("regression_visualization:main_model", visual)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.model_type == "gee_negative_binomial"
    assert diagnostics.cluster_count == 7
    assert "GEE accounted for 7 clusters defined by cluster." in report.narrative
    assert any("GEE models are population-averaged" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {"coefficient_forest.png"}
    assert audit.metadata["cluster_count"] == 7
    assert audit.metadata["negative_binomial_alpha"] == 1.0
