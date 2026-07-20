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
from src.statistics.diagnostics.quantile import (
    build_quantile_diagnostics,
    quantile_diagnostic_summary_to_dataframe,
    quantile_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.quantile import fit_quantile_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _quantile_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260802)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    noise = rng.laplace(loc=0.0, scale=0.6 + 0.25 * np.abs(x), size=n)
    y = 1.0 + 0.7 * x - 0.3 * z + noise
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_quantile_regression_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _quantile_data()
    result = fit_quantile_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        quantile=0.75,
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "quantile_regression"
    assert result.fit_statistics["quantile"] == 0.75
    assert result.fit_statistics["pinball_loss"] > 0
    assert any(effect.effect_type == "standardized_quantile_beta" for effect in effects.effects)
    assert "q=0.75" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["quantile"] == 0.75


def test_selector_routes_explicit_quantile_regression() -> None:
    result = fit_regression_by_level(
        _quantile_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="quantile_regression",
        mixed_effects_options={"quantile": 0.25},
    )

    assert result.model_type == "quantile_regression"
    assert result.fit_statistics["quantile"] == 0.25


def test_quantile_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _quantile_data()
    result = fit_quantile_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
        quantile=0.5,
    )
    diagnostics = build_quantile_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="quantile diagnostics"),
        tmp_path,
    )

    assert diagnostics.quantile == 0.5
    assert quantile_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "pinball_loss" in set(quantile_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_quantile_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "quantile", "quantile": 0.75},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="continuous"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="quantile builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "quantile_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
