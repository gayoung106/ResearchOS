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
from src.statistics.diagnostics.ols import (
    build_ols_diagnostics,
    diagnostic_summary_to_dataframe,
    residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.selector import fit_regression_by_level
from src.statistics.regression.weighted_least_squares import fit_weighted_least_squares
from src.visualization.regression import build_regression_visualizations


def _wls_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260819)
    n = 150
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    weights = rng.uniform(0.5, 3.0, size=n)
    noise = rng.normal(0.0, 0.55 / np.sqrt(weights), size=n)
    y = 1.2 + 0.85 * x - 0.35 * z + noise
    return pd.DataFrame({"y": y, "x": x, "z": z, "weight": weights})


def test_fit_weighted_least_squares_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _wls_data()
    result = fit_weighted_least_squares(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        weight_variable="weight",
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

    assert result.model_type == "weighted_least_squares"
    assert result.metadata["weight_variable"] == "weight"
    assert result.fit_statistics["weight_sum"] > result.sample_size * 0.5
    assert any(effect.effect_type == "standardized_beta" for effect in effects.effects)
    assert "WLS used analytic weights from weight." in report.narrative
    assert any("Weighted least squares reports" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["weight_variable"] == "weight"


def test_selector_routes_explicit_weighted_least_squares() -> None:
    result = fit_regression_by_level(
        _wls_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="weighted_least_squares",
        mixed_effects_options={"weight_variable": "weight"},
    )

    assert result.model_type == "weighted_least_squares"
    assert result.fit_statistics["weight_ratio"] > 1


def test_weighted_least_squares_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _wls_data()
    result = fit_weighted_least_squares(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        weight_variable="weight",
        model_id="main_model",
    )
    diagnostics = build_ols_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="wls diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.model_id == "main_model"
    assert residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "diagnostic_warning_count" in set(diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"
    assert any("WLS diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_weighted_least_squares_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {
                        "estimator": "wls",
                        "weight_variable": "weight",
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
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
            "weight": VariableDefinition(role="weight", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="wls builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "weighted_least_squares"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
