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
from src.statistics.regression.boxcox import fit_boxcox_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _positive_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260818)
    n = 130
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    log_mu = 0.5 + 0.35 * x - 0.2 * z + rng.normal(0.0, 0.25, size=n)
    y = np.exp(log_mu)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_boxcox_regression_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _positive_data()
    result = fit_boxcox_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "boxcox_regression"
    assert np.isfinite(result.fit_statistics["boxcox_lambda"])
    assert result.fit_statistics["original_scale_root_mean_squared_error"] > 0
    assert result.metadata["lambda_estimated"] is True
    assert any(effect.effect_type == "standardized_beta" for effect in effects.effects)
    assert "Box-Cox regression used lambda" in report.narrative
    assert any("Box-Cox regression coefficients" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["boxcox_lambda"] == result.fit_statistics["boxcox_lambda"]


def test_selector_routes_explicit_boxcox_regression() -> None:
    result = fit_regression_by_level(
        _positive_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="boxcox_regression",
        mixed_effects_options={"lambda_value": 0.0},
    )

    assert result.model_type == "boxcox_regression"
    assert result.fit_statistics["boxcox_lambda"] == 0.0
    assert result.metadata["lambda_estimated"] is False


def test_boxcox_regression_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _positive_data()
    result = fit_boxcox_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_ols_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="boxcox diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.sample_size == result.sample_size
    assert residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "large_residual_count" in set(diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"
    assert any("Box-Cox diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_boxcox_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "boxcox"},
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
        context=ResearchContext(project_name="boxcox builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "boxcox_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
