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
from src.statistics.diagnostics.regularized import (
    build_regularized_diagnostics,
    regularized_diagnostic_summary_to_dataframe,
    regularized_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.regularized import fit_regularized_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _regularized_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260811)
    n = 130
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    noise_1 = rng.normal(size=n)
    noise_2 = rng.normal(size=n)
    y = 0.8 + 1.1 * x - 0.55 * z + rng.normal(0.0, 0.35, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "noise_1": noise_1, "noise_2": noise_2})


def test_fit_regularized_regression_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _regularized_data()
    result = fit_regularized_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z", "noise_1", "noise_2"],
        penalty="lasso",
        alpha=0.08,
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "regularized_regression"
    assert result.fit_statistics["penalty"] == "lasso"
    assert result.fit_statistics["selected_coefficient_count"] > 0
    assert "selected_terms" in result.metadata
    assert any(effect.effect_type == "regularized_standardized_beta" for effect in effects.effects)
    assert "Regularized regression used lasso penalty" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["penalty"] == "lasso"


def test_selector_routes_explicit_regularized_regression() -> None:
    result = fit_regression_by_level(
        _regularized_data(),
        dependent_variable="y",
        independent_variables=["x", "z", "noise_1", "noise_2"],
        measurement_level="continuous",
        model_type="regularized_regression",
        mixed_effects_options={"penalty": "ridge", "alpha": 0.2},
    )

    assert result.model_type == "regularized_regression"
    assert result.fit_statistics["penalty"] == "ridge"
    assert result.fit_statistics["l1_ratio"] == 0.0


def test_regularized_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _regularized_data()
    result = fit_regularized_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z", "noise_1", "noise_2"],
        penalty="elastic_net",
        alpha=0.05,
        l1_ratio=0.5,
        model_id="main_model",
    )
    diagnostics = build_regularized_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="regularized diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.sample_size == result.sample_size
    assert regularized_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "selected_coefficient_count" in set(
        regularized_diagnostic_summary_to_dataframe(diagnostics)["item"]
    )
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "regularized_regression"
    assert any("Regularized diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_regularized_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z", "noise_1", "noise_2"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "regularized", "penalty": "elastic_net", "alpha": 0.1},
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
            "noise_1": VariableDefinition(role="independent", measurement_level="continuous"),
            "noise_2": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="regularized builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "regularized_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
