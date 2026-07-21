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
from src.statistics.diagnostics.heckman import (
    build_heckman_diagnostics,
    heckman_diagnostic_summary_to_dataframe,
    heckman_residuals_to_dataframe,
    heckman_selection_coefficients_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.heckman import fit_heckman_selection
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _heckman_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260815)
    n = 180
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    exclusion = rng.normal(size=n)
    shared = rng.normal(size=n)
    selection_latent = -0.15 + 0.5 * x + 0.35 * z + 1.15 * exclusion + 0.7 * shared + rng.normal(
        0.0,
        0.55,
        size=n,
    )
    selected = (selection_latent > 0.0).astype(int)
    y_latent = 1.0 + 0.8 * x - 0.35 * z + 0.65 * shared + rng.normal(0.0, 0.35, size=n)
    y = np.where(selected == 1, y_latent, np.nan)
    return pd.DataFrame(
        {
            "y": y,
            "selected": selected,
            "x": x,
            "z": z,
            "exclusion": exclusion,
        }
    )


def test_fit_heckman_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _heckman_data()
    result = fit_heckman_selection(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        selection_variable="selected",
        selection_variables=["x", "z", "exclusion"],
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

    assert result.model_type == "heckman_selection"
    assert 0.1 < result.fit_statistics["selection_rate"] < 0.9
    assert result.fit_statistics["exclusion_restriction_count"] == 1
    assert result.metadata["exclusion_restrictions"] == ["exclusion"]
    assert any(effect.effect_type == "heckman_standardized_beta" for effect in effects.effects)
    assert "Heckman selection modeled observation through selected." in report.narrative
    assert any("Heckman selection reports" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["selection_variable"] == "selected"


def test_selector_routes_explicit_heckman_selection() -> None:
    result = fit_regression_by_level(
        _heckman_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="heckman_selection",
        mixed_effects_options={
            "selection_variable": "selected",
            "selection_variables": ["x", "z", "exclusion"],
        },
    )

    assert result.model_type == "heckman_selection"
    assert result.metadata["selection_variable"] == "selected"
    assert result.fit_statistics["exclusion_restriction_count"] == 1


def test_heckman_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _heckman_data()
    result = fit_heckman_selection(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        selection_variable="selected",
        selection_variables=["x", "z", "exclusion"],
        model_id="main_model",
    )
    diagnostics = build_heckman_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="heckman diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.sample_size == result.sample_size
    assert heckman_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert heckman_selection_coefficients_to_dataframe(diagnostics).shape[0] >= 3
    assert "inverse_mills_p_value" in set(heckman_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "heckman_selection"
    assert any("Heckman diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_heckman_pipeline(tmp_path: Path) -> None:
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
                        "estimator": "heckman",
                        "selection_variable": "selected",
                        "selection_variables": ["x", "z", "exclusion"],
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
            "selected": VariableDefinition(role="other", measurement_level="binary"),
            "exclusion": VariableDefinition(role="other", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="heckman builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "heckman_selection"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
