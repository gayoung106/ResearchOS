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
from src.statistics.diagnostics.tobit import (
    build_tobit_diagnostics,
    tobit_diagnostic_summary_to_dataframe,
    tobit_observations_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.selector import fit_regression_by_level
from src.statistics.regression.truncated import fit_truncated_regression
from src.visualization.regression import build_regression_visualizations


def _truncated_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260815)
    n = 260
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.5 + 0.75 * x - 0.25 * z + rng.normal(0.0, 0.55, size=n)
    data = pd.DataFrame({"y": y, "x": x, "z": z})
    return data[data["y"] > 0.0].reset_index(drop=True)


def test_fit_truncated_regression_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _truncated_data()
    result = fit_truncated_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        lower_limit=0.0,
        maximum_iterations=120,
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "truncated_regression"
    assert result.fit_statistics["left_truncation_limit"] == 0.0
    assert result.fit_statistics["sigma"] > 0
    assert result.fit_statistics["truncated_sample_count"] == result.sample_size
    assert any(effect.effect_type == "truncated_standardized_beta" for effect in effects.effects)
    assert "Truncated normal regression" in report.narrative
    assert any("Truncated regression" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["left_truncation_limit"] == 0.0
    assert audit.metadata["truncated_sample_count"] == result.sample_size


def test_selector_routes_explicit_truncated_regression() -> None:
    result = fit_regression_by_level(
        _truncated_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="truncated_regression",
        mixed_effects_options={"lower_limit": 0.0, "maximum_iterations": 120},
    )

    assert result.model_type == "truncated_regression"
    assert result.metadata["lower_limit"] == 0.0


def test_truncated_regression_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _truncated_data()
    result = fit_truncated_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        lower_limit=0.0,
        model_id="main_model",
        maximum_iterations=120,
    )
    diagnostics = build_tobit_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="truncated diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.model_type == "truncated_regression"
    assert tobit_observations_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "truncated_sample_count" in set(tobit_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "truncated_regression"
    assert any("Truncated regression diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_truncated_regression_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "truncated", "lower_limit": 0.0},
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
        context=ResearchContext(project_name="truncated builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "truncated_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
