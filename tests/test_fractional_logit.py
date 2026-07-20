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
from src.statistics.diagnostics.fractional_logit import (
    build_fractional_logit_diagnostics,
    fractional_diagnostic_summary_to_dataframe,
    fractional_observations_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.fractional_logit import fit_fractional_logit
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _proportion_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260804)
    n = 130
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    linear = -0.15 + 0.8 * x - 0.45 * z
    mean = 1.0 / (1.0 + np.exp(-linear))
    y = np.clip(mean + rng.normal(0.0, 0.08, size=n), 0.0, 1.0)
    y[:3] = [0.0, 1.0, 0.0]
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_fractional_logit_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _proportion_data()
    result = fit_fractional_logit(
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

    assert result.model_type == "fractional_logit"
    assert result.fit_statistics["boundary_count"] >= 3
    assert any(effect.effect_type == "fractional_odds_ratio" for effect in effects.effects)
    assert "Boundary observations" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "fractional_observed_vs_predicted.png",
    }
    assert audit.metadata["boundary_count"] >= 3


def test_selector_routes_proportion_to_fractional_logit() -> None:
    result = fit_regression_by_level(
        _proportion_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="proportion",
    )

    assert result.model_type == "fractional_logit"
    assert result.fit_statistics["mean_absolute_error"] >= 0


def test_fractional_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _proportion_data()
    result = fit_fractional_logit(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_fractional_logit_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="fractional diagnostics"),
        tmp_path,
    )

    assert diagnostics.sample_size == result.sample_size
    assert fractional_observations_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "root_mean_squared_error" in set(
        fractional_diagnostic_summary_to_dataframe(diagnostics)["item"]
    )
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_proportion_fractional_logit_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="proportion"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="fractional builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "fractional_logit"
    assert registration.measurement_level == "proportion"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
