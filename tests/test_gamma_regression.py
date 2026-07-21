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
from src.statistics.diagnostics.gamma import (
    build_gamma_diagnostics,
    gamma_diagnostic_summary_to_dataframe,
    gamma_observations_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.gamma import fit_gamma_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _gamma_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260812)
    n = 140
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    mean = np.exp(0.4 + 0.45 * x - 0.25 * z)
    shape = 6.0
    y = rng.gamma(shape=shape, scale=mean / shape)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_gamma_regression_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _gamma_data()
    result = fit_gamma_regression(
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

    assert result.model_type == "gamma_regression"
    assert result.fit_statistics["minimum_observed"] > 0
    assert result.fit_statistics["root_mean_squared_error"] > 0
    assert any(effect.effect_type == "mean_ratio" for effect in effects.effects)
    assert "Gamma deviance pseudo R-squared" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "gamma_observed_vs_predicted.png",
    }
    assert audit.metadata["dispersion_ratio"] == result.fit_statistics["dispersion_ratio"]


def test_selector_routes_explicit_gamma_regression() -> None:
    result = fit_regression_by_level(
        _gamma_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="gamma_regression",
    )

    assert result.model_type == "gamma_regression"
    assert result.metadata["family"] == "gamma"
    assert result.metadata["link"] == "log"


def test_gamma_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _gamma_data()
    result = fit_gamma_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_gamma_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="gamma diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.sample_size == result.sample_size
    assert gamma_observations_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "dispersion_ratio" in set(gamma_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "gamma_regression"
    assert any("Gamma diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_gamma_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "gamma"},
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
        context=ResearchContext(project_name="gamma builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "gamma_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
