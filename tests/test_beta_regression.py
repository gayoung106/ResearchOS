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
from src.statistics.diagnostics.beta import (
    beta_diagnostic_summary_to_dataframe,
    beta_observations_to_dataframe,
    build_beta_diagnostics,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.beta import fit_beta_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _beta_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260805)
    n = 140
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    mean = 1.0 / (1.0 + np.exp(-(-0.2 + 0.75 * x - 0.35 * z)))
    precision = 14.0
    y = rng.beta(mean * precision, (1.0 - mean) * precision)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_beta_regression_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _beta_data()
    result = fit_beta_regression(
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

    assert result.model_type == "beta_regression"
    assert result.fit_statistics["precision"] > 0
    assert any(effect.effect_type == "mean_odds_ratio" for effect in effects.effects)
    assert "Beta regression pseudo R-squared" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "beta_observed_vs_predicted.png",
    }
    assert audit.metadata["precision"] == result.fit_statistics["precision"]


def test_selector_routes_explicit_beta_regression() -> None:
    result = fit_regression_by_level(
        _beta_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="proportion",
        model_type="beta_regression",
    )

    assert result.model_type == "beta_regression"
    assert result.fit_statistics["pseudo_r_squared"] > 0


def test_beta_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _beta_data()
    result = fit_beta_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_beta_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="beta diagnostics"),
        tmp_path,
    )

    assert diagnostics.sample_size == result.sample_size
    assert beta_observations_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "precision" in set(beta_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_beta_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "beta"},
                },
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
        context=ResearchContext(project_name="beta builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "beta_regression"
    assert registration.measurement_level == "proportion"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_beta_regression_rejects_boundary_values() -> None:
    data = _beta_data()
    data.loc[0, "y"] = 0.0

    try:
        fit_beta_regression(
            data,
            dependent_variable="y",
            independent_variables=["x", "z"],
        )
    except ValueError as error:
        assert "fractional_logit" in str(error)
    else:
        raise AssertionError("beta regression should reject boundary values")
