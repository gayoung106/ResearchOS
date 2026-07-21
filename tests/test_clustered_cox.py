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
from src.statistics.diagnostics.cox import (
    build_cox_diagnostics,
    cox_baseline_survival_to_dataframe,
    cox_diagnostic_summary_to_dataframe,
    cox_ph_checks_to_dataframe,
    cox_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.cox import fit_clustered_cox
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _clustered_survival_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260818)
    cluster_count = 16
    cluster_size = 12
    n = cluster_count * cluster_size
    cluster = np.repeat([f"clinic_{index:02d}" for index in range(cluster_count)], cluster_size)
    cluster_effect = np.repeat(rng.normal(0.0, 0.35, size=cluster_count), cluster_size)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    hazard = np.exp(0.55 * x - 0.25 * z + cluster_effect)
    event_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.exponential(scale=2.1, size=n)
    event = (event_time <= censor_time).astype(int)
    duration = np.minimum(event_time, censor_time) + 0.01
    return pd.DataFrame({"duration": duration, "event": event, "clinic": cluster, "x": x, "z": z})


def test_fit_clustered_cox_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _clustered_survival_data()
    result = fit_clustered_cox(
        data,
        duration_variable="duration",
        event_variable="event",
        cluster_variable="clinic",
        independent_variables=["x", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "clustered_cox"
    assert result.standard_error_type == "cluster_robust_partial_likelihood"
    assert result.metadata["cluster_variable"] == "clinic"
    assert result.fit_statistics["cluster_count"] == 16
    assert any(effect.effect_type == "hazard_ratio" for effect in effects.effects)
    assert effects.metadata["cluster_variable"] == "clinic"
    assert "Cluster-robust standard errors accounted for 16 clusters" in report.narrative
    assert report.metadata["cluster_variable"] == "clinic"
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "cox_baseline_survival.png",
    }
    assert audit.metadata["cluster_variable"] == "clinic"
    assert audit.metadata["cluster_count"] == 16


def test_selector_routes_clustered_cox_model() -> None:
    result = fit_regression_by_level(
        _clustered_survival_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="clustered_cox",
        mixed_effects_options={"event_variable": "event", "cluster_variable": "clinic"},
    )

    assert result.model_type == "clustered_cox"
    assert result.metadata["duration_variable"] == "duration"
    assert result.metadata["event_variable"] == "event"
    assert result.metadata["cluster_variable"] == "clinic"


def test_clustered_cox_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _clustered_survival_data()
    result = fit_clustered_cox(
        data,
        duration_variable="duration",
        event_variable="event",
        cluster_variable="clinic",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_cox_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="clustered cox diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "clustered_cox"
    assert cox_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert cox_ph_checks_to_dataframe(diagnostics).shape[0] == len(result.coefficients)
    assert cox_baseline_survival_to_dataframe(diagnostics).shape[0] > 0
    assert "events_per_parameter" in set(cox_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_clustered_cox_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["duration"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {
                        "estimator": "clustered_cox",
                        "event_variable": "event",
                        "cluster_variable": "clinic",
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "duration": VariableDefinition(role="dependent", measurement_level="continuous"),
            "event": VariableDefinition(role="other", measurement_level="binary"),
            "clinic": VariableDefinition(role="other", measurement_level="nominal"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="clustered cox builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "clustered_cox"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
