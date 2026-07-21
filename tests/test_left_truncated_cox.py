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
from src.statistics.regression.cox import fit_left_truncated_cox
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _left_truncated_survival_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260816)
    n = 170
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    entry = rng.uniform(0.05, 0.9, size=n)
    hazard = np.exp(0.55 * x - 0.25 * z)
    event_wait = rng.exponential(scale=1.0 / hazard)
    censor_wait = rng.exponential(scale=2.0, size=n)
    event = (event_wait <= censor_wait).astype(int)
    duration = entry + np.minimum(event_wait, censor_wait) + 0.01
    return pd.DataFrame({"duration": duration, "entry": entry, "event": event, "x": x, "z": z})


def test_fit_left_truncated_cox_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _left_truncated_survival_data()
    result = fit_left_truncated_cox(
        data,
        duration_variable="duration",
        event_variable="event",
        entry_variable="entry",
        independent_variables=["x", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "left_truncated_cox"
    assert result.metadata["entry_variable"] == "entry"
    assert result.fit_statistics["left_truncated_count"] == result.sample_size
    assert result.fit_statistics["maximum_entry_time"] > result.fit_statistics["minimum_entry_time"]
    assert any(effect.effect_type == "hazard_ratio" for effect in effects.effects)
    assert effects.metadata["entry_variable"] == "entry"
    assert "delayed entry from entry" in report.narrative
    assert report.metadata["entry_variable"] == "entry"
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "cox_baseline_survival.png",
    }
    assert audit.metadata["entry_variable"] == "entry"
    assert audit.metadata["left_truncated_count"] == result.sample_size


def test_selector_routes_left_truncated_cox_model() -> None:
    result = fit_regression_by_level(
        _left_truncated_survival_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="left_truncated_cox",
        mixed_effects_options={"event_variable": "event", "entry_variable": "entry"},
    )

    assert result.model_type == "left_truncated_cox"
    assert result.metadata["duration_variable"] == "duration"
    assert result.metadata["event_variable"] == "event"
    assert result.metadata["entry_variable"] == "entry"


def test_left_truncated_cox_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _left_truncated_survival_data()
    result = fit_left_truncated_cox(
        data,
        duration_variable="duration",
        event_variable="event",
        entry_variable="entry",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_cox_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="left truncated cox diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "left_truncated_cox"
    assert cox_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert cox_ph_checks_to_dataframe(diagnostics).shape[0] == len(result.coefficients)
    assert cox_baseline_survival_to_dataframe(diagnostics).shape[0] > 0
    assert "events_per_parameter" in set(cox_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_left_truncated_cox_pipeline(tmp_path: Path) -> None:
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
                        "estimator": "left_truncated_cox",
                        "event_variable": "event",
                        "entry_variable": "entry",
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "duration": VariableDefinition(role="dependent", measurement_level="continuous"),
            "entry": VariableDefinition(role="other", measurement_level="continuous"),
            "event": VariableDefinition(role="other", measurement_level="binary"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="left truncated cox builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "left_truncated_cox"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
