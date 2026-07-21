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
from src.statistics.regression.cox import fit_time_varying_cox
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _time_varying_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260821)
    n = 130
    rows = []
    split = 0.65
    for subject in range(n):
        x0 = rng.normal()
        z = rng.normal()
        delta = rng.normal(0.25, 0.35)
        hazard = np.exp(0.45 * x0 - 0.25 * z)
        event_time = rng.exponential(scale=1.0 / hazard)
        censor_time = rng.exponential(scale=1.9)
        event = int(event_time <= censor_time)
        stop_time = min(event_time, censor_time) + 0.01
        if stop_time <= split:
            rows.append(
                {
                    "subject": f"s{subject:03d}",
                    "start": 0.0,
                    "stop": stop_time,
                    "event": event,
                    "x_tv": x0,
                    "z": z,
                }
            )
        else:
            rows.append(
                {
                    "subject": f"s{subject:03d}",
                    "start": 0.0,
                    "stop": split,
                    "event": 0,
                    "x_tv": x0,
                    "z": z,
                }
            )
            rows.append(
                {
                    "subject": f"s{subject:03d}",
                    "start": split,
                    "stop": stop_time,
                    "event": event,
                    "x_tv": x0 + delta,
                    "z": z,
                }
            )
    return pd.DataFrame(rows)


def test_fit_time_varying_cox_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _time_varying_data()
    result = fit_time_varying_cox(
        data,
        start_variable="start",
        stop_variable="stop",
        event_variable="event",
        subject_variable="subject",
        independent_variables=["x_tv", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "time_varying_cox"
    assert result.standard_error_type == "cluster_robust_partial_likelihood"
    assert result.metadata["start_variable"] == "start"
    assert result.metadata["stop_variable"] == "stop"
    assert result.metadata["subject_variable"] == "subject"
    assert result.fit_statistics["subject_count"] == 130
    assert result.fit_statistics["time_varying_row_count"] == len(data)
    assert any(effect.effect_type == "hazard_ratio" for effect in effects.effects)
    assert effects.metadata["subject_variable"] == "subject"
    assert "Time-varying Cox used" in report.narrative
    assert report.metadata["start_variable"] == "start"
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "cox_baseline_survival.png",
    }
    assert audit.metadata["subject_variable"] == "subject"
    assert audit.metadata["time_varying_row_count"] == len(data)


def test_selector_routes_time_varying_cox_model() -> None:
    data = _time_varying_data()
    result = fit_regression_by_level(
        data,
        dependent_variable="stop",
        independent_variables=["x_tv", "z"],
        measurement_level="continuous",
        model_type="time_varying_cox",
        mixed_effects_options={
            "start_variable": "start",
            "event_variable": "event",
            "subject_variable": "subject",
        },
    )

    assert result.model_type == "time_varying_cox"
    assert result.metadata["duration_variable"] == "stop"
    assert result.metadata["start_variable"] == "start"
    assert result.metadata["subject_variable"] == "subject"


def test_time_varying_cox_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _time_varying_data()
    result = fit_time_varying_cox(
        data,
        start_variable="start",
        stop_variable="stop",
        event_variable="event",
        subject_variable="subject",
        independent_variables=["x_tv", "z"],
        model_id="main_model",
    )
    diagnostics = build_cox_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="time varying cox diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "time_varying_cox"
    assert cox_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert cox_ph_checks_to_dataframe(diagnostics).shape[0] == len(result.coefficients)
    assert cox_baseline_survival_to_dataframe(diagnostics).shape[0] > 0
    assert "events_per_parameter" in set(cox_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_time_varying_cox_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["stop"],
                "independent": ["x_tv", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {
                        "estimator": "time_varying_cox",
                        "start_variable": "start",
                        "event_variable": "event",
                        "subject_variable": "subject",
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "stop": VariableDefinition(role="dependent", measurement_level="continuous"),
            "start": VariableDefinition(role="other", measurement_level="continuous"),
            "event": VariableDefinition(role="other", measurement_level="binary"),
            "subject": VariableDefinition(role="other", measurement_level="nominal"),
            "x_tv": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="time varying cox builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "time_varying_cox"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
