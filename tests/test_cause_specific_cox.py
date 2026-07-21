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
from src.statistics.regression.cox import fit_cause_specific_cox
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _competing_risks_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260817)
    n = 190
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    hazard_1 = np.exp(0.55 * x - 0.25 * z)
    hazard_2 = np.exp(-0.25 * x + 0.35 * z)
    target_time = rng.exponential(scale=1.0 / hazard_1)
    competing_time = rng.exponential(scale=1.0 / hazard_2)
    censor_time = rng.exponential(scale=2.4, size=n)
    duration = np.minimum(np.minimum(target_time, competing_time), censor_time) + 0.01
    cause = np.where(
        (target_time <= competing_time) & (target_time <= censor_time),
        1,
        np.where((competing_time < target_time) & (competing_time <= censor_time), 2, 0),
    )
    return pd.DataFrame({"duration": duration, "cause": cause, "x": x, "z": z})


def test_fit_cause_specific_cox_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _competing_risks_data()
    result = fit_cause_specific_cox(
        data,
        duration_variable="duration",
        cause_variable="cause",
        target_event_code=1,
        independent_variables=["x", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "cause_specific_cox"
    assert result.metadata["cause_variable"] == "cause"
    assert result.metadata["target_event_code"] == 1
    assert result.fit_statistics["cause_specific_event_count"] == int((data["cause"] == 1).sum())
    assert result.fit_statistics["competing_event_count"] == int((data["cause"] == 2).sum())
    assert any(effect.effect_type == "hazard_ratio" for effect in effects.effects)
    assert effects.metadata["target_event_code"] == 1
    assert "Cause-specific Cox treated cause=1" in report.narrative
    assert report.metadata["target_event_code"] == 1
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "cox_baseline_survival.png",
    }
    assert audit.metadata["cause_variable"] == "cause"
    assert audit.metadata["competing_event_count"] == int((data["cause"] == 2).sum())


def test_selector_routes_cause_specific_cox_model() -> None:
    result = fit_regression_by_level(
        _competing_risks_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="cause_specific_cox",
        mixed_effects_options={"cause_variable": "cause", "target_event_code": 1},
    )

    assert result.model_type == "cause_specific_cox"
    assert result.metadata["duration_variable"] == "duration"
    assert result.metadata["cause_variable"] == "cause"
    assert result.metadata["target_event_code"] == 1


def test_cause_specific_cox_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _competing_risks_data()
    result = fit_cause_specific_cox(
        data,
        duration_variable="duration",
        cause_variable="cause",
        target_event_code=1,
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_cox_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="cause specific cox diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "cause_specific_cox"
    assert cox_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert cox_ph_checks_to_dataframe(diagnostics).shape[0] == len(result.coefficients)
    assert cox_baseline_survival_to_dataframe(diagnostics).shape[0] > 0
    assert "events_per_parameter" in set(cox_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_cause_specific_cox_pipeline(tmp_path: Path) -> None:
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
                        "estimator": "cause_specific_cox",
                        "cause_variable": "cause",
                        "target_event_code": 1,
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "duration": VariableDefinition(role="dependent", measurement_level="continuous"),
            "cause": VariableDefinition(role="other", measurement_level="nominal"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="cause specific cox builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "cause_specific_cox"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
