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
from src.statistics.regression.cox import fit_stratified_cox
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _stratified_survival_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260815)
    n = 180
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    site = rng.choice(["north", "south", "west"], size=n, p=[0.35, 0.35, 0.30])
    baseline = np.select([site == "north", site == "south", site == "west"], [0.7, 1.2, 1.7])
    hazard = baseline * np.exp(0.55 * x - 0.30 * z)
    event_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.exponential(scale=2.3, size=n)
    event = (event_time <= censor_time).astype(int)
    duration = np.minimum(event_time, censor_time) + 0.01
    data = pd.DataFrame({"duration": duration, "event": event, "site": site, "x": x, "z": z})
    for label in ["north", "south", "west"]:
        if int(data.loc[data["site"] == label, "event"].sum()) == 0:
            first = data.index[data["site"] == label][0]
            data.loc[first, "event"] = 1
    return data


def test_fit_stratified_cox_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _stratified_survival_data()
    result = fit_stratified_cox(
        data,
        duration_variable="duration",
        event_variable="event",
        strata_variable="site",
        independent_variables=["x", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "stratified_cox"
    assert result.metadata["strata_variable"] == "site"
    assert result.metadata["strata_count"] == 3
    assert result.fit_statistics["strata_count"] == 3
    assert len(result.metadata["strata_event_counts"]) == 3
    assert any(effect.effect_type == "hazard_ratio" for effect in effects.effects)
    assert effects.metadata["strata_variable"] == "site"
    assert "stratified by site" in report.narrative
    assert report.metadata["strata_variable"] == "site"
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "cox_baseline_survival.png",
    }
    assert audit.metadata["strata_variable"] == "site"
    assert audit.metadata["strata_count"] == 3


def test_selector_routes_stratified_cox_model() -> None:
    result = fit_regression_by_level(
        _stratified_survival_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="stratified_cox",
        mixed_effects_options={"event_variable": "event", "strata_variable": "site"},
    )

    assert result.model_type == "stratified_cox"
    assert result.metadata["duration_variable"] == "duration"
    assert result.metadata["event_variable"] == "event"
    assert result.metadata["strata_variable"] == "site"


def test_stratified_cox_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _stratified_survival_data()
    result = fit_stratified_cox(
        data,
        duration_variable="duration",
        event_variable="event",
        strata_variable="site",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_cox_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="stratified cox diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "stratified_cox"
    assert cox_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert cox_ph_checks_to_dataframe(diagnostics).shape[0] == len(result.coefficients)
    assert cox_baseline_survival_to_dataframe(diagnostics)["stratum"].nunique() == 3
    assert "events_per_parameter" in set(cox_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_stratified_cox_pipeline(tmp_path: Path) -> None:
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
                        "estimator": "stratified_cox",
                        "event_variable": "event",
                        "strata_variable": "site",
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
            "site": VariableDefinition(role="other", measurement_level="nominal"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="stratified cox builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "stratified_cox"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
