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
from src.statistics.diagnostics.lognormal_aft import (
    build_lognormal_aft_diagnostics,
    lognormal_aft_diagnostic_summary_to_dataframe,
    lognormal_aft_prediction_metrics_to_dataframe,
    lognormal_aft_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.lognormal_aft import fit_lognormal_aft
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _survival_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260902)
    n = 150
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    log_time = 1.05 + 0.38 * x - 0.22 * z + rng.normal(scale=0.5, size=n)
    event_time = np.exp(log_time)
    censor_time = rng.exponential(scale=4.5, size=n)
    event = (event_time <= censor_time).astype(int)
    duration = np.minimum(event_time, censor_time) + 0.02
    return pd.DataFrame({"duration": duration, "event": event, "x": x, "z": z})


def test_fit_lognormal_aft_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _survival_data()
    result = fit_lognormal_aft(
        data,
        duration_variable="duration",
        event_variable="event",
        independent_variables=["x", "z"],
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

    assert result.model_type == "lognormal_aft"
    assert result.metadata["distribution"] == "lognormal"
    assert result.fit_statistics["event_count"] > 20
    assert result.fit_statistics["sigma"] > 0
    assert any(effect.effect_type == "time_ratio" for effect in effects.effects)
    assert "log-normal AFT" in report.narrative
    assert any("Log-normal AFT models report time ratios" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "lognormal_aft_survival_curve.png",
    }
    assert audit.metadata["lognormal_sigma"] == result.fit_statistics["sigma"]


def test_selector_routes_explicit_lognormal_aft() -> None:
    result = fit_regression_by_level(
        _survival_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="lognormal_aft",
        mixed_effects_options={"event_variable": "event"},
    )

    assert result.model_type == "lognormal_aft"
    assert result.metadata["event_variable"] == "event"


def test_lognormal_aft_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _survival_data()
    result = fit_lognormal_aft(
        data,
        duration_variable="duration",
        event_variable="event",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_lognormal_aft_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="lognormal aft diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.model_type == "lognormal_aft"
    assert lognormal_aft_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "concordance_index" in set(lognormal_aft_prediction_metrics_to_dataframe(diagnostics)["item"])
    assert "sigma" in set(lognormal_aft_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"
    assert any("Log-normal AFT diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_lognormal_aft_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["duration"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "lognormal_aft", "event_variable": "event"},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "duration": VariableDefinition(role="dependent", measurement_level="continuous"),
            "event": VariableDefinition(role="other", measurement_level="binary"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="lognormal aft builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "lognormal_aft"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
