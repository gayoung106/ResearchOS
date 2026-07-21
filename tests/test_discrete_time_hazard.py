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
from src.statistics.diagnostics.discrete_time_hazard import (
    build_discrete_time_hazard_diagnostics,
    discrete_time_diagnostic_summary_to_dataframe,
    discrete_time_interval_hazards_to_dataframe,
    discrete_time_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.discrete_time_hazard import fit_discrete_time_hazard_model
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _survival_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260820)
    n = 150
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    hazard = np.exp(0.50 * x - 0.30 * z)
    event_time = rng.exponential(scale=1.0 / hazard)
    censor_time = rng.exponential(scale=1.8, size=n)
    event = (event_time <= censor_time).astype(int)
    duration = np.minimum(event_time, censor_time) + 0.01
    return pd.DataFrame({"duration": duration, "event": event, "x": x, "z": z})


def test_fit_discrete_time_hazard_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _survival_data()
    result = fit_discrete_time_hazard_model(
        data,
        duration_variable="duration",
        event_variable="event",
        independent_variables=["x", "z"],
        breakpoints=[0.4, 0.9, 1.6],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "discrete_time_hazard"
    assert result.metadata["link"] == "logit"
    assert result.fit_statistics["interval_count"] == 4
    assert result.fit_statistics["long_row_count"] > result.sample_size
    assert any(effect.effect_type == "hazard_odds_ratio" for effect in effects.effects)
    assert all(not effect.term.startswith("baseline_interval_") for effect in effects.effects)
    assert "Person-period data used" in report.narrative
    assert report.metadata["discrete_time_link"] == "logit"
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "discrete_time_hazard_survival_curve.png",
    }
    assert audit.metadata["discrete_time_link"] == "logit"
    assert audit.metadata["interval_count"] == 4


def test_selector_routes_discrete_time_hazard_cloglog_model() -> None:
    result = fit_regression_by_level(
        _survival_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="discrete_time_hazard",
        mixed_effects_options={"event_variable": "event", "breakpoints": [0.4, 0.9, 1.6], "link": "cloglog"},
    )
    effects = build_regression_effect_size_report(result)

    assert result.model_type == "discrete_time_hazard"
    assert result.metadata["duration_variable"] == "duration"
    assert result.metadata["event_variable"] == "event"
    assert result.metadata["link"] == "cloglog"
    assert any(effect.effect_type == "discrete_hazard_ratio" for effect in effects.effects)


def test_discrete_time_hazard_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _survival_data()
    result = fit_discrete_time_hazard_model(
        data,
        duration_variable="duration",
        event_variable="event",
        independent_variables=["x", "z"],
        breakpoints=[0.4, 0.9, 1.6],
        model_id="main_model",
    )
    diagnostics = build_discrete_time_hazard_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="discrete time hazard diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "discrete_time_hazard"
    assert discrete_time_interval_hazards_to_dataframe(diagnostics).shape[0] == 4
    assert discrete_time_residuals_to_dataframe(diagnostics).shape[0] == result.fit_statistics["long_row_count"]
    assert "person_period_event_rate" in set(discrete_time_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 3
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_explicit_discrete_time_hazard_pipeline(tmp_path: Path) -> None:
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
                        "estimator": "discrete_time_hazard",
                        "event_variable": "event",
                        "breakpoints": [0.4, 0.9, 1.6],
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
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="discrete time hazard builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "discrete_time_hazard"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
