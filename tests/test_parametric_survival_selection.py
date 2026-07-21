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
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.parametric_survival import fit_parametric_survival_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _survival_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260905)
    n = 150
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    log_time = 1.0 + 0.34 * x - 0.22 * z + rng.normal(scale=0.45, size=n)
    event_time = np.exp(log_time)
    censor_time = rng.exponential(scale=4.6, size=n)
    event = (event_time <= censor_time).astype(int)
    duration = np.minimum(event_time, censor_time) + 0.02
    return pd.DataFrame({"duration": duration, "event": event, "x": x, "z": z})


def test_fit_parametric_survival_selects_and_integrates_outputs(tmp_path: Path) -> None:
    data = _survival_data()
    result = fit_parametric_survival_regression(
        data,
        duration_variable="duration",
        event_variable="event",
        independent_variables=["x", "z"],
        candidate_models=["exponential_aft", "weibull_aft", "lognormal_aft", "loglogistic_aft"],
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

    assert result.model_type in {"exponential_aft", "weibull_aft", "lognormal_aft", "loglogistic_aft"}
    assert result.metadata["selected_survival_model"] == result.model_type
    assert result.metadata["candidate_survival_model_count"] == 4
    assert any(item["status"] == "selected" for item in result.metadata["candidate_survival_models"])
    assert "Parametric survival selection compared 4 AFT candidates" in report.narrative
    assert report.metadata["selected_survival_model"] == result.model_type
    assert visual.output_files
    assert audit.metadata["selected_survival_model"] == result.model_type


def test_selector_routes_parametric_survival_auto() -> None:
    result = fit_regression_by_level(
        _survival_data(),
        dependent_variable="duration",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="parametric_survival_auto",
        mixed_effects_options={"event_variable": "event", "selection_criterion": "bic"},
    )

    assert result.metadata["selected_survival_model"] == result.model_type
    assert result.metadata["survival_selection_criterion"] == "bic"


def test_parametric_survival_diagnostics_pipeline_uses_selected_model(tmp_path: Path) -> None:
    result = fit_parametric_survival_regression(
        _survival_data(),
        duration_variable="duration",
        event_variable="event",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    runtime = PipelineRuntime(dataframe=_survival_data())
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="parametric survival diagnostics"),
        tmp_path,
    )

    assert step_result.success is True
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == result.model_type


def test_builder_registers_parametric_survival_auto_pipeline(tmp_path: Path) -> None:
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
                        "estimator": "parametric_survival",
                        "event_variable": "event",
                        "selection_criterion": "aic",
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
        context=ResearchContext(project_name="parametric survival builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "parametric_survival_auto"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
