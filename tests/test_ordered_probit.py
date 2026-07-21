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
from src.statistics.diagnostics.ordered_logit import (
    build_ordered_logit_diagnostics,
    ordered_diagnostic_summary_to_dataframe,
    ordered_predictions_to_dataframe,
    ordered_thresholds_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.ordered_probit import fit_ordered_probit
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _ordinal_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260817)
    n = 170
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    latent = 0.85 * x - 0.45 * z + rng.normal(0.0, 0.85, size=n)
    y = np.digitize(latent, [-0.75, 0.2, 0.95]) + 1
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_ordered_probit_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _ordinal_data()
    result = fit_ordered_probit(
        data,
        dependent_variable="y",
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

    assert result.model_type == "ordered_probit"
    assert result.metadata["link"] == "probit"
    assert result.fit_statistics["category_count"] == 4
    assert any(effect.effect_type == "ordered_probit_latent_coefficient" for effect in effects.effects)
    assert "Ordered probit" in report.narrative
    assert any("Ordered probit reports" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {"coefficient_forest.png"}
    assert audit.metadata["link"] == "probit"


def test_selector_routes_explicit_ordered_probit() -> None:
    result = fit_regression_by_level(
        _ordinal_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="ordinal",
        model_type="ordered_probit",
    )

    assert result.model_type == "ordered_probit"
    assert result.fit_statistics["category_count"] == 4


def test_ordered_probit_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _ordinal_data()
    result = fit_ordered_probit(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_ordered_logit_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="ordered probit diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.model_type == "ordered_probit"
    assert ordered_predictions_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert ordered_thresholds_to_dataframe(diagnostics).shape[0] == 3
    assert "ranked_probability_score" in set(ordered_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 6
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "ordered_probit"
    assert any("Ordered probit diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_ordered_probit_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "ordered_probit"},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="ordinal"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="ordered probit builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "ordered_probit"
    assert registration.measurement_level == "ordinal"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
