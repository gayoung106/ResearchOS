"""Quasi-binomial regression integration tests."""

from __future__ import annotations

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
from src.statistics.diagnostics.binary_logit import (
    binary_diagnostic_summary_to_dataframe,
    binary_predictions_to_dataframe,
    build_binary_logit_diagnostics,
    classification_metrics_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.quasi_binomial import fit_quasi_binomial
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _binary_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260919)
    n = 170
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    group_noise = rng.normal(0.0, 0.45, size=n)
    linear_predictor = -0.35 + 0.65 * x - 0.35 * z + group_noise
    risk = 1.0 / (1.0 + np.exp(-linear_predictor))
    y = rng.binomial(1, risk)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_quasi_binomial_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _binary_data()
    result = fit_quasi_binomial(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
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

    assert result.model_type == "quasi_binomial"
    assert result.metadata["link"] == "logit"
    assert result.metadata["family"] == "quasi_binomial"
    assert result.fit_statistics["event_count"] > 10
    assert result.fit_statistics["non_event_count"] > 10
    assert result.fit_statistics["dispersion_scale"] > 0
    assert any(effect.effect_type == "odds_ratio" for effect in effects.effects)
    assert any(effect.effect_type == "average_marginal_effect" for effect in effects.effects)
    assert "Quasi-binomial" in report.narrative
    assert any("Quasi-binomial models report" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {"coefficient_forest.png"}
    assert audit.metadata["quasi_binomial"] is True
    assert audit.metadata["dispersion_scale"] > 0


def test_selector_routes_explicit_quasi_binomial() -> None:
    result = fit_regression_by_level(
        _binary_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="binary",
        model_type="quasi_binomial",
        model_id="main_model",
    )

    assert result.model_type == "quasi_binomial"
    assert result.model_id == "main_model"
    assert result.fit_statistics["brier_score"] >= 0


def test_quasi_binomial_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _binary_data()
    result = fit_quasi_binomial(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_binary_logit_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="quasi-binomial diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "quasi_binomial"
    assert binary_predictions_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "roc_auc" in set(binary_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert "brier_score" in set(classification_metrics_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "quasi_binomial"


def test_builder_registers_explicit_quasi_binomial_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x", "z"]},
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "quasi_binomial"},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="binary"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="quasi-binomial builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "quasi_binomial"
    assert registration.measurement_level == "binary"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
