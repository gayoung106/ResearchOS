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
from src.statistics.diagnostics.multinomial_logit import (
    build_multinomial_logit_diagnostics,
    multinomial_confusion_matrix_to_dataframe,
    multinomial_diagnostic_summary_to_dataframe,
    multinomial_predictions_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.multinomial_logit import fit_multinomial_logit
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _nominal_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260801)
    n = 150
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    logits = np.column_stack(
        [
            np.zeros(n),
            -0.25 + 0.9 * x - 0.35 * z,
            0.15 - 0.5 * x + 0.75 * z,
        ]
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    categories = np.array(["control", "choice_a", "choice_b"])
    y = [rng.choice(categories, p=row) for row in probabilities]
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_multinomial_logit_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _nominal_data()
    result = fit_multinomial_logit(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "multinomial_logit"
    assert result.fit_statistics["category_count"] == 3
    assert result.metadata["reference_category"] in result.metadata["category_labels"]
    assert any("::x" in coefficient.term for coefficient in result.coefficients)
    assert any(effect.effect_type == "odds_ratio" for effect in effects.effects)
    assert "reference category" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {"coefficient_forest.png"}
    assert audit.metadata["category_count"] == 3


def test_selector_routes_nominal_to_multinomial_logit() -> None:
    result = fit_regression_by_level(
        _nominal_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="nominal",
    )

    assert result.model_type == "multinomial_logit"
    assert result.fit_statistics["category_count"] == 3


def test_multinomial_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _nominal_data()
    result = fit_multinomial_logit(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_multinomial_logit_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="multinomial diagnostics"),
        tmp_path,
    )

    assert diagnostics.category_count == 3
    assert multinomial_predictions_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert multinomial_confusion_matrix_to_dataframe(diagnostics).shape[0] == 3
    assert "mean_log_loss" in set(multinomial_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 5
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"


def test_builder_registers_nominal_multinomial_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="nominal"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="multinomial builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "multinomial_logit"
    assert registration.measurement_level == "nominal"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
