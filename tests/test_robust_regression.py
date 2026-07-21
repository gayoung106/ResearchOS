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
from src.statistics.diagnostics.robust import (
    build_robust_diagnostics,
    robust_diagnostic_summary_to_dataframe,
    robust_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.robust import fit_robust_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _robust_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260810)
    n = 120
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1.0 + 0.75 * x - 0.35 * z + rng.normal(0.0, 0.35, size=n)
    y[:6] += np.array([6.0, -5.5, 5.2, -6.4, 4.8, -5.0])
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_fit_robust_regression_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _robust_data()
    result = fit_robust_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        norm="huber",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "robust_regression"
    assert result.fit_statistics["downweighted_count"] > 0
    assert result.metadata["norm"] == "huber"
    assert any(effect.effect_type == "robust_standardized_beta" for effect in effects.effects)
    assert "Robust regression used huber M-estimation weights." in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["downweighted_count"] == result.fit_statistics["downweighted_count"]


def test_selector_routes_explicit_robust_regression() -> None:
    result = fit_regression_by_level(
        _robust_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="robust_regression",
        mixed_effects_options={"norm": "tukey"},
    )

    assert result.model_type == "robust_regression"
    assert result.metadata["norm"] == "tukey"


def test_robust_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _robust_data()
    result = fit_robust_regression(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        model_id="main_model",
    )
    diagnostics = build_robust_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="robust diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.sample_size == result.sample_size
    assert robust_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "downweighted_count" in set(robust_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "robust_regression"
    assert any("Robust diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_robust_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "robust", "norm": "huber"},
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="continuous"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "z": VariableDefinition(role="independent", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="robust builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "robust_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
