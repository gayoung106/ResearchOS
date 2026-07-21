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
from src.statistics.diagnostics.iv import (
    build_iv_2sls_diagnostics,
    iv_diagnostic_summary_to_dataframe,
    iv_first_stage_to_dataframe,
    iv_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.iv import fit_iv_2sls_regression
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _iv_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260814)
    n = 160
    z = rng.normal(size=n)
    control = rng.normal(size=n)
    u = rng.normal(size=n)
    endogenous = 0.9 * z + 0.35 * control + 0.65 * u + rng.normal(0.0, 0.25, size=n)
    y = 1.0 + 1.25 * endogenous + 0.45 * control + u + rng.normal(0.0, 0.25, size=n)
    return pd.DataFrame({"y": y, "endog": endogenous, "control": control, "instrument": z})


def test_fit_iv_2sls_integrates_reporting_visualization_and_audit(tmp_path: Path) -> None:
    data = _iv_data()
    result = fit_iv_2sls_regression(
        data,
        dependent_variable="y",
        independent_variables=["control", "endog"],
        endogenous_variables=["endog"],
        instrument_variables=["instrument"],
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "iv_2sls_regression"
    assert result.fit_statistics["minimum_first_stage_f_statistic"] > 10
    assert result.metadata["endogenous_variables"] == ["endog"]
    assert result.metadata["instrument_variables"] == ["instrument"]
    assert any(effect.effect_type == "iv_standardized_beta" for effect in effects.effects)
    assert "IV 2SLS treated endog as endogenous" in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["instrument_count"] == 1


def test_selector_routes_explicit_iv_2sls_regression() -> None:
    result = fit_regression_by_level(
        _iv_data(),
        dependent_variable="y",
        independent_variables=["control", "endog"],
        measurement_level="continuous",
        model_type="iv_2sls_regression",
        mixed_effects_options={
            "endogenous_variables": ["endog"],
            "instrument_variables": ["instrument"],
        },
    )

    assert result.model_type == "iv_2sls_regression"
    assert result.fit_statistics["instrument_count"] == 1


def test_iv_2sls_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _iv_data()
    result = fit_iv_2sls_regression(
        data,
        dependent_variable="y",
        independent_variables=["control", "endog"],
        endogenous_variables=["endog"],
        instrument_variables=["instrument"],
        model_id="main_model",
    )
    diagnostics = build_iv_2sls_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="iv diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.sample_size == result.sample_size
    assert iv_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert iv_first_stage_to_dataframe(diagnostics).shape[0] == 1
    assert "minimum_first_stage_f_statistic" in set(iv_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "iv_2sls_regression"
    assert any("IV diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_iv_2sls_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["control", "endog"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {
                        "estimator": "iv_2sls",
                        "endogenous_variables": ["endog"],
                        "instrument_variables": ["instrument"],
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="continuous"),
            "control": VariableDefinition(role="independent", measurement_level="continuous"),
            "endog": VariableDefinition(role="independent", measurement_level="continuous"),
            "instrument": VariableDefinition(role="other", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="iv builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "iv_2sls_regression"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
