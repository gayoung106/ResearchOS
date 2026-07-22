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
from src.statistics.diagnostics.panel import (
    build_panel_diagnostics,
    build_panel_hausman_diagnostic,
    panel_diagnostic_summary_to_dataframe,
    panel_entity_residuals_to_dataframe,
    panel_hausman_to_dataframe,
    panel_residuals_to_dataframe,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.panel import (
    fit_panel_between_effects,
    fit_panel_correlated_random_effects,
    fit_panel_first_difference,
    fit_panel_fixed_effects,
    fit_panel_pooled_ols,
    fit_panel_random_effects,
)
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations


def _panel_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260808)
    entity_count = 12
    time_count = 6
    rows = []
    entity_effects = rng.normal(0.0, 0.8, size=entity_count)
    time_effects = np.linspace(-0.25, 0.25, time_count)
    for entity in range(entity_count):
        base_x = rng.normal()
        for time in range(time_count):
            x = base_x + rng.normal(0.0, 0.7)
            z = rng.normal()
            y = 1.5 + 0.85 * x - 0.35 * z + entity_effects[entity] + time_effects[time]
            y += rng.normal(0.0, 0.25)
            rows.append({"entity": entity, "time": time, "y": y, "x": x, "z": z})
    return pd.DataFrame(rows)


def test_fit_panel_fixed_effects_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _panel_data()
    result = fit_panel_fixed_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "panel_fixed_effects"
    assert result.fit_statistics["entity_count"] == 12
    assert result.fit_statistics["time_period_count"] == 6
    assert result.metadata["absorbed_effects"] == ["entity", "time"]
    assert any(effect.effect_type == "within_standardized_beta" for effect in effects.effects)
    assert "Panel fixed effects absorbed 12 entities defined by entity." in report.narrative
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["entity_count"] == 12
    assert audit.metadata["within_r_squared"] == result.fit_statistics["within_r_squared"]


def test_selector_routes_explicit_panel_fixed_effects() -> None:
    result = fit_regression_by_level(
        _panel_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="panel_fixed_effects",
        mixed_effects_options={"entity_variable": "entity", "time_variable": "time"},
    )

    assert result.model_type == "panel_fixed_effects"
    assert result.metadata["entity_variable"] == "entity"
    assert result.metadata["time_variable"] == "time"


def test_panel_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _panel_data()
    result = fit_panel_fixed_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="main_model",
    )
    diagnostics = build_panel_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="panel diagnostics"),
        tmp_path,
    )
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert diagnostics.entity_count == 12
    assert panel_entity_residuals_to_dataframe(diagnostics).shape[0] == 12
    assert panel_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "within_r_squared" in set(panel_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_id == "main_model"
    assert any("Panel diagnostics" in item.evidence for item in audit.items)


def test_builder_registers_explicit_panel_fixed_effects_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "panel_fe"},
                },
                "panel": {
                    "enabled": True,
                    "options": {"entity_variable": "entity", "time_variable": "time"},
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
            "entity": VariableDefinition(role="id", measurement_level="nominal"),
            "time": VariableDefinition(role="time", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="panel builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_fixed_effects"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_fit_panel_random_effects_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _panel_data()
    result = fit_panel_random_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "panel_random_effects"
    assert result.fit_statistics["entity_count"] == 12
    assert result.fit_statistics["time_period_count"] == 6
    assert result.fit_statistics["random_intercept_variance"] >= 0
    assert result.fit_statistics["conditional_r_squared"] >= result.fit_statistics["marginal_r_squared"]
    assert any(effect.effect_type == "random_effects_standardized_beta" for effect in effects.effects)
    assert "Panel random effects estimated random intercepts for 12 entities defined by entity." in report.narrative
    assert any("Panel random-effects models report" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["entity_count"] == 12
    assert audit.metadata["marginal_r_squared"] == result.fit_statistics["marginal_r_squared"]


def test_selector_routes_explicit_panel_random_effects() -> None:
    result = fit_regression_by_level(
        _panel_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="panel_random_effects",
        mixed_effects_options={"entity_variable": "entity", "time_variable": "time"},
    )

    assert result.model_type == "panel_random_effects"
    assert result.metadata["entity_variable"] == "entity"
    assert result.metadata["time_variable"] == "time"


def test_panel_random_effects_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _panel_data()
    result = fit_panel_random_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="main_model",
    )
    diagnostics = build_panel_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="panel random effects diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "panel_random_effects"
    assert diagnostics.entity_count == 12
    assert "marginal_r_squared" in set(panel_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "panel_random_effects"


def test_fit_panel_correlated_random_effects_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _panel_data()
    result = fit_panel_correlated_random_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "panel_correlated_random_effects"
    assert result.fit_statistics["entity_count"] == 12
    assert result.fit_statistics["entity_mean_term_count"] == 2
    assert result.metadata["entity_mean_terms"] == ["mean_x", "mean_z"]
    assert any(effect.effect_type == "correlated_random_effects_standardized_beta" for effect in effects.effects)
    assert "Panel correlated random effects used Mundlak entity means" in report.narrative
    assert any("Mundlak" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["entity_count"] == 12
    assert audit.metadata["entity_mean_term_count"] == 2


def test_selector_routes_explicit_panel_correlated_random_effects() -> None:
    result = fit_regression_by_level(
        _panel_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="panel_correlated_random_effects",
        mixed_effects_options={"entity_variable": "entity", "time_variable": "time"},
    )

    assert result.model_type == "panel_correlated_random_effects"
    assert result.metadata["entity_variable"] == "entity"
    assert result.metadata["time_variable"] == "time"
    assert result.metadata["mundlak_correction"] is True


def test_builder_registers_explicit_panel_correlated_random_effects_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "panel_cre"},
                },
                "panel": {
                    "enabled": True,
                    "options": {"entity_variable": "entity", "time_variable": "time"},
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
            "entity": VariableDefinition(role="id", measurement_level="nominal"),
            "time": VariableDefinition(role="time", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="panel cre builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_correlated_random_effects"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_panel_hausman_diagnostic_integrates_audit() -> None:
    data = _panel_data()
    fixed = fit_panel_fixed_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="main_model",
    )
    random = fit_panel_random_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="random_model",
    )
    hausman = build_panel_hausman_diagnostic(fixed, random)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", fixed)
    runtime.set_artifact("panel_hausman:main_model", hausman)
    audit = build_research_audit_report(runtime, model_id="main_model")

    table = panel_hausman_to_dataframe(hausman)

    assert hausman.fixed_model_id == "main_model"
    assert hausman.random_model_id == "random_model"
    assert hausman.shared_terms == ["x", "z"]
    assert hausman.degrees_of_freedom == 2
    assert 0.0 <= hausman.p_value <= 1.0
    assert table.loc[0, "status"] in {"PASS", "WARNING"}
    assert any("Panel Hausman diagnostic" in item.evidence for item in audit.items)
    assert audit.metadata["panel_hausman_p_value"] == hausman.p_value


def test_builder_registers_explicit_panel_random_effects_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "panel_re"},
                },
                "panel": {
                    "enabled": True,
                    "options": {"entity_variable": "entity", "time_variable": "time"},
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
            "entity": VariableDefinition(role="id", measurement_level="nominal"),
            "time": VariableDefinition(role="time", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="panel random effects builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_random_effects"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_fit_panel_between_effects_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _panel_data()
    result = fit_panel_between_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "panel_between_effects"
    assert result.sample_size == 12
    assert result.fit_statistics["entity_count"] == 12
    assert result.fit_statistics["time_period_count"] == 6
    assert result.fit_statistics["overall_observation_count"] == len(data)
    assert any(effect.effect_type == "between_standardized_beta" for effect in effects.effects)
    assert "Panel between effects estimated entity-level mean differences for 12 entities defined by entity." in report.narrative
    assert any("Panel between-effects models report" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["entity_count"] == 12
    assert audit.metadata["between_r_squared"] == result.fit_statistics["between_r_squared"]


def test_selector_routes_explicit_panel_between_effects() -> None:
    result = fit_regression_by_level(
        _panel_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="panel_between_effects",
        mixed_effects_options={"entity_variable": "entity", "time_variable": "time"},
    )

    assert result.model_type == "panel_between_effects"
    assert result.metadata["entity_variable"] == "entity"
    assert result.metadata["time_variable"] == "time"


def test_panel_between_effects_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _panel_data()
    result = fit_panel_between_effects(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="main_model",
    )
    diagnostics = build_panel_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="panel between effects diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "panel_between_effects"
    assert diagnostics.entity_count == 12
    assert panel_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "between_r_squared" in set(panel_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "panel_between_effects"


def test_builder_registers_explicit_panel_between_effects_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "panel_be"},
                },
                "panel": {
                    "enabled": True,
                    "options": {"entity_variable": "entity", "time_variable": "time"},
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
            "entity": VariableDefinition(role="id", measurement_level="nominal"),
            "time": VariableDefinition(role="time", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="panel between effects builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_between_effects"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_fit_panel_first_difference_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _panel_data()
    result = fit_panel_first_difference(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "panel_first_difference"
    assert result.sample_size == 60
    assert result.fit_statistics["entity_count"] == 12
    assert result.fit_statistics["differenced_entity_count"] == 12
    assert result.fit_statistics["time_period_count"] == 6
    assert any(effect.effect_type == "first_difference_standardized_beta" for effect in effects.effects)
    assert "Panel first differences estimated within-entity changes for 12 entities defined by entity." in report.narrative
    assert any("Panel first-difference models report" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["entity_count"] == 12
    assert audit.metadata["first_difference_r_squared"] == result.fit_statistics["first_difference_r_squared"]


def test_selector_routes_explicit_panel_first_difference() -> None:
    result = fit_regression_by_level(
        _panel_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="panel_first_difference",
        mixed_effects_options={"entity_variable": "entity", "time_variable": "time"},
    )

    assert result.model_type == "panel_first_difference"
    assert result.metadata["entity_variable"] == "entity"
    assert result.metadata["time_variable"] == "time"


def test_panel_first_difference_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _panel_data()
    result = fit_panel_first_difference(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="main_model",
    )
    diagnostics = build_panel_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="panel first difference diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "panel_first_difference"
    assert diagnostics.entity_count == 12
    assert panel_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "first_difference_r_squared" in set(panel_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "panel_first_difference"


def test_builder_registers_explicit_panel_first_difference_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "panel_fd"},
                },
                "panel": {
                    "enabled": True,
                    "options": {"entity_variable": "entity", "time_variable": "time"},
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
            "entity": VariableDefinition(role="id", measurement_level="nominal"),
            "time": VariableDefinition(role="time", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="panel first difference builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_first_difference"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True


def test_fit_panel_pooled_ols_integrates_reporting_visualization_and_audit(
    tmp_path: Path,
) -> None:
    data = _panel_data()
    result = fit_panel_pooled_ols(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
    )
    effects = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effects)
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.model_type == "panel_pooled_ols"
    assert result.sample_size == len(data)
    assert result.fit_statistics["entity_count"] == 12
    assert result.fit_statistics["time_period_count"] == 6
    assert result.standard_error_type == "cluster_entity"
    assert any(effect.effect_type == "pooled_standardized_beta" for effect in effects.effects)
    assert "Panel pooled OLS used 12 entities defined by entity." in report.narrative
    assert any("Panel pooled OLS ignores" in note for note in report.notes)
    assert {Path(path).name for path in visual.output_files} == {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
    }
    assert audit.metadata["entity_count"] == 12
    assert audit.metadata["pooled_r_squared"] == result.fit_statistics["pooled_r_squared"]


def test_selector_routes_explicit_panel_pooled_ols() -> None:
    result = fit_regression_by_level(
        _panel_data(),
        dependent_variable="y",
        independent_variables=["x", "z"],
        measurement_level="continuous",
        model_type="panel_pooled_ols",
        mixed_effects_options={"entity_variable": "entity", "time_variable": "time"},
    )

    assert result.model_type == "panel_pooled_ols"
    assert result.metadata["entity_variable"] == "entity"
    assert result.metadata["time_variable"] == "time"


def test_panel_pooled_ols_diagnostics_and_pipeline_step(tmp_path: Path) -> None:
    data = _panel_data()
    result = fit_panel_pooled_ols(
        data,
        dependent_variable="y",
        independent_variables=["x", "z"],
        entity_variable="entity",
        time_variable="time",
        model_id="main_model",
    )
    diagnostics = build_panel_diagnostics(result)
    runtime = PipelineRuntime(dataframe=data)
    runtime.set_artifact("regression_result:main_model", result)
    step_result = RegressionDiagnosticsStep(runtime, model_id="main_model").run(
        ResearchContext(project_name="panel pooled diagnostics"),
        tmp_path,
    )

    assert diagnostics.model_type == "panel_pooled_ols"
    assert diagnostics.entity_count == 12
    assert panel_residuals_to_dataframe(diagnostics).shape[0] == result.sample_size
    assert "pooled_r_squared" in set(panel_diagnostic_summary_to_dataframe(diagnostics)["item"])
    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert runtime.get_artifact("regression_diagnostics:main_model").model_type == "panel_pooled_ols"


def test_builder_registers_explicit_panel_pooled_ols_pipeline(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x", "z"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                    "options": {"estimator": "panel_pooled"},
                },
                "panel": {
                    "enabled": True,
                    "options": {"entity_variable": "entity", "time_variable": "time"},
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
            "entity": VariableDefinition(role="id", measurement_level="nominal"),
            "time": VariableDefinition(role="time", measurement_level="continuous"),
        }
    )
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="panel pooled builder"),
        working_directory=tmp_path,
    )

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=PipelineRuntime(),
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.registered is True
    assert registration.model_type == "panel_pooled_ols"
    assert registration.measurement_level == "continuous"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
