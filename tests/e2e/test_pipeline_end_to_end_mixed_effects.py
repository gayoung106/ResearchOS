from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from src.statistics.regression.base import RegressionResult
from tests.support.assertions import assert_registry_matches
from tests.support.expected_pipeline import full_mixed_effects_pipeline


def make_mixed_effects_analysis_plan() -> AnalysisPlan:
    """Random Intercept E2E 분석계획을 생성한다."""
    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["group"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {
                    "enabled": True,
                    "options": {
                        "group_variable": "group",
                        "reml": False,
                        "optimizer": "lbfgs",
                        "max_iterations": 200,
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )


def make_mixed_effects_variable_map() -> VariableMap:
    """Random Intercept E2E 변수정의를 생성한다."""
    return VariableMap.model_validate(
        {
            "variables": {
                "y": {
                    "role": "dependent",
                    "measurement_level": "continuous",
                },
                "x": {
                    "role": "independent",
                    "measurement_level": "continuous",
                },
                "group": {
                    "role": "cluster",
                    "measurement_level": "nominal",
                },
            }
        }
    )


def test_pipeline_end_to_end_mixed_effects(tmp_path: Path) -> None:
    """Random Intercept가 전체 파이프라인에서 산출물을 생성하는지 검증한다."""
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()

    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "data" / "mixed_effects_sample.csv"
    )
    shutil.copy(
        fixture_path,
        rawdata_dir / fixture_path.name,
    )

    context = ResearchContext(project_name="Mixed Effects E2E")
    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=make_mixed_effects_analysis_plan(),
        variable_map=make_mixed_effects_variable_map(),
        working_directory=tmp_path,
    )

    assert_registry_matches(
        orchestrator,
        full_mixed_effects_pipeline(),
    )

    registration = runtime.get_artifact("regression_registration")
    assert registration.registered is True
    assert registration.model_type == "mixed_random_intercept"
    assert registration.group_variable == "group"
    assert registration.diagnostics_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True

    pipeline_result = orchestrator.run()

    assert pipeline_result.success is True
    assert pipeline_result.failed_stage is None
    assert "09_regression_analysis" in pipeline_result.completed_stages
    assert "10_regression_diagnostics" in pipeline_result.completed_stages
    assert "13_effect_size_analysis" in pipeline_result.completed_stages
    assert "14_regression_reporting" in pipeline_result.completed_stages
    assert "15_regression_visualization" in pipeline_result.completed_stages
    assert "16_research_audit" in pipeline_result.completed_stages

    model_result = runtime.get_artifact("regression_result:main_model")
    assert isinstance(model_result, RegressionResult)
    assert model_result.model_type == "mixed_random_intercept"
    assert model_result.converged is True
    assert model_result.sample_size == 192
    assert model_result.metadata["group_variable"] == "group"
    assert model_result.fit_statistics["group_count"] == 24
    assert model_result.fit_statistics["intraclass_correlation"] > 0

    x_coefficient = next(
        coefficient for coefficient in model_result.coefficients if coefficient.term == "x"
    )
    assert x_coefficient.estimate == pytest.approx(2.0, abs=0.15)

    coefficient_path = tmp_path / "result" / "09_models" / "main_model_coefficients.xlsx"
    fit_path = tmp_path / "result" / "09_models" / "main_model_fit_statistics.xlsx"

    assert coefficient_path.exists()
    assert fit_path.exists()
    assert str(coefficient_path) in context.generated_files
    assert str(fit_path) in context.generated_files

    coefficient_table = pd.read_excel(coefficient_path)
    fit_table = pd.read_excel(fit_path)

    assert "x" in coefficient_table["term"].tolist()
    assert "random_intercept_variance" in fit_table["item"].tolist()
    assert "intraclass_correlation" in fit_table["item"].tolist()

    diagnostics = runtime.get_artifact("regression_diagnostics:main_model")
    assert diagnostics.model_id == "main_model"
    assert diagnostics.sample_size == 192
    assert diagnostics.group_count == 24
    assert diagnostics.summary["converged"] is True

    diagnostics_dir = tmp_path / "result" / "10_diagnostics" / "main_model"
    diagnostic_paths = [
        diagnostics_dir / "diagnostic_tests.xlsx",
        diagnostics_dir / "residuals.xlsx",
        diagnostics_dir / "group_residuals.xlsx",
        diagnostics_dir / "random_effects.xlsx",
        diagnostics_dir / "diagnostic_summary.xlsx",
    ]

    assert all(path.exists() for path in diagnostic_paths)
    assert all(str(path) in context.generated_files for path in diagnostic_paths)

    residuals_table = pd.read_excel(diagnostics_dir / "residuals.xlsx")
    random_effects_table = pd.read_excel(diagnostics_dir / "random_effects.xlsx")
    summary_table = pd.read_excel(diagnostics_dir / "diagnostic_summary.xlsx")

    assert len(residuals_table) == 192
    assert len(random_effects_table) == 24
    assert "standardized_residual" in residuals_table.columns
    assert "random_intercept" in random_effects_table.columns
    assert "intraclass_correlation" in summary_table["item"].tolist()

    effect_size_report = runtime.get_artifact("effect_size_report:main_model")
    assert effect_size_report.model_type == "mixed_random_intercept"
    assert effect_size_report.metadata["group_variable"] == "group"
    assert effect_size_report.metadata["group_count"] == 24
    assert effect_size_report.model_effects["marginal_r_squared"] > 0
    assert (
        effect_size_report.model_effects["conditional_r_squared"]
        >= effect_size_report.model_effects["marginal_r_squared"]
    )
    assert effect_size_report.model_effects["intraclass_correlation"] > 0

    effects_dir = tmp_path / "result" / "13_effect_sizes" / "main_model"
    effects_path = effects_dir / "effect_sizes.xlsx"
    effects_summary_path = effects_dir / "effect_size_summary.xlsx"

    assert effects_path.exists()
    assert effects_summary_path.exists()
    assert str(effects_path) in context.generated_files
    assert str(effects_summary_path) in context.generated_files

    # Fasoo DRM 환경에서는 생성 직후 Excel 파일이 암호화될 수 있으므로,
    # E2E에서는 재열기 대신 파일 생성과 비어 있지 않은지를 검증한다.
    assert effects_path.stat().st_size > 0
    assert effects_summary_path.stat().st_size > 0

    assert "intraclass_correlation" in effect_size_report.model_effects
    assert "marginal_r_squared" in effect_size_report.model_effects
    assert "conditional_r_squared" in effect_size_report.model_effects

    publication_report = runtime.get_artifact("regression_publication_report:main_model")
    assert publication_report.model_type == "mixed_random_intercept"
    assert publication_report.metadata["group_variable"] == "group"
    assert publication_report.metadata["group_count"] == 24
    assert "Random Intercept 혼합효과모형" in publication_report.narrative
    assert "ICC=" in publication_report.narrative
    assert "marginal R²=" in publication_report.narrative
    assert "conditional R²=" in publication_report.narrative

    reporting_dir = tmp_path / "result" / "14_reporting" / "main_model"
    reporting_paths = [
        reporting_dir / "regression_publication_table.xlsx",
        reporting_dir / "model_summary.xlsx",
        reporting_dir / "results_narrative_ko.txt",
        reporting_dir / "table_notes_ko.txt",
    ]

    assert all(path.exists() for path in reporting_paths)
    assert all(path.stat().st_size > 0 for path in reporting_paths)
    assert all(str(path) in context.generated_files for path in reporting_paths)

    narrative_text = (reporting_dir / "results_narrative_ko.txt").read_text(encoding="utf-8")
    notes_text = (reporting_dir / "table_notes_ko.txt").read_text(encoding="utf-8")
    assert "Random Intercept 혼합효과모형" in narrative_text
    assert "marginal R²=" in narrative_text
    assert "혼합효과모형의 고정효과" in notes_text

    visualization_report = runtime.get_artifact("regression_visualization:main_model")
    assert visualization_report.model_type == "mixed_random_intercept"
    assert visualization_report.metadata["figure_count"] == 4
    assert visualization_report.warnings == []

    visualization_dir = tmp_path / "result" / "15_visualization" / "main_model"
    visualization_paths = [
        visualization_dir / "coefficient_forest.png",
        visualization_dir / "residuals_vs_fitted.png",
        visualization_dir / "residual_qq_plot.png",
        visualization_dir / "random_intercepts.png",
    ]

    assert all(path.exists() for path in visualization_paths)
    assert all(path.stat().st_size > 0 for path in visualization_paths)
    assert all(str(path) in context.generated_files for path in visualization_paths)

    audit_report = runtime.get_artifact("research_audit:main_model")
    assert audit_report.metadata["model_type"] == "mixed_random_intercept"
    assert audit_report.metadata["group_variable"] == "group"
    assert audit_report.metadata["group_count"] == 24
    assert audit_report.metadata["intraclass_correlation"] > 0
    assert audit_report.metadata["not_applicable_item_count"] == 1

    audit_items = {item.item: item for item in audit_report.items}
    assert "Random Intercept" in audit_items["회귀모형 추정"].evidence
    assert audit_items["강건성 분석"].status == "NOT_APPLICABLE"
    assert audit_items["강건성 분석"].maximum_score == 0

    audit_dir = tmp_path / "result" / "16_audit" / "main_model"
    audit_paths = [
        audit_dir / "audit_items.xlsx",
        audit_dir / "audit_summary.xlsx",
        audit_dir / "audit_report_ko.txt",
    ]
    assert all(path.exists() for path in audit_paths)
    assert all(path.stat().st_size > 0 for path in audit_paths)
    assert all(str(path) in context.generated_files for path in audit_paths)

    audit_narrative = (audit_dir / "audit_report_ko.txt").read_text(encoding="utf-8")
    assert "연구 품질 감사 결과" in audit_narrative
