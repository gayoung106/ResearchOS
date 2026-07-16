from __future__ import annotations

import shutil
from pathlib import Path

from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from tests.support.assertions import assert_registry_matches
from tests.support.expected_pipeline import full_ordered_logit_pipeline


def test_pipeline_end_to_end_ordered_logit(
    tmp_path: Path,
    ols_with_robustness_analysis_plan,
    ordinal_variable_map,
) -> None:
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()

    fixture_dir = Path(__file__).resolve().parent.parent / "fixtures" / "data"

    shutil.copy(
        fixture_dir / "ordered_logit_sample.csv",
        rawdata_dir / "ordered_logit_sample.csv",
    )

    context = ResearchContext(
        project_name="Ordered Logit E2E",
    )

    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=ordinal_variable_map,
        working_directory=tmp_path,
    )

    assert_registry_matches(
        orchestrator,
        full_ordered_logit_pipeline(),
    )

    result = orchestrator.run()

    assert result.success is True
    assert result.failed_stage is None
    assert runtime.dataframe is not None
    assert len(runtime.dataframe) == 30

    registered = orchestrator.registry.names()

    assert "09_regression_analysis" in registered
    assert "10_regression_diagnostics" in registered
    assert "11_robustness_analysis" not in registered
    assert "12_advanced_robustness" not in registered
    assert "13_effect_size_analysis" in registered
    assert "14_regression_reporting" in registered
    assert "15_regression_visualization" in registered
    assert "16_research_audit" in registered

    diagnostics = runtime.get_artifact(
        "regression_diagnostics:main_model",
    )

    assert diagnostics.model_id == "main_model"
    assert diagnostics.sample_size == 30
    assert diagnostics.category_count >= 3
