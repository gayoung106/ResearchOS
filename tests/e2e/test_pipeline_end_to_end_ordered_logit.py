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
    # ------------------------------------------------------------------
    # rawdata 준비
    # ------------------------------------------------------------------
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()

    fixture_dir = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "data"
    )

    shutil.copy(
        fixture_dir / "ordered_logit_sample.csv",
        rawdata_dir / "ordered_logit_sample.csv",
    )

    # ------------------------------------------------------------------
    # Pipeline 생성
    # ------------------------------------------------------------------
    context = ResearchContext(
        project_name="Ordered Logit E2E",
    )

    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=ordinal_variable_map,
        working_directory=tmp_path,
    )

    # ------------------------------------------------------------------
    # Registry 검증
    # ------------------------------------------------------------------
    assert_registry_matches(
        orchestrator,
        full_ordered_logit_pipeline(),
    )

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------
    result = orchestrator.run()

    assert result.success is True
    assert result.failed_stage is None

    assert runtime.dataframe is not None
    assert len(runtime.dataframe) == 30
