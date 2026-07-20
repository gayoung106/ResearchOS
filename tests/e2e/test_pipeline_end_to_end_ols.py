from __future__ import annotations

import shutil
from pathlib import Path

from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from tests.support.assertions import assert_registry_matches
from tests.support.expected_pipeline import full_ols_pipeline


def test_pipeline_end_to_end_ols(
    tmp_path: Path,
    ols_with_robustness_analysis_plan,
    continuous_variable_map,
) -> None:
    # ------------------------------------------------------------------
    # rawdata 디렉터리 생성
    # ------------------------------------------------------------------
    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()

    fixture_dir = Path(__file__).resolve().parent.parent / "fixtures" / "data"

    shutil.copy(
        fixture_dir / "ols_sample.csv",
        rawdata_dir / "ols_sample.csv",
    )

    # ------------------------------------------------------------------
    # Pipeline 생성
    # ------------------------------------------------------------------
    context = ResearchContext(
        project_name="OLS E2E",
    )

    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=continuous_variable_map,
        working_directory=tmp_path,
    )

    # ------------------------------------------------------------------
    # Registry 검증
    # ------------------------------------------------------------------
    assert_registry_matches(
        orchestrator,
        full_ols_pipeline(
            robustness=True,
            advanced_robustness=True,
        ),
    )

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------
    result = orchestrator.run()

    assert result.success is True
    assert result.failed_stage is None

    assert runtime.dataframe is not None
    assert len(runtime.dataframe) == 20
