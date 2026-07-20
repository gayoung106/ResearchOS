"""Random Intercept 혼합효과모형 Research Audit 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.research import build_research_audit_report
from src.pipeline.context import ResearchContext
from src.pipeline.research_audit_step import ResearchAuditStep
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import build_regression_publication_report
from src.statistics.diagnostics.mixed_effects import build_mixed_effects_diagnostics
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_effects import fit_random_intercept
from src.visualization.regression import build_regression_visualizations


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    groups = np.repeat(np.arange(12), 8)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(scale=0.9, size=12)
    y = (
        1.2
        + 1.7 * x
        + random_intercepts[groups]
        + rng.normal(
            scale=0.45,
            size=len(groups),
        )
    )
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def make_runtime(tmp_path: Path) -> PipelineRuntime:
    dataframe = make_dataframe()
    regression = fit_random_intercept(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        model_id="main_model",
        method="lbfgs",
        max_iterations=200,
    )
    diagnostics = build_mixed_effects_diagnostics(regression)
    effects = build_regression_effect_size_report(regression)
    publication = build_regression_publication_report(regression, effects)
    visualization = build_regression_visualizations(
        regression,
        output_directory=tmp_path / "figures",
    )

    runtime = PipelineRuntime(dataframe=dataframe)
    runtime.set_artifact("regression_result:main_model", regression)
    runtime.set_artifact("regression_diagnostics:main_model", diagnostics)
    runtime.set_artifact("effect_size_report:main_model", effects)
    runtime.set_artifact(
        "regression_publication_report:main_model",
        publication,
    )
    runtime.set_artifact(
        "regression_visualization:main_model",
        visualization,
    )
    runtime.missingness_report = object()
    runtime.outlier_report = object()
    return runtime


def test_mixed_effects_audit_uses_model_specific_evidence(tmp_path: Path) -> None:
    report = build_research_audit_report(
        make_runtime(tmp_path),
        model_id="main_model",
    )

    items = {item.item: item for item in report.items}

    assert report.metadata["model_type"] == "mixed_random_intercept"
    assert report.metadata["group_variable"] == "group"
    assert report.metadata["group_count"] == 12
    assert report.metadata["intraclass_correlation"] > 0
    assert report.metadata["not_applicable_item_count"] == 1

    assert "Random Intercept" in items["회귀모형 추정"].evidence
    assert "그룹 수=12" in items["회귀모형 추정"].evidence
    assert "혼합효과 진단" in items["회귀진단"].evidence
    assert items["강건성 분석"].status == "NOT_APPLICABLE"
    assert items["강건성 분석"].maximum_score == 0
    assert "marginal_r_squared" in items["효과크기"].evidence
    assert "conditional_r_squared" in items["효과크기"].evidence


def test_mixed_effects_audit_step_outputs_files(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)

    result = ResearchAuditStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="Mixed Audit"),
        tmp_path,
    )

    report = runtime.get_artifact("research_audit:main_model")

    assert result.success is True
    assert len(result.output_files) == 3
    assert all(Path(path).exists() for path in result.output_files)
    assert report.metadata["model_type"] == "mixed_random_intercept"
    assert report.metadata["group_count"] == 12
