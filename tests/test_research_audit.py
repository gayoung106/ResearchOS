"""Research Audit Engine 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.audit.research import (
    build_research_audit_report,
    write_audit_narrative,
)
from src.pipeline.context import ResearchContext
from src.pipeline.research_audit_step import ResearchAuditStep
from src.pipeline.runtime import PipelineRuntime
from src.reporting.regression import (
    build_regression_publication_report,
)
from src.statistics.effects.regression import (
    build_regression_effect_size_report,
)
from src.statistics.regression.ols import fit_ols
from src.visualization.regression import (
    RegressionVisualizationReport,
)


def make_runtime() -> PipelineRuntime:
    rng = np.random.default_rng(50)
    x = rng.normal(size=180)
    y = 1 + 1.5 * x + rng.normal(size=180)
    dataframe = pd.DataFrame({"y": y, "x": x})

    regression = fit_ols(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        model_id="main_model",
    )
    effects = build_regression_effect_size_report(regression)
    publication = build_regression_publication_report(
        regression,
        effects,
    )

    runtime = PipelineRuntime(
        dataframe=dataframe,
    )
    runtime.set_artifact(
        "regression_result:main_model",
        regression,
    )
    runtime.set_artifact(
        "effect_size_report:main_model",
        effects,
    )
    runtime.set_artifact(
        "regression_publication_report:main_model",
        publication,
    )
    runtime.set_artifact(
        "regression_visualization:main_model",
        RegressionVisualizationReport(
            model_id="main_model",
            model_type="ols",
            output_files=["figure.png"],
            metadata={"figure_count": 1},
        ),
    )

    runtime.missingness_report = object()
    runtime.outlier_report = object()

    return runtime


def test_audit_report_scores_available_outputs() -> None:
    report = build_research_audit_report(
        make_runtime(),
        model_id="main_model",
    )

    assert report.total_score > 0
    assert report.maximum_score > report.total_score
    assert 0 <= report.percentage <= 100
    assert report.grade in {
        "A",
        "B",
        "C",
        "D",
        "F",
    }


def test_missing_outputs_lower_score() -> None:
    runtime = PipelineRuntime()

    report = build_research_audit_report(
        runtime,
        model_id="main_model",
    )

    assert report.percentage < 50
    assert report.warnings


def test_audit_narrative_is_korean() -> None:
    report = build_research_audit_report(
        make_runtime(),
        model_id="main_model",
    )
    narrative = write_audit_narrative(report)

    assert "연구 품질 감사 결과" in narrative
    assert "종합 판정" in narrative


def test_audit_pipeline_step_outputs_files(
    tmp_path: Path,
) -> None:
    runtime = make_runtime()

    result = ResearchAuditStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert result.success is True
    assert len(result.output_files) == 3
    assert all(Path(path).exists() for path in result.output_files)
    assert runtime.get_artifact("research_audit:main_model").percentage >= 0
