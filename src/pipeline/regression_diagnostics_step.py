"""OLS 회귀진단 엔진을 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.diagnostics.ols import (
    build_ols_diagnostics,
    diagnostic_summary_to_dataframe,
    influence_to_dataframe,
    multicollinearity_to_dataframe,
    residuals_to_dataframe,
    tests_to_dataframe,
)


class RegressionDiagnosticsStep(PipelineStep):
    """저장된 OLS 회귀결과를 대상으로 진단을 실행한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        order: int = 100,
    ) -> None:
        super().__init__(
            name="10_regression_diagnostics",
            order=order,
            required=False,
        )
        self.runtime = runtime
        self.model_id = model_id

    def should_run(self, context: ResearchContext) -> bool:
        return f"regression_result:{self.model_id}" in self.runtime.artifacts

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        result = self.runtime.get_artifact(f"regression_result:{self.model_id}")

        if result.model_type != "ols":
            return StepResult(
                stage_name=self.name,
                success=True,
                warnings=["현재 진단 단계는 OLS 모형만 지원하므로 생략했습니다."],
                metadata={"model_id": self.model_id, "skipped": True},
            )

        report = build_ols_diagnostics(result)
        self.runtime.set_artifact(
            f"regression_diagnostics:{self.model_id}",
            report,
        )

        output_dir = working_directory / "result" / "10_diagnostics" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        vif_path = output_dir / "multicollinearity.xlsx"
        tests_path = output_dir / "diagnostic_tests.xlsx"
        residuals_path = output_dir / "residuals.xlsx"
        influence_path = output_dir / "influence.xlsx"
        summary_path = output_dir / "diagnostic_summary.xlsx"

        multicollinearity_to_dataframe(report).to_excel(vif_path, index=False)
        tests_to_dataframe(report).to_excel(tests_path, index=False)
        residuals_to_dataframe(report).to_excel(residuals_path, index=False)
        influence_to_dataframe(report).to_excel(influence_path, index=False)
        diagnostic_summary_to_dataframe(report).to_excel(
            summary_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(vif_path),
                str(tests_path),
                str(residuals_path),
                str(influence_path),
                str(summary_path),
            ],
            warnings=report.warnings,
            metadata=report.summary,
        )
