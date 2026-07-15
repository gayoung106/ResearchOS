"""회귀 시각화 엔진을 파이프라인에 연결한다."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.visualization.regression import (
    build_regression_visualizations,
)


class RegressionVisualizationStep(PipelineStep):
    """저장된 회귀결과의 논문용 시각자료를 생성한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        order: int = 150,
    ) -> None:
        super().__init__(
            name="15_regression_visualization",
            order=order,
            required=False,
        )
        self.runtime = runtime
        self.model_id = model_id

    def should_run(
        self,
        context: ResearchContext,
    ) -> bool:
        return f"regression_result:{self.model_id}" in self.runtime.artifacts

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        regression_result = self.runtime.get_artifact(f"regression_result:{self.model_id}")

        output_dir = working_directory / "result" / "15_visualization" / self.model_id

        report = build_regression_visualizations(
            regression_result,
            output_directory=output_dir,
        )
        self.runtime.set_artifact(
            f"regression_visualization:{self.model_id}",
            report,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=report.output_files,
            warnings=report.warnings,
            metadata={
                "model_id": report.model_id,
                "model_type": report.model_type,
                **report.metadata,
            },
        )
