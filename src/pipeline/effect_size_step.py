"""효과크기 엔진을 회귀 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.effects.regression import (
    build_regression_effect_size_report,
    effect_size_report_to_dataframe,
    effect_size_summary_to_dataframe,
)


class RegressionEffectSizeStep(PipelineStep):
    """저장된 회귀결과의 효과크기를 계산한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        order: int = 130,
    ) -> None:
        super().__init__(
            name="13_effect_size_analysis",
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
        regression_result = self.runtime.get_artifact(f"regression_result:{self.model_id}")

        report = build_regression_effect_size_report(regression_result)
        self.runtime.set_artifact(
            f"effect_size_report:{self.model_id}",
            report,
        )

        output_dir = working_directory / "result" / "13_effect_sizes" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        effects_path = output_dir / "effect_sizes.xlsx"
        summary_path = output_dir / "effect_size_summary.xlsx"

        effect_size_report_to_dataframe(report).to_excel(
            effects_path,
            index=False,
        )
        effect_size_summary_to_dataframe(report).to_excel(
            summary_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(effects_path),
                str(summary_path),
            ],
            warnings=report.warnings,
            metadata={
                "model_id": report.model_id,
                "model_type": report.model_type,
                "effect_count": len(report.effects),
            },
        )
