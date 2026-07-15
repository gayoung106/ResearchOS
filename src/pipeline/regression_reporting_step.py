"""논문용 회귀보고서를 파이프라인에 연결한다."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.reporting.regression import (
    build_regression_publication_report,
    model_summary_to_dataframe,
    publication_table_to_dataframe,
)


class RegressionReportingStep(PipelineStep):
    """회귀·효과크기 결과를 논문용 표와 결과문으로 변환한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        order: int = 140,
    ) -> None:
        super().__init__(
            name="14_regression_reporting",
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
        effect_key = f"effect_size_report:{self.model_id}"
        effect_report = self.runtime.artifacts.get(effect_key)

        report = build_regression_publication_report(
            regression_result,
            effect_report,
        )
        self.runtime.set_artifact(
            f"regression_publication_report:{self.model_id}",
            report,
        )

        output_dir = working_directory / "result" / "14_reporting" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        table_path = output_dir / "regression_publication_table.xlsx"
        summary_path = output_dir / "model_summary.xlsx"
        narrative_path = output_dir / "results_narrative_ko.txt"
        notes_path = output_dir / "table_notes_ko.txt"

        publication_table_to_dataframe(report).to_excel(
            table_path,
            index=False,
        )
        model_summary_to_dataframe(report).to_excel(
            summary_path,
            index=False,
        )
        narrative_path.write_text(
            report.narrative + "\n",
            encoding="utf-8",
        )
        notes_path.write_text(
            "\n".join(report.notes) + "\n",
            encoding="utf-8",
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(table_path),
                str(summary_path),
                str(narrative_path),
                str(notes_path),
            ],
            metadata={
                "model_id": report.model_id,
                "model_type": report.model_type,
                "table_row_count": len(report.publication_table),
            },
        )
