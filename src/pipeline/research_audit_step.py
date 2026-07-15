"""Research Audit Engine을 파이프라인에 연결한다."""

from __future__ import annotations

from pathlib import Path

from src.audit.research import (
    audit_items_to_dataframe,
    audit_summary_to_dataframe,
    build_research_audit_report,
    write_audit_narrative,
)
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult


class ResearchAuditStep(PipelineStep):
    """현재 연구 파이프라인 산출물을 감사한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str = "main_model",
        order: int = 160,
    ) -> None:
        super().__init__(
            name="16_research_audit",
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
        report = build_research_audit_report(
            self.runtime,
            model_id=self.model_id,
        )
        self.runtime.set_artifact(
            f"research_audit:{self.model_id}",
            report,
        )

        output_dir = working_directory / "result" / "16_audit" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        items_path = output_dir / "audit_items.xlsx"
        summary_path = output_dir / "audit_summary.xlsx"
        narrative_path = output_dir / "audit_report_ko.txt"

        audit_items_to_dataframe(report).to_excel(
            items_path,
            index=False,
        )
        audit_summary_to_dataframe(report).to_excel(
            summary_path,
            index=False,
        )
        narrative_path.write_text(
            write_audit_narrative(report) + "\n",
            encoding="utf-8",
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(items_path),
                str(summary_path),
                str(narrative_path),
            ],
            warnings=report.warnings,
            metadata={
                "model_id": report.model_id,
                "percentage": report.percentage,
                "grade": report.grade,
                "submission_status": report.submission_status,
            },
        )
