"""기술통계 엔진을 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.descriptive import (
    build_descriptive_report,
    dataset_summary_to_dataframe,
    quality_warnings_to_dataframe,
)


class DescriptiveStatisticsStep(PipelineStep):
    """기술통계와 데이터 품질 리포트를 생성한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        order: int = 70,
    ) -> None:
        super().__init__(
            name="07_descriptive_statistics",
            order=order,
            required=True,
        )
        self.runtime = runtime

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        report = build_descriptive_report(dataframe)
        self.runtime.set_artifact("descriptive_report", report)

        output_dir = working_directory / "result" / "07_descriptive"
        output_dir.mkdir(parents=True, exist_ok=True)

        numeric_path = output_dir / "numeric_descriptive.xlsx"
        categorical_path = output_dir / "categorical_frequencies.xlsx"
        quality_path = output_dir / "data_quality_warnings.xlsx"
        dataset_path = output_dir / "dataset_summary.xlsx"

        report.numeric_summary.to_excel(
            numeric_path,
            index=False,
        )
        report.categorical_summary.to_excel(
            categorical_path,
            index=False,
        )
        quality_warnings_to_dataframe(report.quality_warnings).to_excel(
            quality_path,
            index=False,
        )
        dataset_summary_to_dataframe(report.dataset_summary).to_excel(
            dataset_path,
            index=False,
        )

        warnings = [
            f"{warning.variable_name}: {warning.message}"
            for warning in report.quality_warnings
            if warning.severity == "high"
        ]

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(numeric_path),
                str(categorical_path),
                str(quality_path),
                str(dataset_path),
            ],
            warnings=warnings,
            metadata=report.dataset_summary,
        )
