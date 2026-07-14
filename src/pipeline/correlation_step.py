"""상관분석 엔진을 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.correlation import (
    correlation_results_to_dataframe,
    run_correlation_analysis,
)


class CorrelationAnalysisStep(PipelineStep):
    """설정된 변수의 상관분석을 수행한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        variables: list[str],
        *,
        method: str = "pearson",
        p_adjust_method: str = "holm",
        order: int = 80,
    ) -> None:
        super().__init__(
            name="08_correlation_analysis",
            order=order,
            required=False,
        )
        self.runtime = runtime
        self.variables = variables
        self.method = method
        self.p_adjust_method = p_adjust_method

    def should_run(self, context: ResearchContext) -> bool:
        return len(self.variables) >= 2

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        report = run_correlation_analysis(
            dataframe,
            self.variables,
            method=self.method,
            p_adjust_method=self.p_adjust_method,
        )
        self.runtime.set_artifact(
            "correlation_report",
            report,
        )

        output_dir = working_directory / "result" / "08_correlation"
        output_dir.mkdir(parents=True, exist_ok=True)

        pairwise_path = output_dir / "correlation_results.xlsx"
        coefficient_path = output_dir / "correlation_matrix.xlsx"
        p_value_path = output_dir / "adjusted_p_values.xlsx"
        sample_size_path = output_dir / "pairwise_sample_sizes.xlsx"
        publication_path = output_dir / "publication_correlation_table.xlsx"

        correlation_results_to_dataframe(report).to_excel(
            pairwise_path,
            index=False,
        )
        report.coefficient_matrix.to_excel(coefficient_path)
        report.p_value_matrix.to_excel(p_value_path)
        report.sample_size_matrix.to_excel(sample_size_path)
        report.publication_table.to_excel(
            publication_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(pairwise_path),
                str(coefficient_path),
                str(p_value_path),
                str(sample_size_path),
                str(publication_path),
            ],
            warnings=report.warnings,
            metadata={
                "method": self.method,
                "variable_count": len(self.variables),
                "pair_count": len(report.results),
            },
        )
