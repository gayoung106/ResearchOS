"""Regression Core를 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.regression.base import (
    coefficients_to_dataframe,
    fit_statistics_to_dataframe,
)
from src.statistics.regression.selector import (
    fit_regression_by_level,
)


class RegressionAnalysisStep(PipelineStep):
    """설정된 회귀모형 하나를 실행한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        dependent_variable: str,
        independent_variables: list[str],
        measurement_level: str,
        fixed_effects: list[str] | None = None,
        model_id: str = "model_1",
        model_type: str | None = None,
        group_variable: str | None = None,
        mixed_effects_options: dict[str, object] | None = None,
        order: int = 90,
    ) -> None:
        super().__init__(
            name="09_regression_analysis",
            order=order,
            required=False,
        )
        self.runtime = runtime
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables
        self.measurement_level = measurement_level
        self.fixed_effects = fixed_effects or []
        self.model_id = model_id
        self.model_type = model_type
        self.group_variable = group_variable
        self.mixed_effects_options = mixed_effects_options or {}

    def should_run(
        self,
        context: ResearchContext,
    ) -> bool:
        return bool(self.dependent_variable and self.independent_variables)

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()

        result = fit_regression_by_level(
            dataframe,
            dependent_variable=self.dependent_variable,
            independent_variables=self.independent_variables,
            measurement_level=self.measurement_level,
            fixed_effects=self.fixed_effects,
            model_id=self.model_id,
            model_type=self.model_type,
            group_variable=self.group_variable,
            mixed_effects_options=self.mixed_effects_options,
        )

        self.runtime.set_artifact(
            f"regression_result:{self.model_id}",
            result,
        )

        output_dir = working_directory / "result" / "09_models"
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        coefficient_path = output_dir / f"{self.model_id}_coefficients.xlsx"
        fit_path = output_dir / f"{self.model_id}_fit_statistics.xlsx"

        coefficients_to_dataframe(result).to_excel(
            coefficient_path,
            index=False,
        )

        fit_statistics_to_dataframe(result).to_excel(
            fit_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=result.converged,
            output_files=[
                str(coefficient_path),
                str(fit_path),
            ],
            warnings=result.warnings,
            metadata={
                "model_id": result.model_id,
                "model_type": result.model_type,
                "sample_size": result.sample_size,
                "fixed_effects": self.fixed_effects,
                "group_variable": self.group_variable,
                "error_message": (None if result.converged else "회귀모형이 수렴하지 않았습니다."),
            },
        )
