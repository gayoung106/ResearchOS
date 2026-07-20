"""OLS Robustness Engine을 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.robustness.ols import (
    build_ols_robustness_report,
    coefficient_comparison_to_dataframe,
    model_comparison_to_dataframe,
    robustness_summary_to_dataframe,
    stability_summary_to_dataframe,
)


class OLSRobustnessStep(PipelineStep):
    """저장된 OLS 모형 설정을 이용해 표준오차 강건성 비교를 수행한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        order: int = 110,
    ) -> None:
        super().__init__(
            name="11_robustness_analysis",
            order=order,
            required=False,
        )
        self.runtime = runtime
        self.model_id = model_id

    def should_run(self, context: ResearchContext) -> bool:
        key = f"regression_result:{self.model_id}"
        if key not in self.runtime.artifacts:
            return False

        result = self.runtime.artifacts[key]
        return result.model_type == "ols"

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        result = self.runtime.get_artifact(f"regression_result:{self.model_id}")
        dataframe = self.runtime.require_dataframe()

        report = build_ols_robustness_report(
            dataframe,
            dependent_variable=result.dependent_variable,
            independent_variables=result.independent_variables,
            model_id=self.model_id,
        )
        self.runtime.set_artifact(
            f"robustness_report:{self.model_id}",
            report,
        )

        output_dir = working_directory / "result" / "11_robustness" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        coefficient_path = output_dir / "coefficient_comparison.xlsx"
        stability_path = output_dir / "term_stability.xlsx"
        model_path = output_dir / "model_comparison.xlsx"
        summary_path = output_dir / "robustness_summary.xlsx"

        coefficient_comparison_to_dataframe(report).to_excel(
            coefficient_path,
            index=False,
        )
        stability_summary_to_dataframe(report).to_excel(
            stability_path,
            index=False,
        )
        model_comparison_to_dataframe(report).to_excel(
            model_path,
            index=False,
        )
        robustness_summary_to_dataframe(report).to_excel(
            summary_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(coefficient_path),
                str(stability_path),
                str(model_path),
                str(summary_path),
            ],
            warnings=report.warnings,
            metadata=report.summary,
        )


class MixedEffectsRobustnessStep(PipelineStep):
    """Random Intercept 모형의 optimizer 민감도 분석을 수행한다."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        optimizers: tuple[str, ...] = ("lbfgs", "bfgs", "cg", "powell"),
        order: int = 110,
    ) -> None:
        super().__init__(name="11_robustness_analysis", order=order, required=False)
        self.runtime = runtime
        self.model_id = model_id
        self.optimizers = optimizers

    def should_run(self, context: ResearchContext) -> bool:
        key = f"regression_result:{self.model_id}"
        return key in self.runtime.artifacts and self.runtime.artifacts[key].model_type in {
            "mixed_random_intercept",
            "mixed_random_slope",
            "mixed_three_level",
        }

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        from src.statistics.robustness.mixed_effects import (
            build_mixed_effects_robustness_report,
        )
        from src.statistics.robustness.mixed_effects import (
            coefficient_comparison_to_dataframe as mixed_coefficient_dataframe,
        )
        from src.statistics.robustness.mixed_effects import (
            model_comparison_to_dataframe as mixed_model_dataframe,
        )
        from src.statistics.robustness.mixed_effects import (
            robustness_summary_to_dataframe as mixed_summary_dataframe,
        )
        from src.statistics.robustness.mixed_effects import (
            stability_summary_to_dataframe as mixed_stability_dataframe,
        )

        result = self.runtime.get_artifact(f"regression_result:{self.model_id}")
        report = build_mixed_effects_robustness_report(
            self.runtime.require_dataframe(),
            dependent_variable=result.dependent_variable,
            independent_variables=result.independent_variables,
            group_variable=result.metadata["group_variable"],
            model_id=self.model_id,
            reml=bool(result.metadata.get("reml", False)),
            max_iterations=int(result.metadata.get("max_iterations", 200)),
            optimizers=self.optimizers,
            random_slope_variable=result.metadata.get("random_slope_variable"),
        )
        self.runtime.set_artifact(f"robustness_report:{self.model_id}", report)

        output_dir = working_directory / "result" / "11_robustness" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "coefficient": output_dir / "coefficient_comparison.xlsx",
            "stability": output_dir / "term_stability.xlsx",
            "model": output_dir / "model_comparison.xlsx",
            "summary": output_dir / "robustness_summary.xlsx",
        }
        mixed_coefficient_dataframe(report).to_excel(paths["coefficient"], index=False)
        mixed_stability_dataframe(report).to_excel(paths["stability"], index=False)
        mixed_model_dataframe(report).to_excel(paths["model"], index=False)
        mixed_summary_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )
