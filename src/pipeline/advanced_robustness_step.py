"""군집강건·부트스트랩·잭나이프를 파이프라인에 연결한다."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.robustness.resampling import (
    bootstrap_ols,
    fit_cluster_robust_ols,
    jackknife_ols,
    resampling_report_to_dataframe,
    resampling_summary_to_dataframe,
)


class AdvancedOLSRobustnessStep(PipelineStep):
    """OLS 고급 강건성 분석 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        cluster_variable: str | None = None,
        bootstrap_replications: int = 2000,
        run_jackknife: bool = True,
        order: int = 120,
    ) -> None:
        super().__init__(
            name="12_advanced_robustness",
            order=order,
            required=False,
        )
        self.runtime = runtime
        self.model_id = model_id
        self.cluster_variable = cluster_variable
        self.bootstrap_replications = bootstrap_replications
        self.run_jackknife = run_jackknife

    def should_run(self, context: ResearchContext) -> bool:
        key = f"regression_result:{self.model_id}"
        if key not in self.runtime.artifacts:
            return False

        return self.runtime.artifacts[key].model_type == "ols"

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        regression_result = self.runtime.get_artifact(f"regression_result:{self.model_id}")

        output_dir = working_directory / "result" / "12_advanced_robustness" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files: list[str] = []
        warnings: list[str] = []
        completed_methods: list[str] = []

        bootstrap_report = bootstrap_ols(
            dataframe,
            dependent_variable=regression_result.dependent_variable,
            independent_variables=(regression_result.independent_variables),
            model_id=self.model_id,
            replications=self.bootstrap_replications,
        )
        self.runtime.set_artifact(
            f"bootstrap_report:{self.model_id}",
            bootstrap_report,
        )

        bootstrap_result_path = output_dir / "bootstrap_coefficients.xlsx"
        bootstrap_summary_path = output_dir / "bootstrap_summary.xlsx"
        resampling_report_to_dataframe(bootstrap_report).to_excel(
            bootstrap_result_path,
            index=False,
        )
        resampling_summary_to_dataframe(bootstrap_report).to_excel(
            bootstrap_summary_path,
            index=False,
        )
        output_files.extend(
            [
                str(bootstrap_result_path),
                str(bootstrap_summary_path),
            ]
        )
        warnings.extend(bootstrap_report.warnings)
        completed_methods.append("bootstrap")

        if self.run_jackknife:
            jackknife_report = jackknife_ols(
                dataframe,
                dependent_variable=(regression_result.dependent_variable),
                independent_variables=(regression_result.independent_variables),
                model_id=self.model_id,
            )
            self.runtime.set_artifact(
                f"jackknife_report:{self.model_id}",
                jackknife_report,
            )

            jackknife_result_path = output_dir / "jackknife_coefficients.xlsx"
            jackknife_summary_path = output_dir / "jackknife_summary.xlsx"
            resampling_report_to_dataframe(jackknife_report).to_excel(
                jackknife_result_path,
                index=False,
            )
            resampling_summary_to_dataframe(jackknife_report).to_excel(
                jackknife_summary_path,
                index=False,
            )
            output_files.extend(
                [
                    str(jackknife_result_path),
                    str(jackknife_summary_path),
                ]
            )
            warnings.extend(jackknife_report.warnings)
            completed_methods.append("jackknife")

        if self.cluster_variable:
            cluster_report = fit_cluster_robust_ols(
                dataframe,
                dependent_variable=(regression_result.dependent_variable),
                independent_variables=(regression_result.independent_variables),
                cluster_variable=self.cluster_variable,
                model_id=self.model_id,
            )
            self.runtime.set_artifact(
                f"cluster_report:{self.model_id}",
                cluster_report,
            )

            cluster_result_path = output_dir / "cluster_robust_coefficients.xlsx"
            cluster_summary_path = output_dir / "cluster_robust_summary.xlsx"
            resampling_report_to_dataframe(cluster_report).to_excel(
                cluster_result_path,
                index=False,
            )
            resampling_summary_to_dataframe(cluster_report).to_excel(
                cluster_summary_path,
                index=False,
            )
            output_files.extend(
                [
                    str(cluster_result_path),
                    str(cluster_summary_path),
                ]
            )
            warnings.extend(cluster_report.warnings)
            completed_methods.append("cluster_robust")

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "model_id": self.model_id,
                "completed_methods": completed_methods,
                "bootstrap_replications": (self.bootstrap_replications),
            },
        )


class AdvancedMixedEffectsRobustnessStep(PipelineStep):
    """Random Intercept 집단 단위 고급 강건성 분석 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        model_id: str,
        bootstrap_replications: int = 500,
        run_leave_one_group_out: bool = True,
        order: int = 120,
    ) -> None:
        super().__init__(name="12_advanced_robustness", order=order, required=False)
        self.runtime = runtime
        self.model_id = model_id
        self.bootstrap_replications = bootstrap_replications
        self.run_leave_one_group_out = run_leave_one_group_out

    def should_run(self, context: ResearchContext) -> bool:
        key = f"regression_result:{self.model_id}"
        return key in self.runtime.artifacts and self.runtime.artifacts[key].model_type in {
            "mixed_random_intercept",
            "mixed_random_slope",
            "mixed_three_level",
        }

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        from src.statistics.robustness.advanced_mixed_effects import (
            build_mixed_advanced_robustness_report,
            mixed_advanced_summary_to_dataframe,
            mixed_resampling_to_dataframe,
        )

        dataframe = self.runtime.require_dataframe()
        regression_result = self.runtime.get_artifact(f"regression_result:{self.model_id}")
        group_variable = str(regression_result.metadata["group_variable"])
        report = build_mixed_advanced_robustness_report(
            dataframe,
            dependent_variable=regression_result.dependent_variable,
            independent_variables=regression_result.independent_variables,
            group_variable=group_variable,
            model_id=self.model_id,
            bootstrap_replications=self.bootstrap_replications,
            run_leave_one_group_out=self.run_leave_one_group_out,
            reml=bool(regression_result.metadata.get("reml", False)),
            optimizer=str(regression_result.metadata.get("optimizer", "lbfgs")),
            max_iterations=int(regression_result.metadata.get("max_iterations", 200)),
            random_slope_variable=regression_result.metadata.get("random_slope_variable"),
        )
        self.runtime.set_artifact(f"advanced_robustness_report:{self.model_id}", report)
        output_dir = working_directory / "result" / "12_advanced_robustness" / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        coefficient_path = output_dir / "group_bootstrap_coefficients.xlsx"
        logo_path = output_dir / "leave_one_group_out.xlsx"
        summary_path = output_dir / "advanced_robustness_summary.xlsx"
        mixed_resampling_to_dataframe(report).to_excel(coefficient_path, index=False)
        report.leave_one_group_out.to_excel(logo_path, index=False)
        mixed_advanced_summary_to_dataframe(report).to_excel(summary_path, index=False)
        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(coefficient_path), str(logo_path), str(summary_path)],
            warnings=report.warnings,
            metadata={"model_id": self.model_id, **report.metadata},
        )
