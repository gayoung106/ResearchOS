"""회귀진단 엔진을 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.diagnostics.binary_logit import (
    binary_diagnostic_summary_to_dataframe,
    binary_multicollinearity_to_dataframe,
    binary_predictions_to_dataframe,
    build_binary_logit_diagnostics,
    classification_metrics_to_dataframe,
)
from src.statistics.diagnostics.count import (
    build_count_diagnostics,
    count_diagnostic_summary_to_dataframe,
    count_multicollinearity_to_dataframe,
    count_observations_to_dataframe,
    count_prediction_metrics_to_dataframe,
)
from src.statistics.diagnostics.gee import (
    build_gee_diagnostics,
    gee_cluster_diagnostics_to_dataframe,
    gee_diagnostic_summary_to_dataframe,
    gee_residuals_to_dataframe,
)
from src.statistics.diagnostics.mixed_effects import (
    build_mixed_effects_diagnostics,
    mixed_effects_diagnostic_summary_to_dataframe,
    mixed_effects_group_residuals_to_dataframe,
    mixed_effects_random_effects_to_dataframe,
    mixed_effects_residuals_to_dataframe,
    mixed_effects_tests_to_dataframe,
)
from src.statistics.diagnostics.multinomial_logit import (
    build_multinomial_logit_diagnostics,
    multinomial_classification_metrics_to_dataframe,
    multinomial_confusion_matrix_to_dataframe,
    multinomial_diagnostic_summary_to_dataframe,
    multinomial_multicollinearity_to_dataframe,
    multinomial_predictions_to_dataframe,
)
from src.statistics.diagnostics.ols import (
    build_ols_diagnostics,
    diagnostic_summary_to_dataframe,
    influence_to_dataframe,
    multicollinearity_to_dataframe,
    residuals_to_dataframe,
    tests_to_dataframe,
)
from src.statistics.diagnostics.ordered_logit import (
    build_ordered_logit_diagnostics,
    ordered_classification_metrics_to_dataframe,
    ordered_confusion_matrix_to_dataframe,
    ordered_diagnostic_summary_to_dataframe,
    ordered_multicollinearity_to_dataframe,
    ordered_predictions_to_dataframe,
    ordered_thresholds_to_dataframe,
)


class RegressionDiagnosticsStep(PipelineStep):
    """저장된 회귀결과에 적합한 진단을 실행한다."""

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
        result = self.runtime.get_artifact(f"regression_result:{self.model_id}")

        output_dir = working_directory / "result" / "10_diagnostics" / self.model_id
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if result.model_type == "multinomial_logit":
            return self._run_multinomial_logit(
                result,
                output_dir,
            )

        if result.model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
            return self._run_gee(
                result,
                output_dir,
            )

        if result.model_type in {
            "mixed_random_intercept",
            "mixed_random_slope",
            "mixed_three_level",
        }:
            return self._run_mixed_effects(
                result,
                output_dir,
            )

        if result.model_type == "ols":
            return self._run_ols(
                result,
                output_dir,
            )

        if result.model_type in {
            "binary_logit",
            "mixed_binary_logit_random_intercept",
            "mixed_binary_logit_random_slope",
            "mixed_binary_logit_three_level",
        }:
            return self._run_binary_logit(
                result,
                output_dir,
            )

        if result.model_type == "ordered_logit":
            return self._run_ordered_logit(
                result,
                output_dir,
            )

        if result.model_type in {
            "poisson",
            "negative_binomial",
            "zero_inflated_poisson",
            "zero_inflated_negative_binomial",
            "mixed_poisson_random_intercept",
            "mixed_poisson_random_slope",
            "mixed_poisson_three_level",
            "mixed_negative_binomial_random_intercept",
            "mixed_negative_binomial_random_slope",
            "mixed_negative_binomial_three_level",
        }:
            return self._run_count(
                result,
                output_dir,
            )

        return StepResult(
            stage_name=self.name,
            success=True,
            warnings=["현재 진단 단계가 지원하지 않는 회귀모형이므로 생략했습니다."],
            metadata={
                "model_id": self.model_id,
                "model_type": result.model_type,
                "skipped": True,
            },
        )

    def _store_report(
        self,
        report: Any,
    ) -> None:
        self.runtime.set_artifact(
            f"regression_diagnostics:{self.model_id}",
            report,
        )


    def _run_multinomial_logit(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_multinomial_logit_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "classification_metrics.xlsx",
            "predictions": output_dir / "predictions.xlsx",
            "confusion": output_dir / "confusion_matrix.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        multinomial_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        multinomial_classification_metrics_to_dataframe(report).to_excel(
            paths["metrics"], index=False
        )
        multinomial_predictions_to_dataframe(report).to_excel(paths["predictions"], index=False)
        multinomial_confusion_matrix_to_dataframe(report).to_excel(
            paths["confusion"], index=False
        )
        multinomial_diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"], index=False
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_gee(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_gee_diagnostics(result)
        self._store_report(report)

        paths = {
            "clusters": output_dir / "gee_cluster_diagnostics.xlsx",
            "residuals": output_dir / "gee_residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        gee_cluster_diagnostics_to_dataframe(report).to_excel(paths["clusters"], index=False)
        gee_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        gee_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_mixed_effects(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_mixed_effects_diagnostics(result)
        self._store_report(report)

        paths = {
            "tests": output_dir / "diagnostic_tests.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "group_residuals": output_dir / "group_residuals.xlsx",
            "random_effects": output_dir / "random_effects.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }

        mixed_effects_tests_to_dataframe(report).to_excel(
            paths["tests"],
            index=False,
        )
        mixed_effects_residuals_to_dataframe(report).to_excel(
            paths["residuals"],
            index=False,
        )
        mixed_effects_group_residuals_to_dataframe(report).to_excel(
            paths["group_residuals"],
            index=False,
        )
        mixed_effects_random_effects_to_dataframe(report).to_excel(
            paths["random_effects"],
            index=False,
        )
        mixed_effects_diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"],
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_ols(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_ols_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "tests": output_dir / "diagnostic_tests.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "influence": output_dir / "influence.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }

        multicollinearity_to_dataframe(report).to_excel(
            paths["vif"],
            index=False,
        )
        tests_to_dataframe(report).to_excel(
            paths["tests"],
            index=False,
        )
        residuals_to_dataframe(report).to_excel(
            paths["residuals"],
            index=False,
        )
        influence_to_dataframe(report).to_excel(
            paths["influence"],
            index=False,
        )
        diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"],
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_binary_logit(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_binary_logit_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "classification_metrics.xlsx",
            "predictions": output_dir / "predictions.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }

        binary_multicollinearity_to_dataframe(report).to_excel(
            paths["vif"],
            index=False,
        )
        classification_metrics_to_dataframe(report).to_excel(
            paths["metrics"],
            index=False,
        )
        binary_predictions_to_dataframe(report).to_excel(
            paths["predictions"],
            index=False,
        )
        binary_diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"],
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_ordered_logit(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_ordered_logit_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "classification_metrics.xlsx",
            "predictions": output_dir / "predictions.xlsx",
            "confusion": output_dir / "confusion_matrix.xlsx",
            "thresholds": output_dir / "thresholds.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }

        ordered_multicollinearity_to_dataframe(report).to_excel(
            paths["vif"],
            index=False,
        )
        ordered_classification_metrics_to_dataframe(report).to_excel(
            paths["metrics"],
            index=False,
        )
        ordered_predictions_to_dataframe(report).to_excel(
            paths["predictions"],
            index=False,
        )
        ordered_confusion_matrix_to_dataframe(report).to_excel(
            paths["confusion"],
            index=False,
        )
        ordered_thresholds_to_dataframe(report).to_excel(
            paths["thresholds"],
            index=False,
        )
        ordered_diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"],
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_count(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_count_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }

        count_multicollinearity_to_dataframe(report).to_excel(
            paths["vif"],
            index=False,
        )
        count_prediction_metrics_to_dataframe(report).to_excel(
            paths["metrics"],
            index=False,
        )
        count_observations_to_dataframe(report).to_excel(
            paths["observations"],
            index=False,
        )
        count_diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"],
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )
