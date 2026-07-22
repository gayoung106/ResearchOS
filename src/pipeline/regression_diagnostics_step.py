"""회귀진단 엔진을 파이프라인에 연결하는 단계."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.statistics.diagnostics.beta import (
    beta_diagnostic_summary_to_dataframe,
    beta_multicollinearity_to_dataframe,
    beta_observations_to_dataframe,
    beta_prediction_metrics_to_dataframe,
    build_beta_diagnostics,
)
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
from src.statistics.diagnostics.cox import (
    build_cox_diagnostics,
    cox_baseline_survival_to_dataframe,
    cox_diagnostic_summary_to_dataframe,
    cox_multicollinearity_to_dataframe,
    cox_ph_checks_to_dataframe,
    cox_residuals_to_dataframe,
)
from src.statistics.diagnostics.discrete_time_hazard import (
    build_discrete_time_hazard_diagnostics,
    discrete_time_diagnostic_summary_to_dataframe,
    discrete_time_interval_hazards_to_dataframe,
    discrete_time_residuals_to_dataframe,
)
from src.statistics.diagnostics.exponential_aft import (
    build_exponential_aft_diagnostics,
    exponential_aft_diagnostic_summary_to_dataframe,
    exponential_aft_multicollinearity_to_dataframe,
    exponential_aft_prediction_metrics_to_dataframe,
    exponential_aft_residuals_to_dataframe,
)
from src.statistics.diagnostics.fractional_logit import (
    build_fractional_logit_diagnostics,
    fractional_diagnostic_summary_to_dataframe,
    fractional_multicollinearity_to_dataframe,
    fractional_observations_to_dataframe,
    fractional_prediction_metrics_to_dataframe,
)
from src.statistics.diagnostics.gamma import (
    build_gamma_diagnostics,
    gamma_diagnostic_summary_to_dataframe,
    gamma_multicollinearity_to_dataframe,
    gamma_observations_to_dataframe,
    gamma_prediction_metrics_to_dataframe,
)
from src.statistics.diagnostics.gee import (
    build_gee_diagnostics,
    gee_cluster_diagnostics_to_dataframe,
    gee_diagnostic_summary_to_dataframe,
    gee_residuals_to_dataframe,
)
from src.statistics.diagnostics.heckman import (
    build_heckman_diagnostics,
    heckman_diagnostic_summary_to_dataframe,
    heckman_multicollinearity_to_dataframe,
    heckman_residuals_to_dataframe,
    heckman_selection_coefficients_to_dataframe,
)
from src.statistics.diagnostics.inverse_gaussian import (
    build_inverse_gaussian_diagnostics,
    inverse_gaussian_diagnostic_summary_to_dataframe,
    inverse_gaussian_multicollinearity_to_dataframe,
    inverse_gaussian_observations_to_dataframe,
    inverse_gaussian_prediction_metrics_to_dataframe,
)
from src.statistics.diagnostics.iv import (
    build_iv_2sls_diagnostics,
    iv_diagnostic_summary_to_dataframe,
    iv_first_stage_to_dataframe,
    iv_multicollinearity_to_dataframe,
    iv_residuals_to_dataframe,
)
from src.statistics.diagnostics.loglogistic_aft import (
    build_loglogistic_aft_diagnostics,
    loglogistic_aft_diagnostic_summary_to_dataframe,
    loglogistic_aft_multicollinearity_to_dataframe,
    loglogistic_aft_prediction_metrics_to_dataframe,
    loglogistic_aft_residuals_to_dataframe,
)
from src.statistics.diagnostics.lognormal_aft import (
    build_lognormal_aft_diagnostics,
    lognormal_aft_diagnostic_summary_to_dataframe,
    lognormal_aft_multicollinearity_to_dataframe,
    lognormal_aft_prediction_metrics_to_dataframe,
    lognormal_aft_residuals_to_dataframe,
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
from src.statistics.diagnostics.panel import (
    build_panel_diagnostics,
    panel_diagnostic_summary_to_dataframe,
    panel_entity_residuals_to_dataframe,
    panel_multicollinearity_to_dataframe,
    panel_residuals_to_dataframe,
)
from src.statistics.diagnostics.piecewise_exponential import (
    build_piecewise_exponential_diagnostics,
    piecewise_diagnostic_summary_to_dataframe,
    piecewise_interval_hazards_to_dataframe,
    piecewise_residuals_to_dataframe,
)
from src.statistics.diagnostics.quantile import (
    build_quantile_diagnostics,
    quantile_diagnostic_summary_to_dataframe,
    quantile_multicollinearity_to_dataframe,
    quantile_residual_summary_to_dataframe,
    quantile_residuals_to_dataframe,
)
from src.statistics.diagnostics.regularized import (
    build_regularized_diagnostics,
    regularized_coefficients_to_dataframe,
    regularized_diagnostic_summary_to_dataframe,
    regularized_multicollinearity_to_dataframe,
    regularized_prediction_metrics_to_dataframe,
    regularized_residuals_to_dataframe,
)
from src.statistics.diagnostics.robust import (
    build_robust_diagnostics,
    robust_diagnostic_summary_to_dataframe,
    robust_multicollinearity_to_dataframe,
    robust_residuals_to_dataframe,
    robust_weight_summary_to_dataframe,
)
from src.statistics.diagnostics.tobit import (
    build_tobit_diagnostics,
    tobit_diagnostic_summary_to_dataframe,
    tobit_multicollinearity_to_dataframe,
    tobit_observations_to_dataframe,
    tobit_prediction_metrics_to_dataframe,
)
from src.statistics.diagnostics.tweedie import (
    build_tweedie_diagnostics,
    tweedie_diagnostic_summary_to_dataframe,
    tweedie_multicollinearity_to_dataframe,
    tweedie_observations_to_dataframe,
    tweedie_prediction_metrics_to_dataframe,
)
from src.statistics.diagnostics.weibull_aft import (
    build_weibull_aft_diagnostics,
    weibull_aft_diagnostic_summary_to_dataframe,
    weibull_aft_multicollinearity_to_dataframe,
    weibull_aft_prediction_metrics_to_dataframe,
    weibull_aft_residuals_to_dataframe,
)
from src.statistics.diagnostics.weibull_ph import (
    build_weibull_ph_diagnostics,
    weibull_ph_diagnostic_summary_to_dataframe,
    weibull_ph_multicollinearity_to_dataframe,
    weibull_ph_prediction_metrics_to_dataframe,
    weibull_ph_residuals_to_dataframe,
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

        if result.model_type == "heckman_selection":
            return self._run_heckman_selection(
                result,
                output_dir,
            )

        if result.model_type == "iv_2sls_regression":
            return self._run_iv_2sls_regression(
                result,
                output_dir,
            )

        if result.model_type == "tweedie_regression":
            return self._run_tweedie_regression(
                result,
                output_dir,
            )

        if result.model_type == "inverse_gaussian_regression":
            return self._run_inverse_gaussian_regression(
                result,
                output_dir,
            )

        if result.model_type == "gamma_regression":
            return self._run_gamma_regression(
                result,
                output_dir,
            )

        if result.model_type == "regularized_regression":
            return self._run_regularized_regression(
                result,
                output_dir,
            )

        if result.model_type == "robust_regression":
            return self._run_robust_regression(
                result,
                output_dir,
            )

        if result.model_type == "tobit_regression":
            return self._run_tobit_regression(
                result,
                output_dir,
            )

        if result.model_type == "panel_fixed_effects":
            return self._run_panel_fixed_effects(
                result,
                output_dir,
            )

        if result.model_type == "beta_regression":
            return self._run_beta_regression(
                result,
                output_dir,
            )

        if result.model_type == "fractional_logit":
            return self._run_fractional_logit(
                result,
                output_dir,
            )

        if result.model_type in {"cox_proportional_hazards", "stratified_cox", "left_truncated_cox", "cause_specific_cox", "clustered_cox", "time_varying_cox"}:
            return self._run_cox(
                result,
                output_dir,
            )

        if result.model_type == "piecewise_exponential":
            return self._run_piecewise_exponential(
                result,
                output_dir,
            )

        if result.model_type == "discrete_time_hazard":
            return self._run_discrete_time_hazard(
                result,
                output_dir,
            )

        if result.model_type == "exponential_aft":
            return self._run_exponential_aft(
                result,
                output_dir,
            )

        if result.model_type == "loglogistic_aft":
            return self._run_loglogistic_aft(
                result,
                output_dir,
            )

        if result.model_type == "lognormal_aft":
            return self._run_lognormal_aft(
                result,
                output_dir,
            )

        if result.model_type == "weibull_aft":
            return self._run_weibull_aft(
                result,
                output_dir,
            )

        if result.model_type == "weibull_ph":
            return self._run_weibull_ph(
                result,
                output_dir,
            )

        if result.model_type == "quantile_regression":
            return self._run_quantile(
                result,
                output_dir,
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

        if result.model_type in {"ols", "weighted_least_squares"}:
            return self._run_ols(
                result,
                output_dir,
            )

        if result.model_type in {
            "binary_logit",
            "log_binomial",
            "modified_poisson",
            "binary_cloglog",
            "binary_probit",
            "mixed_binary_logit_random_intercept",
            "mixed_binary_logit_random_slope",
            "mixed_binary_logit_three_level",
        }:
            return self._run_binary_logit(
                result,
                output_dir,
            )

        if result.model_type in {"ordered_logit", "ordered_probit"}:
            return self._run_ordered_logit(
                result,
                output_dir,
            )

        if result.model_type in {
            "poisson",
            "quasi_poisson",
            "negative_binomial",
            "generalized_poisson",
            "zero_inflated_poisson",
            "zero_inflated_negative_binomial",
            "hurdle_poisson",
            "hurdle_negative_binomial",
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


    def _run_heckman_selection(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_heckman_diagnostics(result)
        self._store_report(report)

        paths = {
            "selection": output_dir / "selection_equation.xlsx",
            "vif": output_dir / "multicollinearity.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        heckman_selection_coefficients_to_dataframe(report).to_excel(
            paths["selection"], index=False
        )
        heckman_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        heckman_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        heckman_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_iv_2sls_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_iv_2sls_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "first_stage": output_dir / "first_stage.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        iv_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        iv_first_stage_to_dataframe(report).to_excel(paths["first_stage"], index=False)
        iv_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        iv_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_inverse_gaussian_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_inverse_gaussian_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        inverse_gaussian_multicollinearity_to_dataframe(report).to_excel(
            paths["vif"], index=False
        )
        inverse_gaussian_prediction_metrics_to_dataframe(report).to_excel(
            paths["metrics"], index=False
        )
        inverse_gaussian_observations_to_dataframe(report).to_excel(
            paths["observations"], index=False
        )
        inverse_gaussian_diagnostic_summary_to_dataframe(report).to_excel(
            paths["summary"], index=False
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_tweedie_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_tweedie_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        tweedie_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        tweedie_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        tweedie_observations_to_dataframe(report).to_excel(paths["observations"], index=False)
        tweedie_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_gamma_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_gamma_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        gamma_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        gamma_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        gamma_observations_to_dataframe(report).to_excel(paths["observations"], index=False)
        gamma_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_regularized_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_regularized_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "coefficients": output_dir / "coefficient_selection.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        regularized_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        regularized_coefficients_to_dataframe(report).to_excel(paths["coefficients"], index=False)
        regularized_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        regularized_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        regularized_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_robust_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_robust_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "weights": output_dir / "weight_summary.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        robust_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        robust_weight_summary_to_dataframe(report).to_excel(paths["weights"], index=False)
        robust_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        robust_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_tobit_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_tobit_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        tobit_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        tobit_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        tobit_observations_to_dataframe(report).to_excel(paths["observations"], index=False)
        tobit_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_panel_fixed_effects(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_panel_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "entity_residuals": output_dir / "entity_residuals.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        panel_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        panel_entity_residuals_to_dataframe(report).to_excel(paths["entity_residuals"], index=False)
        panel_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        panel_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_beta_regression(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_beta_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        beta_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        beta_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        beta_observations_to_dataframe(report).to_excel(paths["observations"], index=False)
        beta_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_fractional_logit(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_fractional_logit_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "observations": output_dir / "observations.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        fractional_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        fractional_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        fractional_observations_to_dataframe(report).to_excel(paths["observations"], index=False)
        fractional_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_cox(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_cox_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "ph": output_dir / "proportional_hazards_checks.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "baseline": output_dir / "baseline_survival.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        cox_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        cox_ph_checks_to_dataframe(report).to_excel(paths["ph"], index=False)
        cox_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        cox_baseline_survival_to_dataframe(report).to_excel(paths["baseline"], index=False)
        cox_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )


    def _run_piecewise_exponential(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_piecewise_exponential_diagnostics(result)
        self._store_report(report)

        paths = {
            "interval_hazards": output_dir / "interval_hazards.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        piecewise_interval_hazards_to_dataframe(report).to_excel(paths["interval_hazards"], index=False)
        piecewise_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        piecewise_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )


    def _run_discrete_time_hazard(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_discrete_time_hazard_diagnostics(result)
        self._store_report(report)

        paths = {
            "interval_hazards": output_dir / "interval_hazards.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        discrete_time_interval_hazards_to_dataframe(report).to_excel(paths["interval_hazards"], index=False)
        discrete_time_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        discrete_time_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_exponential_aft(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_exponential_aft_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        exponential_aft_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        exponential_aft_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        exponential_aft_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        exponential_aft_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_loglogistic_aft(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_loglogistic_aft_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        loglogistic_aft_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        loglogistic_aft_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        loglogistic_aft_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        loglogistic_aft_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_lognormal_aft(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_lognormal_aft_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        lognormal_aft_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        lognormal_aft_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        lognormal_aft_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        lognormal_aft_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_weibull_aft(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_weibull_aft_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        weibull_aft_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        weibull_aft_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        weibull_aft_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        weibull_aft_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_weibull_ph(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_weibull_ph_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "metrics": output_dir / "prediction_metrics.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        weibull_ph_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        weibull_ph_prediction_metrics_to_dataframe(report).to_excel(paths["metrics"], index=False)
        weibull_ph_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        weibull_ph_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
        )

    def _run_quantile(
        self,
        result: Any,
        output_dir: Path,
    ) -> StepResult:
        report = build_quantile_diagnostics(result)
        self._store_report(report)

        paths = {
            "vif": output_dir / "multicollinearity.xlsx",
            "residual_summary": output_dir / "residual_summary.xlsx",
            "residuals": output_dir / "residuals.xlsx",
            "summary": output_dir / "diagnostic_summary.xlsx",
        }
        quantile_multicollinearity_to_dataframe(report).to_excel(paths["vif"], index=False)
        quantile_residual_summary_to_dataframe(report).to_excel(
            paths["residual_summary"], index=False
        )
        quantile_residuals_to_dataframe(report).to_excel(paths["residuals"], index=False)
        quantile_diagnostic_summary_to_dataframe(report).to_excel(paths["summary"], index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(path) for path in paths.values()],
            warnings=report.warnings,
            metadata=report.summary,
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
