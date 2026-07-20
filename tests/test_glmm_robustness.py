from pathlib import Path

import pandas as pd

from src.audit.research import build_research_audit_report
from src.common.config_models import AnalysisPlan, VariableDefinition, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.robustness_step import GLMMRobustnessStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.robustness import glmm
from src.statistics.robustness.glmm import (
    build_glmm_robustness_report,
    glmm_coefficient_comparison_to_dataframe,
    glmm_model_comparison_to_dataframe,
    glmm_robustness_summary_to_dataframe,
    glmm_stability_summary_to_dataframe,
)


def _coef(term: str, estimate: float) -> ModelCoefficient:
    return ModelCoefficient(
        term=term,
        estimate=estimate,
        standard_error=0.1,
        statistic=estimate / 0.1,
        p_value=0.01,
        confidence_interval_lower=estimate - 0.2,
        confidence_interval_upper=estimate + 0.2,
        exponentiated_estimate=1.2,
    )


def _baseline() -> RegressionResult:
    return RegressionResult(
        model_id="main_model",
        model_type="mixed_poisson_random_intercept",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=8,
        coefficients=[_coef("const", 0.1), _coef("x", 0.4)],
        fit_statistics={"group_count": 2, "random_intercept_variance": 0.1},
        converged=True,
        standard_error_type="variational_bayes_posterior_sd",
        metadata={"group_variable": "group", "optimizer": "BFGS", "max_iterations": 20},
        raw_result=object(),
    )


def test_build_glmm_robustness_report_with_optimizer_refits(monkeypatch) -> None:
    baseline = _baseline()

    def fake_fit(*args, **kwargs):
        optimizer = kwargs["mixed_effects_options"]["optimizer"]
        estimate = 0.4 if optimizer == "BFGS" else 0.42
        return RegressionResult(
            model_id=f"main_model_{optimizer}",
            model_type=baseline.model_type,
            dependent_variable="y",
            independent_variables=["x"],
            sample_size=8,
            coefficients=[_coef("const", 0.1), _coef("x", estimate)],
            fit_statistics={"group_count": 2, "random_intercept_variance": 0.1},
            converged=True,
            standard_error_type="variational_bayes_posterior_sd",
            metadata=baseline.metadata,
            raw_result=object(),
        )

    monkeypatch.setattr(glmm, "fit_regression_by_level", fake_fit)

    report = build_glmm_robustness_report(
        pd.DataFrame({"y": [0, 1], "x": [0.0, 1.0], "group": ["a", "b"]}),
        baseline_result=baseline,
        optimizers=("BFGS", "Powell"),
    )

    assert report.model_type == "mixed_poisson_random_intercept"
    assert report.summary["successful_optimizer_count"] == 2
    assert report.summary["stable_term_count"] == 1
    assert glmm_coefficient_comparison_to_dataframe(report).shape[0] == 4
    assert glmm_stability_summary_to_dataframe(report).shape[0] == 2
    assert glmm_model_comparison_to_dataframe(report).shape[0] == 2
    assert "warning_count" in set(glmm_robustness_summary_to_dataframe(report)["item"])


def test_glmm_robustness_step_and_audit_accept_report(monkeypatch, tmp_path: Path) -> None:
    baseline = _baseline()
    runtime = PipelineRuntime(
        dataframe=pd.DataFrame({"y": [0, 1], "x": [0.0, 1.0], "group": ["a", "b"]})
    )
    runtime.set_artifact("regression_result:main_model", baseline)

    def fake_report(*args, **kwargs):
        return build_glmm_robustness_report(
            runtime.dataframe,
            baseline_result=baseline,
            optimizers=("BFGS",),
        )

    monkeypatch.setattr(
        "src.statistics.robustness.glmm.build_glmm_robustness_report",
        fake_report,
    )
    monkeypatch.setattr(
        glmm,
        "fit_regression_by_level",
        lambda *args, **kwargs: baseline,
    )

    result = GLMMRobustnessStep(
        runtime,
        model_id="main_model",
        optimizers=("BFGS",),
    ).run(ResearchContext(project_name="glmm robustness"), tmp_path)
    audit = build_research_audit_report(runtime, model_id="main_model")

    assert result.success is True
    assert len(result.output_files) == 4
    assert runtime.get_artifact("robustness_report:main_model").model_type == baseline.model_type
    robustness_item = next(item for item in audit.items if item.item == "강건성 분석")
    assert robustness_item.status == "PASS"


def test_builder_registers_glmm_robustness_when_enabled(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["group"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {"enabled": True, "options": {"group_variable": "group"}},
                "robustness": {
                    "enabled": True,
                    "options": {"mixed_glmm_optimizers": ["BFGS", "Powell"]},
                },
            },
        }
    )
    variable_map = VariableMap(
        variables={
            "y": VariableDefinition(role="dependent", measurement_level="count"),
            "x": VariableDefinition(role="independent", measurement_level="continuous"),
            "group": VariableDefinition(role="cluster", measurement_level="nominal"),
        }
    )
    runtime = PipelineRuntime()
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(project_name="glmm builder"),
        working_directory=tmp_path,
    )

    from src.pipeline.regression_builder import register_regression_pipeline

    registration = register_regression_pipeline(
        orchestrator=orchestrator,
        runtime=runtime,
        analysis_plan=plan,
        variable_map=variable_map,
    )

    assert registration.model_type == "mixed_poisson_random_intercept"
    assert registration.robustness_registered is True
    step = orchestrator.registry.get("11_robustness_analysis")
    assert isinstance(step, GLMMRobustnessStep)
    assert step.optimizers == ("BFGS", "Powell")

