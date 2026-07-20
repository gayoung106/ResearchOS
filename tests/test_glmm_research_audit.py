from src.audit.research import build_research_audit_report
from src.pipeline.runtime import PipelineRuntime
from src.statistics.regression.base import ModelCoefficient, RegressionResult


def _coefficient() -> ModelCoefficient:
    return ModelCoefficient(
        term="x",
        estimate=0.3,
        standard_error=0.1,
        statistic=3.0,
        p_value=0.003,
        confidence_interval_lower=0.1,
        confidence_interval_upper=0.5,
        exponentiated_estimate=1.35,
    )


def _three_level_glmm_result() -> RegressionResult:
    return RegressionResult(
        model_id="main_model",
        model_type="mixed_negative_binomial_three_level",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=108,
        coefficients=[_coefficient()],
        fit_statistics={
            "level2_group_count": 9,
            "level3_group_count": 3,
            "level2_vpc": 0.62,
            "level3_vpc": 0.38,
        },
        converged=True,
        standard_error_type="maximum_likelihood_hessian",
        metadata={
            "level2_group": "cluster",
            "level3_group": "region",
            "nested_structure": True,
        },
        raw_result=object(),
    )


def test_audit_reports_three_level_glmm_structure() -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact("regression_result:main_model", _three_level_glmm_result())

    report = build_research_audit_report(runtime, model_id="main_model")
    regression_item = next(item for item in report.items if item.maximum_score == 15)

    assert report.metadata["model_type"] == "mixed_negative_binomial_three_level"
    assert report.metadata["level2_group"] == "cluster"
    assert report.metadata["level3_group"] == "region"
    assert report.metadata["level2_group_count"] == 9
    assert report.metadata["level3_group_count"] == 3
    assert report.metadata["level2_vpc"] == 0.62
    assert report.metadata["level3_vpc"] == 0.38
    assert "3" in regression_item.evidence
    assert "Level 2=cluster(9" in regression_item.evidence
    assert "Level 3=region(3" in regression_item.evidence
