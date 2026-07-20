from src.reporting.regression import build_regression_publication_report
from src.statistics.effects.regression import EffectSizeReport, EffectSizeResult
from src.statistics.regression.base import ModelCoefficient, RegressionResult


def _coefficient() -> ModelCoefficient:
    return ModelCoefficient(
        term="x",
        estimate=0.4,
        standard_error=0.1,
        statistic=4.0,
        p_value=0.001,
        confidence_interval_lower=0.2,
        confidence_interval_upper=0.6,
        exponentiated_estimate=1.49,
    )


def _effect_report(model_type: str) -> EffectSizeReport:
    return EffectSizeReport(
        model_id="main_model",
        model_type=model_type,
        effects=[
            EffectSizeResult(
                term="x",
                effect_type="incidence_rate_ratio",
                estimate=1.49,
                standard_error=None,
                statistic=4.0,
                p_value=0.001,
                confidence_interval_lower=1.22,
                confidence_interval_upper=1.82,
                magnitude=None,
                interpretation="IRR",
            )
        ],
        model_effects={},
    )


def test_reporting_narrative_describes_random_slope_glmm_structure() -> None:
    result = RegressionResult(
        model_id="main_model",
        model_type="mixed_negative_binomial_random_slope",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=84,
        coefficients=[_coefficient()],
        fit_statistics={
            "group_count": 6,
            "random_intercept_variance": 0.12,
            "random_slope_variance": 0.04,
            "dispersion_alpha": 0.55,
        },
        converged=True,
        standard_error_type="maximum_likelihood_hessian",
        metadata={"group_variable": "group", "random_slope_variable": "x"},
        raw_result=object(),
    )

    report = build_regression_publication_report(result, _effect_report(result.model_type))

    assert report.model_type == "mixed_negative_binomial_random_slope"
    assert report.metadata["group_variable"] == "group"
    assert "The GLMM included 6 groups defined by group." in report.narrative
    assert "A random slope for x was estimated" in report.narrative
    assert "alpha was 0.550" in report.narrative
    assert "The GLMM converged." in report.narrative
    assert any("GLMM notes include group structure" in note for note in report.notes)


def test_reporting_narrative_describes_three_level_glmm_structure() -> None:
    result = RegressionResult(
        model_id="main_model",
        model_type="mixed_poisson_three_level",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=108,
        coefficients=[_coefficient()],
        fit_statistics={
            "level2_group_count": 9,
            "level3_group_count": 3,
            "level2_vpc": 0.6,
            "level3_vpc": 0.4,
        },
        converged=True,
        standard_error_type="variational_bayes_posterior_sd",
        metadata={"level2_group": "cluster", "level3_group": "region"},
        raw_result=object(),
    )

    report = build_regression_publication_report(result, _effect_report(result.model_type))

    assert report.metadata["level2_group"] == "cluster"
    assert report.metadata["level3_group"] == "region"
    assert report.metadata["level2_group_count"] == 9
    assert "3-level GLMM included 9 cluster groups nested within 3 region groups." in report.narrative
    assert "Variance partition coefficients were Level 2=0.600 and Level 3=0.400." in report.narrative
