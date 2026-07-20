from pathlib import Path

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.visualization.regression import build_regression_visualizations


def _coefficient() -> ModelCoefficient:
    return ModelCoefficient(
        term="x",
        estimate=0.4,
        standard_error=0.1,
        statistic=4.0,
        p_value=0.01,
        confidence_interval_lower=0.2,
        confidence_interval_upper=0.6,
        exponentiated_estimate=1.5,
    )


def test_glmm_random_slope_visualization_outputs_slope_plot(tmp_path: Path) -> None:
    result = RegressionResult(
        model_id="main_model",
        model_type="mixed_poisson_random_slope",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=12,
        coefficients=[_coefficient()],
        fit_statistics={"group_count": 3, "random_slope_variance": 0.05},
        converged=True,
        standard_error_type="test",
        metadata={
            "group_variable": "group",
            "random_effects": {"a": -0.1, "b": 0.0, "c": 0.1},
            "random_slopes": {"a": -0.2, "b": 0.05, "c": 0.25},
            "random_slope_variable": "x",
        },
        raw_result=object(),
    )

    report = build_regression_visualizations(result, output_directory=tmp_path)

    assert {Path(path).name for path in report.output_files} == {
        "coefficient_forest.png",
        "random_intercepts.png",
        "random_slopes.png",
    }
    assert all(Path(path).exists() for path in report.output_files)


def test_glmm_three_level_visualization_outputs_level3_plot(tmp_path: Path) -> None:
    result = RegressionResult(
        model_id="main_model",
        model_type="mixed_poisson_three_level",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=18,
        coefficients=[_coefficient()],
        fit_statistics={
            "level2_group_count": 3,
            "level3_group_count": 2,
            "level2_vpc": 0.6,
            "level3_vpc": 0.3,
        },
        converged=True,
        standard_error_type="test",
        metadata={
            "level2_group": "cluster",
            "level3_group": "region",
            "level2_random_effects": {"c1": -0.1, "c2": 0.0, "c3": 0.1},
            "level3_random_effects": {"r1": -0.2, "r2": 0.2},
        },
        raw_result=object(),
    )

    report = build_regression_visualizations(result, output_directory=tmp_path)

    assert {Path(path).name for path in report.output_files} == {
        "coefficient_forest.png",
        "random_intercepts.png",
        "level3_random_intercepts.png",
        "three_level_variance_partition.png",
    }
    assert all(Path(path).exists() for path in report.output_files)
