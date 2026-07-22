"""회귀분석 결과 시각화 생성."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats

from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class RegressionVisualizationReport:
    """회귀 시각화 생성 결과."""

    model_id: str
    model_type: str
    output_files: list[str]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_FONT_INITIALIZED = False


def _configure_matplotlib_font() -> None:
    """운영체제에 맞는 폰트를 자동 설정한다."""
    global _FONT_INITIALIZED

    if _FONT_INITIALIZED:
        return

    system = platform.system()

    if system == "Windows":
        candidates = [
            "Malgun Gothic",
            "맑은 고딕",
        ]
    elif system == "Darwin":
        candidates = [
            "Apple SD Gothic Neo",
            "AppleGothic",
        ]
    else:
        candidates = [
            "Noto Sans CJK KR",
            "NanumGothic",
            "DejaVu Sans",
        ]

    available = {font.name for font in font_manager.fontManager.ttflist}

    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break

    plt.rcParams["axes.unicode_minus"] = False

    _FONT_INITIALIZED = True


def _coefficient_base_term(term: str) -> str:
    return term.rsplit("::", 1)[-1]


def _is_intercept_term(term: str) -> bool:
    return _coefficient_base_term(term).lower() in {"const", "intercept"}


def _save_figure(
    figure: Any,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def _plot_residuals_vs_fitted(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()

    fitted = regression_result.raw_result
    fitted_values = np.asarray(fitted.fittedvalues)
    residuals = np.asarray(fitted.resid)

    figure, axis = plt.subplots()

    axis.scatter(fitted_values, residuals)
    axis.axhline(0)

    axis.set_xlabel("적합값")
    axis.set_ylabel("잔차")
    axis.set_title("잔차-적합값 도표")

    _save_figure(
        figure,
        output_path,
    )


def _plot_qq(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()

    fitted = regression_result.raw_result
    residuals = np.asarray(fitted.resid)

    figure = plt.figure()
    axis = figure.add_subplot(111)

    stats.probplot(
        residuals,
        dist="norm",
        plot=axis,
    )

    axis.set_title("잔차 정규 Q-Q 도표")

    _save_figure(
        figure,
        output_path,
    )


def _plot_influence(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()

    fitted = regression_result.raw_result
    influence = fitted.get_influence()

    leverage = np.asarray(influence.hat_matrix_diag)
    studentized = np.asarray(influence.resid_studentized_external)
    cooks_distance = np.asarray(influence.cooks_distance[0])

    marker_sizes = 20 + 200 * (
        cooks_distance
        / max(
            cooks_distance.max(),
            1e-12,
        )
    )

    figure, axis = plt.subplots()

    axis.scatter(
        leverage,
        studentized,
        s=marker_sizes,
        alpha=0.7,
    )

    axis.axhline(0)

    axis.set_xlabel("Leverage")
    axis.set_ylabel("외적 학생화 잔차")
    axis.set_title("영향력 관측치 도표")

    _save_figure(
        figure,
        output_path,
    )


def _plot_observed_vs_predicted(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    observed = np.asarray(fitted.model.endog, dtype=float)
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(predicted, observed, alpha=0.75)
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Predicted proportion")
    axis.set_ylabel("Observed proportion")
    axis.set_title("Observed vs Predicted Proportions")
    _save_figure(figure, output_path)


def _plot_observed_vs_predicted_continuous(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    observed = np.asarray(fitted.model.endog, dtype=float)
    predicted = np.asarray(fitted.fittedvalues, dtype=float)
    lower = float(min(np.min(observed), np.min(predicted)))
    upper = float(max(np.max(observed), np.max(predicted)))
    padding = max((upper - lower) * 0.05, 1e-6)
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(predicted, observed, alpha=0.75)
    axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], linestyle="--")
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Observed")
    axis.set_title("Observed vs Predicted")
    _save_figure(figure, output_path)



def _plot_count_observed_vs_predicted(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    observed = np.asarray(fitted.model.endog, dtype=float)
    predicted = np.asarray(fitted.predict(), dtype=float)
    upper = float(max(np.max(observed), np.max(predicted), 1.0))
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(predicted, observed, alpha=0.7)
    axis.plot([0.0, upper], [0.0, upper], linestyle="--")
    axis.set_xlim(0.0, upper * 1.05)
    axis.set_ylim(0.0, upper * 1.05)
    axis.set_xlabel("Predicted count")
    axis.set_ylabel("Observed count")
    axis.set_title("Observed vs Predicted Counts")
    _save_figure(figure, output_path)

def _plot_baseline_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    frame = regression_result.metadata.get("baseline_survival", [])
    if not frame:
        raise ValueError("No baseline survival estimates are available for visualization.")
    import pandas as pd

    data = pd.DataFrame(frame)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for _, group in data.groupby("stratum"):
        axis.step(group["time"], group["baseline_survival"], where="post")
    axis.set_xlabel("Time")
    axis.set_ylabel("Baseline survival")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Cox Baseline Survival")
    _save_figure(figure, output_path)


def _plot_piecewise_exponential_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    frame = regression_result.metadata.get("baseline_interval_hazards", [])
    if not frame:
        raise ValueError("No piecewise baseline hazard estimates are available for visualization.")
    import pandas as pd

    data = pd.DataFrame(frame)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.step(data["stop"], data["baseline_survival"], where="post")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Baseline survival")
    axis.set_title("Piecewise Exponential Baseline Survival")
    _save_figure(figure, output_path)


def _plot_discrete_time_hazard_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    frame = regression_result.metadata.get("baseline_interval_hazards", [])
    if not frame:
        raise ValueError("No discrete-time baseline hazard estimates are available for visualization.")
    import pandas as pd

    data = pd.DataFrame(frame)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.step(data["stop"], data["baseline_survival"], where="post")
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Baseline survival")
    axis.set_title("Discrete-Time Hazard Baseline Survival")
    _save_figure(figure, output_path)


def _plot_exponential_aft_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    scale = float(np.median(fitted.predict(kind="scale")))
    durations = np.asarray(fitted.model.endog, dtype=float)
    grid = np.linspace(float(np.min(durations)), float(np.max(durations)), 120)
    survival = np.exp(-grid / scale)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(grid, survival)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Predicted survival")
    axis.set_title("Exponential AFT Survival Curve")
    _save_figure(figure, output_path)


def _plot_loglogistic_aft_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    shape = float(regression_result.fit_statistics["shape"])
    scale = float(np.median(fitted.predict(kind="scale")))
    durations = np.asarray(fitted.model.endog, dtype=float)
    grid = np.linspace(float(np.min(durations)), float(np.max(durations)), 120)
    survival = stats.fisk.sf(grid, c=shape, scale=scale)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(grid, survival)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Predicted survival")
    axis.set_title("Log-logistic AFT Survival Curve")
    _save_figure(figure, output_path)


def _plot_lognormal_aft_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    sigma = float(regression_result.fit_statistics["sigma"])
    scale = float(np.median(fitted.predict(kind="scale")))
    durations = np.asarray(fitted.model.endog, dtype=float)
    grid = np.linspace(float(np.min(durations)), float(np.max(durations)), 120)
    survival = stats.lognorm.sf(grid, s=sigma, scale=scale)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(grid, survival)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Predicted survival")
    axis.set_title("Log-normal AFT Survival Curve")
    _save_figure(figure, output_path)


def _plot_weibull_ph_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    shape = float(regression_result.fit_statistics["shape"])
    baseline_rate = regression_result.fit_statistics.get("baseline_rate")
    if baseline_rate is None:
        scale = float(np.median(fitted.predict(kind="scale")))
        baseline_rate = scale ** (-shape)
    durations = np.asarray(fitted.model.endog, dtype=float)
    grid = np.linspace(float(np.min(durations)), float(np.max(durations)), 120)
    survival = np.exp(-float(baseline_rate) * (grid**shape))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(grid, survival)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Baseline survival")
    axis.set_title("Weibull PH Baseline Survival")
    _save_figure(figure, output_path)


def _plot_weibull_aft_survival(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    fitted = regression_result.raw_result
    shape = float(regression_result.fit_statistics["shape"])
    scale = float(np.median(fitted.predict(kind="scale")))
    durations = np.asarray(fitted.model.endog, dtype=float)
    grid = np.linspace(float(np.min(durations)), float(np.max(durations)), 120)
    survival = np.exp(-((grid / scale) ** shape))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(grid, survival)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Time")
    axis.set_ylabel("Predicted survival")
    axis.set_title("Weibull AFT Survival Curve")
    _save_figure(figure, output_path)


def _plot_coefficient_forest(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()

    coefficients = [
        coefficient
        for coefficient in regression_result.coefficients
        if not _is_intercept_term(coefficient.term)
        and "/" not in coefficient.term
        and not (regression_result.model_type == "beta_regression" and coefficient.term == "precision")
    ]

    if not coefficients:
        raise ValueError("포리스트 플롯에 표시할 실질 계수가 없습니다.")

    terms = [item.term for item in coefficients]

    estimates = np.array(
        [item.estimate for item in coefficients],
        dtype=float,
    )

    lower = np.array(
        [item.confidence_interval_lower for item in coefficients],
        dtype=float,
    )

    upper = np.array(
        [item.confidence_interval_upper for item in coefficients],
        dtype=float,
    )

    lower_error = estimates - lower
    upper_error = upper - estimates

    positions = np.arange(len(terms))

    figure, axis = plt.subplots()

    axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    axis.axvline(0)

    axis.set_yticks(positions)
    axis.set_yticklabels(terms)

    axis.set_xlabel("추정계수와 95% 신뢰구간")
    axis.set_title("회귀계수 포리스트 플롯")

    _save_figure(
        figure,
        output_path,
    )


def _plot_random_intercepts(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    """그룹별 Random Intercept 추정치를 정렬해 표시한다."""
    _configure_matplotlib_font()

    fitted = regression_result.raw_result
    rows: list[tuple[str, float]] = []

    if regression_result.model_type in {
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
        "mixed_poisson_random_intercept",
        "mixed_poisson_random_slope",
        "mixed_poisson_three_level",
        "mixed_negative_binomial_random_intercept",
            "mixed_negative_binomial_random_slope",
            "mixed_negative_binomial_three_level",
    }:
        random_effects = regression_result.metadata.get("random_effects", {})
        if regression_result.model_type in {
            "mixed_binary_logit_three_level",
            "mixed_poisson_three_level",
            "mixed_negative_binomial_three_level",
        }:
            random_effects = regression_result.metadata.get("level2_random_effects", {})
        rows = [(str(group), float(effect)) for group, effect in random_effects.items()]
    else:
        for group, effect in fitted.random_effects.items():
            values = np.asarray(effect, dtype=float).reshape(-1)
            if values.size == 0:
                continue
            rows.append((str(group), float(values[0])))

    if not rows:
        raise ValueError("표시할 Random Intercept 추정치가 없습니다.")

    rows.sort(key=lambda item: item[1])
    groups = [item[0] for item in rows]
    estimates = np.asarray([item[1] for item in rows], dtype=float)
    positions = np.arange(len(rows))

    figure_height = max(4.0, 0.28 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(7.0, figure_height))

    axis.scatter(estimates, positions)
    axis.axvline(0)
    axis.set_yticks(positions)
    axis.set_yticklabels(groups)
    axis.set_xlabel("Random Intercept 추정치")
    axis.set_ylabel("그룹")
    axis.set_title("그룹별 Random Intercept 도표")

    _save_figure(figure, output_path)




def _plot_random_slopes(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    rows: list[tuple[str, float]] = []
    if regression_result.model_type in {
        "mixed_binary_logit_random_slope",
        "mixed_poisson_random_slope",
        "mixed_negative_binomial_random_slope",
    }:
        random_slopes = regression_result.metadata.get("random_slopes", {})
        rows = [(str(group), float(effect)) for group, effect in random_slopes.items()]
    elif regression_result.model_type == "mixed_random_slope":
        fitted = regression_result.raw_result
        slope_variable = str(regression_result.metadata.get("random_slope_variable", ""))
        for group, effect in fitted.random_effects.items():
            if hasattr(effect, "get") and slope_variable:
                value = effect.get(slope_variable)
            else:
                values = np.asarray(effect, dtype=float).reshape(-1)
                value = values[1] if values.size > 1 else None
            if value is not None:
                rows.append((str(group), float(value)))

    if not rows:
        raise ValueError("No random slope estimates are available for visualization.")

    rows.sort(key=lambda item: item[1])
    groups = [item[0] for item in rows]
    estimates = np.asarray([item[1] for item in rows], dtype=float)
    positions = np.arange(len(rows))
    figure_height = max(4.0, 0.28 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(7.0, figure_height))
    axis.scatter(estimates, positions)
    axis.axvline(0)
    axis.set_yticks(positions)
    axis.set_yticklabels(groups)
    axis.set_xlabel("Random slope estimate")
    axis.set_ylabel("Group")
    axis.set_title("Group Random Slope Estimates")
    _save_figure(figure, output_path)


def _plot_level3_random_intercepts(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    _configure_matplotlib_font()
    random_effects = regression_result.metadata.get("level3_random_effects", {})
    rows = [(str(group), float(effect)) for group, effect in random_effects.items()]
    if not rows:
        raise ValueError("No level 3 random effects are available for visualization.")
    rows.sort(key=lambda item: item[1])
    groups = [item[0] for item in rows]
    estimates = np.asarray([item[1] for item in rows], dtype=float)
    positions = np.arange(len(rows))
    figure_height = max(4.0, 0.28 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(7.0, figure_height))
    axis.scatter(estimates, positions)
    axis.axvline(0)
    axis.set_yticks(positions)
    axis.set_yticklabels(groups)
    axis.set_xlabel("Level 3 random intercept estimate")
    axis.set_ylabel("Level 3 group")
    axis.set_title("Level 3 Random Intercept Estimates")
    _save_figure(figure, output_path)

def _plot_cross_level_interaction(result: RegressionResult, output_path: Path) -> None:
    metadata = result.metadata.get("cross_level_interaction") or {}
    slopes = metadata.get("conditional_effects") or []
    if not slopes:
        raise ValueError("교차수준 상호작용 시각화에 필요한 조건부 효과가 없습니다.")
    fitted = result.raw_result
    predictor_term = metadata["predictor_term"]
    moderator_term = metadata["moderator_term"]

    x_index = fitted.model.exog_names.index(predictor_term)
    x_values = np.asarray(fitted.model.exog[:, x_index], dtype=float)
    grid = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), 100)
    intercept = float(fitted.fe_params.get("const", 0.0))
    moderator_main = float(fitted.fe_params[moderator_term])
    figure, axis = plt.subplots(figsize=(8, 5))
    for slope in slopes:
        z = float(slope["moderator_value"])
        y = intercept + moderator_main * z + float(slope["estimate"]) * grid
        axis.plot(grid, y, label=str(slope["label"]))
    axis.set_xlabel(str(metadata.get("predictor", predictor_term)))
    axis.set_ylabel(result.dependent_variable)
    axis.set_title("교차수준 상호작용 조건부 예측선")
    axis.legend(title=str(metadata.get("moderator", moderator_term)))
    _save_figure(figure, output_path)


def _plot_three_level_variance_partition(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    partition = regression_result.fit_statistics.get("variance_partition") or {}
    level2_vpc = regression_result.fit_statistics.get("level2_vpc")
    level3_vpc = regression_result.fit_statistics.get("level3_vpc")
    labels = ["Level 1", "Level 2", "Level 3"]
    if level2_vpc is not None and level3_vpc is not None:
        level2 = float(level2_vpc)
        level3 = float(level3_vpc)
        values = [max(0.0, 1.0 - level2 - level3), level2, level3]
    else:
        values = [
            float(partition.get("level1", 0.0)),
            float(partition.get("level2", 0.0)),
            float(partition.get("level3", 0.0)),
        ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(labels, values)
    axis.set_ylim(0.0, max(1.0, max(values) * 1.15 if values else 1.0))
    axis.set_ylabel("Variance proportion")
    axis.set_title("Three-Level Variance Partition")
    for bar, value in zip(bars, values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom"
        )
    figure.tight_layout()
    _save_figure(figure, output_path)


def build_regression_visualizations(
    regression_result: RegressionResult,
    *,
    output_directory: str | Path,
) -> RegressionVisualizationReport:
    """회귀모형 유형에 맞는 시각화를 생성한다."""

    if regression_result.raw_result is None:
        raise ValueError("시각화에 필요한 원본 회귀결과 객체가 없습니다.")

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files: list[str] = []
    warnings: list[str] = []

    forest_path = output_directory / "coefficient_forest.png"

    _plot_coefficient_forest(
        regression_result,
        forest_path,
    )

    output_files.append(str(forest_path))

    if regression_result.model_type in {
        "ols",
        "weighted_least_squares",
        "heckman_selection",
        "iv_2sls_regression",
        "quantile_regression",
        "regularized_regression",
        "robust_regression",
        "tobit_regression",
        "panel_fixed_effects",
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
    }:
        residual_path = output_directory / "residuals_vs_fitted.png"
        qq_path = output_directory / "residual_qq_plot.png"

        _plot_residuals_vs_fitted(
            regression_result,
            residual_path,
        )
        _plot_qq(
            regression_result,
            qq_path,
        )

        output_files.extend(
            [
                str(residual_path),
                str(qq_path),
            ]
        )

    if regression_result.model_type == "ols":
        influence_path = output_directory / "influence_plot.png"
        _plot_influence(
            regression_result,
            influence_path,
        )
        output_files.append(str(influence_path))
    elif regression_result.model_type == "inverse_gaussian_regression":
        observed_path = output_directory / "inverse_gaussian_observed_vs_predicted.png"
        _plot_observed_vs_predicted_continuous(regression_result, observed_path)
        output_files.append(str(observed_path))
    elif regression_result.model_type == "gamma_regression":
        observed_path = output_directory / "gamma_observed_vs_predicted.png"
        _plot_observed_vs_predicted_continuous(regression_result, observed_path)
        output_files.append(str(observed_path))
    elif regression_result.model_type == "fractional_logit":
        observed_path = output_directory / "fractional_observed_vs_predicted.png"
        _plot_observed_vs_predicted(regression_result, observed_path)
        output_files.append(str(observed_path))
    elif regression_result.model_type == "beta_regression":
        observed_path = output_directory / "beta_observed_vs_predicted.png"
        _plot_observed_vs_predicted(regression_result, observed_path)
        output_files.append(str(observed_path))
    elif regression_result.model_type in {
        "poisson",
        "negative_binomial",
        "generalized_poisson",
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
        "hurdle_poisson",
        "hurdle_negative_binomial",
    }:
        plot_name = (
            "zero_inflated_observed_vs_predicted.png"
            if regression_result.model_type.startswith("zero_inflated")
            else "count_observed_vs_predicted.png"
        )
        observed_path = output_directory / plot_name
        _plot_count_observed_vs_predicted(regression_result, observed_path)
        output_files.append(str(observed_path))
    elif regression_result.model_type in {"cox_proportional_hazards", "stratified_cox", "left_truncated_cox", "cause_specific_cox", "clustered_cox", "time_varying_cox"}:
        baseline_path = output_directory / "cox_baseline_survival.png"
        _plot_baseline_survival(regression_result, baseline_path)
        output_files.append(str(baseline_path))
    elif regression_result.model_type == "piecewise_exponential":
        survival_path = output_directory / "piecewise_exponential_survival_curve.png"
        _plot_piecewise_exponential_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type == "discrete_time_hazard":
        survival_path = output_directory / "discrete_time_hazard_survival_curve.png"
        _plot_discrete_time_hazard_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type == "exponential_aft":
        survival_path = output_directory / "exponential_aft_survival_curve.png"
        _plot_exponential_aft_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type == "loglogistic_aft":
        survival_path = output_directory / "loglogistic_aft_survival_curve.png"
        _plot_loglogistic_aft_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type == "lognormal_aft":
        survival_path = output_directory / "lognormal_aft_survival_curve.png"
        _plot_lognormal_aft_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type == "weibull_ph":
        survival_path = output_directory / "weibull_ph_survival_curve.png"
        _plot_weibull_ph_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type == "weibull_aft":
        survival_path = output_directory / "weibull_aft_survival_curve.png"
        _plot_weibull_aft_survival(regression_result, survival_path)
        output_files.append(str(survival_path))
    elif regression_result.model_type in {
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
        "mixed_poisson_random_intercept",
        "mixed_poisson_random_slope",
        "mixed_poisson_three_level",
        "mixed_negative_binomial_random_intercept",
        "mixed_negative_binomial_random_slope",
        "mixed_negative_binomial_three_level",
    }:
        random_intercepts_path = output_directory / "random_intercepts.png"
        _plot_random_intercepts(
            regression_result,
            random_intercepts_path,
        )
        output_files.append(str(random_intercepts_path))
        if regression_result.model_type in {
            "mixed_random_slope",
            "mixed_binary_logit_random_slope",
            "mixed_poisson_random_slope",
            "mixed_negative_binomial_random_slope",
        }:
            try:
                random_slopes_path = output_directory / "random_slopes.png"
                _plot_random_slopes(regression_result, random_slopes_path)
                output_files.append(str(random_slopes_path))
            except ValueError as error:
                warnings.append(str(error))
        if regression_result.model_type in {
            "mixed_three_level",
            "mixed_binary_logit_three_level",
            "mixed_poisson_three_level",
            "mixed_negative_binomial_three_level",
        }:
            if regression_result.metadata.get("level3_random_effects"):
                level3_path = output_directory / "level3_random_intercepts.png"
                _plot_level3_random_intercepts(regression_result, level3_path)
                output_files.append(str(level3_path))
            variance_path = output_directory / "three_level_variance_partition.png"
            _plot_three_level_variance_partition(regression_result, variance_path)
            output_files.append(str(variance_path))
        if regression_result.metadata.get("cross_level_interaction"):
            interaction_path = output_directory / "cross_level_interaction.png"
            _plot_cross_level_interaction(regression_result, interaction_path)
            output_files.append(str(interaction_path))
    else:
        warnings.append("잔차·Q-Q·영향력 도표는 현재 OLS 모형만 지원합니다.")

    return RegressionVisualizationReport(
        model_id=regression_result.model_id,
        model_type=regression_result.model_type,
        output_files=output_files,
        warnings=warnings,
        metadata={
            "figure_count": len(output_files),
        },
    )
