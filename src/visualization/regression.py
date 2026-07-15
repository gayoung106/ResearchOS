"""회귀분석 결과 시각화 생성."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    fitted = regression_result.raw_result
    fitted_values = np.asarray(fitted.fittedvalues)
    residuals = np.asarray(fitted.resid)

    figure, axis = plt.subplots()
    axis.scatter(fitted_values, residuals)
    axis.axhline(0)
    axis.set_xlabel("적합값")
    axis.set_ylabel("잔차")
    axis.set_title("잔차-적합값 도표")

    _save_figure(figure, output_path)


def _plot_qq(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
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

    _save_figure(figure, output_path)


def _plot_influence(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    fitted = regression_result.raw_result
    influence = fitted.get_influence()

    leverage = np.asarray(influence.hat_matrix_diag)
    studentized = np.asarray(influence.resid_studentized_external)
    cooks_distance = np.asarray(influence.cooks_distance[0])

    marker_sizes = 20 + 200 * (cooks_distance / max(cooks_distance.max(), 1e-12))

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

    _save_figure(figure, output_path)


def _plot_coefficient_forest(
    regression_result: RegressionResult,
    output_path: Path,
) -> None:
    coefficients = [
        coefficient
        for coefficient in regression_result.coefficients
        if coefficient.term.lower()
        not in {
            "const",
            "intercept",
        }
        and "/" not in coefficient.term
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

    if regression_result.model_type == "ols":
        residual_path = output_directory / "residuals_vs_fitted.png"
        qq_path = output_directory / "residual_qq_plot.png"
        influence_path = output_directory / "influence_plot.png"

        _plot_residuals_vs_fitted(
            regression_result,
            residual_path,
        )
        _plot_qq(
            regression_result,
            qq_path,
        )
        _plot_influence(
            regression_result,
            influence_path,
        )

        output_files.extend(
            [
                str(residual_path),
                str(qq_path),
                str(influence_path),
            ]
        )
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
