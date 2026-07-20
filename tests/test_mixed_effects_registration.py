"""Random Intercept 선택기와 회귀 빌더 등록 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.regression_step import RegressionAnalysisStep
from src.statistics.regression.selector import fit_regression_by_level
from tests.support.builders import build_regression_pipeline


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    groups = np.repeat(np.arange(12), 6)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(scale=1.0, size=12)
    y = 1.0 + 1.5 * x + random_intercepts[groups] + rng.normal(scale=0.4, size=len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def make_multilevel_plan(*, group_variable: str | None = "group") -> AnalysisPlan:
    options: dict[str, object] = {
        "reml": False,
        "optimizer": "lbfgs",
        "max_iterations": 200,
    }
    if group_variable is not None:
        options["group_variable"] = group_variable

    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {
                    "enabled": True,
                    "options": options,
                },
            },
        }
    )


def make_variable_map(*, include_group: bool = True) -> VariableMap:
    variables = {
        "y": {"role": "dependent", "measurement_level": "continuous"},
        "x": {"role": "independent", "measurement_level": "continuous"},
    }
    if include_group:
        variables["group"] = {"role": "cluster", "measurement_level": "nominal"}
    return VariableMap.model_validate({"variables": variables})


def test_selector_runs_explicit_random_intercept() -> None:
    result = fit_regression_by_level(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="continuous",
        model_type="mixed_random_intercept",
        group_variable="group",
        mixed_effects_options={"optimizer": "lbfgs", "max_iterations": 200},
    )

    assert result.model_type == "mixed_random_intercept"
    assert result.metadata["group_variable"] == "group"
    assert result.converged is True


def test_selector_requires_group_variable_for_random_intercept() -> None:
    with pytest.raises(ValueError, match="그룹변수"):
        fit_regression_by_level(
            make_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            measurement_level="continuous",
            model_type="mixed_random_intercept",
        )


def test_builder_registers_random_intercept_analysis_and_diagnostics(tmp_path: Path) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=make_multilevel_plan(),
        variable_map=make_variable_map(),
        project_name="혼합효과 연구",
    )

    assert registration.registered is True
    assert registration.model_type == "mixed_random_intercept"
    assert registration.group_variable == "group"
    assert registration.diagnostics_registered is True
    assert registration.robustness_registered is True
    assert registration.effect_size_registered is True
    assert registration.reporting_registered is True
    assert registration.visualization_registered is True
    assert registration.audit_registered is True
    assert orchestrator.registry.names() == [
        "09_regression_analysis",
        "10_regression_diagnostics",
        "11_robustness_analysis",
        "12_advanced_robustness",
        "13_effect_size_analysis",
        "14_regression_reporting",
        "15_regression_visualization",
        "16_research_audit",
    ]

    step = orchestrator.registry.get("09_regression_analysis")
    assert isinstance(step, RegressionAnalysisStep)
    assert step.model_type == "mixed_random_intercept"
    assert step.group_variable == "group"
    assert step.mixed_effects_options["optimizer"] == "lbfgs"


def test_builder_uses_first_cluster_as_group_fallback(tmp_path: Path) -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "clusters": ["group"],
            },
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {"enabled": True},
            },
        }
    )

    _, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=plan,
        variable_map=make_variable_map(),
        project_name="혼합효과 연구",
    )

    assert registration.registered is True
    assert registration.group_variable == "group"


def test_builder_rejects_missing_group_variable(tmp_path: Path) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=make_multilevel_plan(group_variable=None),
        variable_map=make_variable_map(),
        project_name="혼합효과 연구",
    )

    assert registration.registered is False
    assert "그룹변수가 지정되지 않았습니다" in registration.warnings[0]
    assert orchestrator.registry.names() == []


def test_builder_rejects_undefined_group_variable(tmp_path: Path) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=make_multilevel_plan(),
        variable_map=make_variable_map(include_group=False),
        project_name="혼합효과 연구",
    )

    assert registration.registered is False
    assert "variable_map 정의가 없습니다" in registration.warnings[0]
    assert orchestrator.registry.names() == []
