"""공통 pytest fixture."""

import pytest

from src.common.config_models import AnalysisPlan, VariableMap
from tests.support.builders import (
    make_binary_variable_map,
    make_continuous_variable_map,
    make_empty_analysis_plan,
    make_empty_variable_map,
    make_ordinal_variable_map,
)


@pytest.fixture
def empty_analysis_plan() -> AnalysisPlan:
    """빈 분석 계획을 제공한다."""
    return make_empty_analysis_plan()


@pytest.fixture
def ols_with_robustness_analysis_plan() -> AnalysisPlan:
    """OLS와 강건성 분석이 활성화된 계획을 제공한다."""
    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
                "robustness": {
                    "enabled": True,
                },
            },
        }
    )


@pytest.fixture
def ols_without_robustness_analysis_plan() -> AnalysisPlan:
    """OLS는 활성화하고 강건성 분석은 비활성화한 계획을 제공한다."""
    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
                "robustness": {
                    "enabled": False,
                },
            },
        }
    )


@pytest.fixture
def advanced_robustness_disabled_analysis_plan() -> AnalysisPlan:
    """기본 강건성 분석은 활성화하고 고급 강건성은 비활성화한 계획을 제공한다."""
    return AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
                "robustness": {
                    "enabled": True,
                    "options": {
                        "advanced_enabled": False,
                    },
                },
            },
        }
    )


@pytest.fixture
def empty_variable_map() -> VariableMap:
    """빈 variable map을 제공한다."""
    return make_empty_variable_map()


@pytest.fixture
def continuous_variable_map() -> VariableMap:
    """연속형 회귀 테스트용 variable map을 제공한다."""
    return make_continuous_variable_map()


@pytest.fixture
def binary_variable_map() -> VariableMap:
    """이항 로짓 테스트용 variable map을 제공한다."""
    return make_binary_variable_map()


@pytest.fixture
def ordinal_variable_map() -> VariableMap:
    """순서형 로짓 테스트용 variable map을 제공한다."""
    return make_ordinal_variable_map()
