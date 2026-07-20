from __future__ import annotations

from pathlib import Path

from src.pipeline.advanced_robustness_step import AdvancedMixedEffectsRobustnessStep
from tests.support.builders import build_regression_pipeline
from tests.test_mixed_effects_robustness_registration import make_plan, make_variable_map


def test_registers_mixed_advanced_robustness_when_enabled(tmp_path: Path) -> None:
    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=make_plan(),
        variable_map=make_variable_map(),
        project_name="Mixed advanced robustness",
    )
    assert registration.advanced_robustness_registered is True
    step = orchestrator.registry.get("12_advanced_robustness")
    assert isinstance(step, AdvancedMixedEffectsRobustnessStep)
