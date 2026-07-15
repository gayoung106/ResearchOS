"""파이프라인 테스트에서 사용하는 공통 assertion helper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class RegistryProtocol(Protocol):
    """테스트에서 필요한 registry 인터페이스."""

    def names(self) -> list[str]:
        """등록된 파이프라인 단계명을 반환한다."""


class OrchestratorProtocol(Protocol):
    """테스트에서 필요한 orchestrator 인터페이스."""

    registry: RegistryProtocol


def get_registry_names(
    orchestrator: OrchestratorProtocol,
) -> list[str]:
    """orchestrator에 등록된 파이프라인 단계명을 반환한다."""
    return list(orchestrator.registry.names())


def assert_registry_matches(
    orchestrator: OrchestratorProtocol,
    expected: Iterable[str],
) -> None:
    """registry 전체 순서가 expected와 일치하는지 확인한다."""
    actual_names = get_registry_names(orchestrator)
    expected_names = list(expected)

    assert actual_names == expected_names, (
        "파이프라인 registry가 예상과 다릅니다.\n"
        f"expected: {expected_names}\n"
        f"actual:   {actual_names}"
    )


def assert_steps_registered(
    orchestrator: OrchestratorProtocol,
    *expected_steps: str,
) -> None:
    """지정한 단계들이 registry에 등록됐는지 확인한다."""
    actual_names = get_registry_names(orchestrator)

    missing_steps = [step_name for step_name in expected_steps if step_name not in actual_names]

    assert not missing_steps, (
        "등록되지 않은 파이프라인 단계가 있습니다.\n"
        f"missing: {missing_steps}\n"
        f"actual:  {actual_names}"
    )


def assert_steps_not_registered(
    orchestrator: OrchestratorProtocol,
    *unexpected_steps: str,
) -> None:
    """지정한 단계들이 registry에 등록되지 않았는지 확인한다."""
    actual_names = get_registry_names(orchestrator)

    registered_steps = [step_name for step_name in unexpected_steps if step_name in actual_names]

    assert not registered_steps, (
        "등록되면 안 되는 파이프라인 단계가 있습니다.\n"
        f"registered: {registered_steps}\n"
        f"actual:     {actual_names}"
    )


def assert_step_order(
    orchestrator: OrchestratorProtocol,
    *,
    before: str,
    after: str,
) -> None:
    """before 단계가 after 단계보다 먼저 등록됐는지 확인한다."""
    actual_names = get_registry_names(orchestrator)

    assert before in actual_names, f"{before!r} 단계가 registry에 없습니다.\nactual: {actual_names}"
    assert after in actual_names, f"{after!r} 단계가 registry에 없습니다.\nactual: {actual_names}"

    before_index = actual_names.index(before)
    after_index = actual_names.index(after)

    assert before_index < after_index, (
        "파이프라인 단계 순서가 예상과 다릅니다.\n"
        f"expected order: {before!r} -> {after!r}\n"
        f"actual:         {actual_names}"
    )
