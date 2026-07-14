"""오탈자 및 수치 일관성 검토 인터페이스."""

from __future__ import annotations


def proofread_outputs() -> dict[str, object]:
    """표, 본문, 변수명 및 수치의 일관성을 점검한다."""
    raise NotImplementedError
