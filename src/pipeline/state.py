"""분석 파이프라인 실행상태 모델."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class StageStatus(StrEnum):
    """파이프라인 단계 상태."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class StageRecord:
    """개별 분석단계 실행기록."""

    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    output_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineState:
    """전체 파이프라인 진행상태."""

    stages: dict[str, StageRecord] = field(default_factory=dict)
    active_stage: str | None = None

    def register_stage(self, stage_name: str) -> None:
        """분석단계를 등록한다."""
        self.stages.setdefault(stage_name, StageRecord(name=stage_name))

    def start_stage(self, stage_name: str) -> None:
        """분석단계를 실행중 상태로 변경한다."""
        self.register_stage(stage_name)
        record = self.stages[stage_name]
        record.status = StageStatus.RUNNING
        record.started_at = datetime.now().isoformat(timespec="seconds")
        record.error_message = None
        self.active_stage = stage_name

    def complete_stage(
        self,
        stage_name: str,
        *,
        output_files: list[str] | None = None,
    ) -> None:
        """분석단계를 완료 상태로 변경한다."""
        self.register_stage(stage_name)
        record = self.stages[stage_name]
        record.status = StageStatus.COMPLETED
        record.completed_at = datetime.now().isoformat(timespec="seconds")
        if output_files:
            record.output_files.extend(output_files)
        self.active_stage = None

    def fail_stage(self, stage_name: str, error_message: str) -> None:
        """분석단계를 실패 상태로 변경한다."""
        self.register_stage(stage_name)
        record = self.stages[stage_name]
        record.status = StageStatus.FAILED
        record.completed_at = datetime.now().isoformat(timespec="seconds")
        record.error_message = error_message
        self.active_stage = None

    def skip_stage(self, stage_name: str) -> None:
        """분석단계를 생략 상태로 변경한다."""
        self.register_stage(stage_name)
        record = self.stages[stage_name]
        record.status = StageStatus.SKIPPED
        record.completed_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        """파이프라인 상태를 직렬화 가능한 딕셔너리로 변환한다."""
        return {
            "active_stage": self.active_stage,
            "stages": {
                name: {
                    **asdict(record),
                    "status": record.status.value,
                }
                for name, record in self.stages.items()
            },
        }

    def save_json(self, output_path: str | Path) -> Path:
        """파이프라인 상태를 JSON 파일로 저장한다."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_json(cls, input_path: str | Path) -> PipelineState:
        """JSON 파일에서 파이프라인 상태를 불러온다."""
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"PipelineState 파일이 없습니다: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        state = cls(active_stage=data.get("active_stage"))

        for name, record_data in data.get("stages", {}).items():
            state.stages[name] = StageRecord(
                name=name,
                status=StageStatus(record_data["status"]),
                started_at=record_data.get("started_at"),
                completed_at=record_data.get("completed_at"),
                error_message=record_data.get("error_message"),
                output_files=record_data.get("output_files", []),
            )

        return state
