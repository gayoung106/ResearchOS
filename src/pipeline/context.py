"""연구 프로젝트 전체 상태를 관리하는 ResearchContext."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ResearchContext:
    """연구 프로젝트의 핵심 정보를 보관한다."""

    project_name: str
    research_topic: str = ""
    research_questions: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)

    raw_data_files: list[str] = field(default_factory=list)
    questionnaire_files: list[str] = field(default_factory=list)
    codebook_files: list[str] = field(default_factory=list)

    dependent_variables: list[str] = field(default_factory=list)
    independent_variables: list[str] = field(default_factory=list)
    mediator_variables: list[str] = field(default_factory=list)
    moderator_variables: list[str] = field(default_factory=list)
    control_variables: list[str] = field(default_factory=list)

    analysis_plan: dict[str, Any] = field(default_factory=dict)
    variable_map: dict[str, Any] = field(default_factory=dict)

    completed_stages: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review_comments: list[dict[str, Any]] = field(default_factory=list)
    revision_history: list[dict[str, Any]] = field(default_factory=list)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def mark_stage_completed(self, stage_name: str) -> None:
        """분석 단계를 완료 상태로 기록한다."""
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
        self.touch()

    def add_generated_file(self, file_path: str | Path) -> None:
        """생성된 결과 파일을 기록한다."""
        normalized = str(Path(file_path))
        if normalized not in self.generated_files:
            self.generated_files.append(normalized)
        self.touch()

    def add_warning(self, message: str) -> None:
        """경고 메시지를 기록한다."""
        self.warnings.append(message)
        self.touch()

    def add_review_comment(
        self,
        role: str,
        comment: str,
        *,
        severity: str = "normal",
    ) -> None:
        """지도교수 또는 심사위원 검토 의견을 기록한다."""
        self.review_comments.append(
            {
                "role": role,
                "comment": comment,
                "severity": severity,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.touch()

    def add_revision(
        self,
        description: str,
        *,
        stage: str | None = None,
    ) -> None:
        """수정 이력을 기록한다."""
        self.revision_history.append(
            {
                "description": description,
                "stage": stage,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.touch()

    def touch(self) -> None:
        """최종 수정시각을 갱신한다."""
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        """컨텍스트를 딕셔너리로 변환한다."""
        return asdict(self)

    def save_yaml(self, output_path: str | Path) -> Path:
        """현재 연구 컨텍스트를 YAML 파일로 저장한다."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.to_dict(),
                file,
                allow_unicode=True,
                sort_keys=False,
            )

        return path

    @classmethod
    def load_yaml(cls, input_path: str | Path) -> ResearchContext:
        """YAML 파일에서 연구 컨텍스트를 불러온다."""
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"ResearchContext 파일이 없습니다: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        return cls(**data)
