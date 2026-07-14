"""파일 로딩과 측정수준 근거 통합을 위한 파이프라인 단계."""

from __future__ import annotations

from pathlib import Path

from src.common.config_models import VariableMap
from src.common.file_metadata import build_variable_metadata
from src.common.file_reader import find_data_files, read_data_file
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.preprocess.evidence_resolver import (
    VariableEvidence,
    resolve_all_variable_evidence,
    resolved_levels_to_dataframe,
)


class DataLoadingStep(PipelineStep):
    """rawdata 폴더에서 분석대상 파일을 읽는 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        *,
        source_file: str | Path | None = None,
        order: int = 10,
    ) -> None:
        super().__init__(
            name="01_data_loading",
            order=order,
            required=True,
        )
        self.runtime = runtime
        self.source_file = Path(source_file) if source_file else None

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        if self.source_file is not None:
            source_path = self.source_file
            if not source_path.is_absolute():
                source_path = working_directory / source_path
        else:
            rawdata_dir = working_directory / "rawdata"
            files = find_data_files(rawdata_dir)

            if not files:
                raise FileNotFoundError(
                    f"rawdata 폴더에 지원되는 데이터 파일이 없습니다: {rawdata_dir}"
                )

            if len(files) > 1:
                raise ValueError(
                    "분석대상 파일이 여러 개입니다. source_file을 명시하세요: "
                    + ", ".join(str(path.name) for path in files)
                )

            source_path = files[0]

        read_result = read_data_file(source_path)
        self.runtime.dataframe = read_result.dataframe
        self.runtime.variable_metadata = build_variable_metadata(
            read_result.dataframe,
            source_metadata=read_result.metadata,
        )
        self.runtime.set_artifact("read_result", read_result)

        output_dir = working_directory / "result" / "01_import"
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = output_dir / "analysis_base.parquet"
        metadata_path = output_dir / "variable_metadata.xlsx"

        read_result.dataframe.to_parquet(
            parquet_path,
            index=False,
        )
        self.runtime.variable_metadata.to_excel(
            metadata_path,
            index=False,
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[
                str(parquet_path),
                str(metadata_path),
            ],
            metadata={
                "source_file": str(source_path),
                "row_count": len(read_result.dataframe),
                "column_count": len(read_result.dataframe.columns),
            },
        )


class EvidenceResolutionStep(PipelineStep):
    """자동 탐지 결과와 variable_map 근거를 통합하는 단계."""

    def __init__(
        self,
        runtime: PipelineRuntime,
        variable_map: VariableMap,
        *,
        order: int = 25,
    ) -> None:
        super().__init__(
            name="02_evidence_resolution",
            order=order,
            required=True,
        )
        self.runtime = runtime
        self.variable_map = variable_map

    def should_run(self, context: ResearchContext) -> bool:
        return bool(self.runtime.detections)

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        evidences: list[VariableEvidence] = []

        for variable_name, definition in self.variable_map.variables.items():
            evidence = definition.evidence or {}

            evidences.append(
                VariableEvidence(
                    variable_name=variable_name,
                    questionnaire_level=(
                        evidence.get("questionnaire_level") or evidence.get("questionnaire")
                    ),
                    codebook_level=(evidence.get("codebook_level") or evidence.get("codebook")),
                    value_label_level=(
                        evidence.get("value_label_level") or evidence.get("value_labels")
                    ),
                    questionnaire_text=definition.question_text or None,
                    codebook_note=definition.notes or None,
                    value_labels=(
                        definition.coding if isinstance(definition.coding, dict) else None
                    ),
                    source_files=[],
                )
            )

        results = resolve_all_variable_evidence(
            self.runtime.detections,
            evidences,
        )
        self.runtime.resolved_levels = results

        output_dir = working_directory / "result" / "02_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "resolved_variable_levels.xlsx"

        resolved_levels_to_dataframe(results).to_excel(
            output_path,
            index=False,
        )

        warnings = [
            f"{result.variable_name}: {result.status}"
            for result in results
            if result.status in {"review_required", "conflict"}
        ]

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(output_path)],
            warnings=warnings,
            metadata={
                "resolved_count": len(results),
                "conflict_count": sum(result.status == "conflict" for result in results),
            },
        )
