"""Variable Evidence Resolver 테스트."""

import pandas as pd

from src.preprocess.detector import detect_variable_level
from src.preprocess.evidence_resolver import (
    VariableEvidence,
    evidence_from_dataframe,
    resolution_summary,
    resolve_all_variable_evidence,
    resolve_variable_evidence,
)


def test_questionnaire_and_codebook_confirm_level() -> None:
    detection = detect_variable_level(
        "satisfaction",
        pd.Series([1, 2, 3, 4, 5]),
    )
    evidence = VariableEvidence(
        variable_name="satisfaction",
        questionnaire_level="리커트",
        codebook_level="scale_item",
    )

    result = resolve_variable_evidence(
        detection,
        evidence,
    )

    assert result.status == "confirmed"
    assert result.resolved_level == "scale_item"
    assert result.confidence == 0.98


def test_external_sources_conflict() -> None:
    detection = detect_variable_level(
        "group",
        pd.Series([1, 2, 1, 2]),
    )
    evidence = VariableEvidence(
        variable_name="group",
        questionnaire_level="binary",
        codebook_level="nominal",
    )

    result = resolve_variable_evidence(
        detection,
        evidence,
    )

    assert result.status == "conflict"
    assert result.resolved_level == "unknown"
    assert result.conflicts


def test_detection_conflict_requires_review() -> None:
    detection = detect_variable_level(
        "education",
        pd.Series([1, 2, 3, 4]),
    )
    evidence = VariableEvidence(
        variable_name="education",
        questionnaire_level="nominal",
    )

    result = resolve_variable_evidence(
        detection,
        evidence,
    )

    assert result.status == "review_required"
    assert result.resolved_level == "nominal"
    assert result.conflicts


def test_no_external_evidence_stays_review_required() -> None:
    detection = detect_variable_level(
        "income",
        pd.Series([10.2, 20.5, 30.1]),
    )
    evidence = VariableEvidence(
        variable_name="income",
    )

    result = resolve_variable_evidence(
        detection,
        evidence,
    )

    assert result.status == "review_required"
    assert result.resolved_level == "continuous"


def test_resolve_all_variables() -> None:
    detections = [
        detect_variable_level(
            "outcome",
            pd.Series([0, 1, 0, 1]),
        ),
        detect_variable_level(
            "score",
            pd.Series([1.1, 2.2, 3.3]),
        ),
    ]
    evidences = [
        VariableEvidence(
            variable_name="outcome",
            codebook_level="binary",
        )
    ]

    results = resolve_all_variable_evidence(
        detections,
        evidences,
    )

    assert len(results) == 2
    assert results[0].resolved_level == "binary"
    assert results[1].status == "review_required"


def test_evidence_dataframe_conversion() -> None:
    dataframe = pd.DataFrame(
        {
            "variable_name": ["q1"],
            "questionnaire_level": ["리커트"],
            "source_files": ["survey.pdf | codebook.xlsx"],
        }
    )

    evidences = evidence_from_dataframe(dataframe)

    assert evidences[0].questionnaire_level == "리커트"
    assert evidences[0].source_files == [
        "survey.pdf",
        "codebook.xlsx",
    ]


def test_resolution_summary() -> None:
    detections = [
        detect_variable_level(
            "binary_var",
            pd.Series([0, 1, 0]),
        ),
        detect_variable_level(
            "score",
            pd.Series([1.1, 2.2, 3.3]),
        ),
    ]
    evidences = [
        VariableEvidence(
            variable_name="binary_var",
            questionnaire_level="binary",
        ),
        VariableEvidence(
            variable_name="score",
            questionnaire_level="continuous",
        ),
    ]

    results = resolve_all_variable_evidence(
        detections,
        evidences,
    )
    summary = resolution_summary(results)

    assert summary["variable_count"] == 2
    assert summary["status_counts"]["confirmed"] == 2
