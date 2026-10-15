import runpy
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval.domain.scenario import Decision
from rollouteval.evaluation import (
    GroundednessCalibrationRunner,
    GroundednessReviewDecision,
    GroundednessJudgment,
    GraderStatus,
    HumanGroundednessReview,
    load_groundedness_calibration,
    render_groundedness_review,
)
from rollouteval.evaluation.grounding import GroundingContext


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "calibration" / "groundedness_v1.yaml"
REVIEW_PATH = ROOT / "calibration" / "groundedness_v1_review.md"


def _review(
    decision=GroundednessReviewDecision.APPROVE,
    reviewed_status=GraderStatus.PASS,
    **overrides,
):
    values = {
        "case_id": "observed-metrics-supported",
        "decision": decision,
        "reviewed_status": reviewed_status,
        "reviewer_rationale": "The observed metrics support the explanation.",
        "reviewer": "Reviewer One",
        "reviewed_at": "2026-10-15",
    }
    values.update(overrides)
    return HumanGroundednessReview(**values)


def test_review_decision_values_are_exact() -> None:
    assert [decision.value for decision in GroundednessReviewDecision] == [
        "APPROVE",
        "CHANGE",
        "FLAG",
    ]


def test_review_contract_represents_approve_change_and_flag() -> None:
    approve = _review()
    change = _review(
        decision=GroundednessReviewDecision.CHANGE,
        reviewed_status=GraderStatus.FAIL,
    )
    flag = _review(decision=GroundednessReviewDecision.FLAG, reviewed_status=None)

    assert approve.decision is GroundednessReviewDecision.APPROVE
    assert change.reviewed_status is GraderStatus.FAIL
    assert flag.reviewed_status is None


def test_review_contract_is_immutable() -> None:
    review = _review()
    with pytest.raises(ValidationError):
        review.reviewer = "Someone else"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "  "),
        ("reviewer_rationale", "\n"),
        ("reviewer", " "),
        ("reviewed_at", "\t"),
    ],
)
def test_completed_review_rejects_blank_required_fields(field, value) -> None:
    with pytest.raises(ValidationError):
        _review(**{field: value})


@pytest.mark.parametrize("decision", [GroundednessReviewDecision.APPROVE, GroundednessReviewDecision.CHANGE])
def test_approve_and_change_require_reviewed_label(decision) -> None:
    with pytest.raises(ValidationError, match="requires reviewed_status"):
        _review(decision=decision, reviewed_status=None)


def test_approve_must_match_candidate_expected_status() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    candidate = calibration.cases[0]

    assert _review().validate_against(candidate).reviewed_status is candidate.expected_status
    mismatched = _review(reviewed_status=GraderStatus.FAIL)
    with pytest.raises(ValueError, match="equal candidate expected_status"):
        mismatched.validate_against(candidate)


def test_review_case_id_must_match_candidate() -> None:
    candidate = load_groundedness_calibration(DATASET_PATH).cases[0]
    with pytest.raises(ValueError, match="case_id"):
        _review(case_id="different-case").validate_against(candidate)


def test_change_can_differ_and_flag_can_omit_reviewed_label() -> None:
    candidate = load_groundedness_calibration(DATASET_PATH).cases[0]
    change = _review(
        decision=GroundednessReviewDecision.CHANGE,
        reviewed_status=GraderStatus.FAIL,
    )
    flag = _review(decision=GroundednessReviewDecision.FLAG, reviewed_status=None)

    assert change.validate_against(candidate).reviewed_status is GraderStatus.FAIL
    assert flag.validate_against(candidate).reviewed_status is None


def test_renderer_shows_every_case_once_in_dataset_order() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)

    markdown = render_groundedness_review(calibration)

    headings = [f"## {index}. {case.case_id}" for index, case in enumerate(calibration.cases, 1)]
    assert all(markdown.count(heading) == 1 for heading in headings)
    assert [markdown.index(heading) for heading in headings] == sorted(
        markdown.index(heading) for heading in headings
    )
    assert len(calibration.cases) == 15


def test_renderer_includes_case_fields_evidence_and_candidate_labels() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    markdown = render_groundedness_review(calibration)

    for case in calibration.cases:
        context = case.context
        assert case.case_id in markdown
        assert case.description in markdown
        assert context.task in markdown
        assert context.decision.value in markdown
        assert context.explanation in markdown
        assert f"Proposed label:** `{case.expected_status.value}`" in markdown
        assert case.rationale in markdown
        for item in context.evidence:
            assert f"Interaction index: {item.interaction_index}" in markdown
            assert item.evidence_id.value in markdown
            assert item.source_tool in markdown
            assert item.status.value in markdown
            assert item.content in markdown


def test_renderer_preserves_evidence_order_and_raw_contents() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    markdown = render_groundedness_review(calibration)
    case = next(case for case in calibration.cases if case.case_id == "evidence-evolved-to-healthy")
    content_positions = [markdown.index(item.content) for item in case.context.evidence]

    assert content_positions == sorted(content_positions)
    assert "{\"available\":false,\"metrics\":null,\"message\":\"temporary telemetry outage\"}" in markdown
    assert "{\"available\":true,\"metrics\":{" in markdown


def test_renderer_leaves_all_human_review_fields_blank_and_unselected() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    markdown = render_groundedness_review(calibration)

    assert markdown.count("Decision: [ ] APPROVE  [ ] CHANGE  [ ] FLAG") == 15
    assert markdown.count("- Reviewed label:\n") == 15
    assert markdown.count("- Reviewer rationale:\n") == 15
    assert markdown.count("- Reviewer:\n") == 15
    assert markdown.count("- Review date:\n") == 15
    assert "human_reviewed=true" not in markdown
    assert "reviewer=Codex" not in markdown
    assert "[x]" not in markdown.lower()
    assert "Reviewed label: `" not in markdown


def test_renderer_is_deterministic_and_adds_no_timestamp_or_random_id() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)

    first = render_groundedness_review(calibration)
    second = render_groundedness_review(calibration)

    assert first == second
    assert "Generated at" not in first
    assert "uuid" not in first.lower()


def test_rendering_leaves_context_and_candidate_labels_unchanged() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    original = tuple(
        (case.expected_status, case.context.model_dump(mode="python"))
        for case in calibration.cases
    )

    render_groundedness_review(calibration)

    assert tuple(
        (case.expected_status, case.context.model_dump(mode="python"))
        for case in calibration.cases
    ) == original


def test_review_metadata_is_separate_from_context_and_cannot_be_sent_to_judge() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    candidate = calibration.cases[0]
    review = _review().validate_against(candidate)
    fields = set(GroundingContext.model_fields)

    assert not {
        "reviewer",
        "reviewed_at",
        "reviewed_status",
        "reviewer_rationale",
    }.intersection(fields)
    assert "review_decision" not in fields
    assert review.case_id == candidate.case_id
    assert candidate.context is not review


class RecordingJudge:
    def __init__(self):
        self.contexts = []

    def judge(self, context):
        self.contexts.append(context)
        return GroundednessJudgment(status=GraderStatus.PASS, reason="scripted")


def test_existing_calibration_runner_only_receives_context_after_review_record_exists() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    _review().validate_against(calibration.cases[0])
    judge = RecordingJudge()

    GroundednessCalibrationRunner(judge).run(
        type(calibration)(calibration_id=calibration.calibration_id, cases=(calibration.cases[0],))
    )

    assert judge.contexts == [calibration.cases[0].context]
    assert type(judge.contexts[0]) is GroundingContext


def test_script_generates_expected_review_packet_offline(monkeypatch) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("review rendering must not perform network access")

    monkeypatch.setattr(socket, "create_connection", no_network)
    runpy.run_path(str(ROOT / "scripts" / "render_groundedness_review.py"), run_name="__main__")

    assert REVIEW_PATH.is_file()
    markdown = REVIEW_PATH.read_text(encoding="utf-8")
    assert markdown.count("## ") == 15
    assert "- Decision: [ ] APPROVE  [ ] CHANGE  [ ] FLAG" in markdown


def test_render_script_loads_dataset_with_existing_loader() -> None:
    script = (ROOT / "scripts" / "render_groundedness_review.py").read_text(encoding="utf-8")
    assert "load_groundedness_calibration(dataset_path)" in script


def test_review_packet_generation_does_not_modify_source_yaml() -> None:
    before = DATASET_PATH.read_bytes()
    calibration = load_groundedness_calibration(DATASET_PATH)

    render_groundedness_review(calibration)

    assert DATASET_PATH.read_bytes() == before


def test_renderer_preserves_decision_as_declared_without_relabeling_candidate() -> None:
    calibration = load_groundedness_calibration(DATASET_PATH)
    candidate = next(case for case in calibration.cases if case.case_id == "wrong-action-grounded-facts")
    markdown = render_groundedness_review(calibration)

    assert candidate.context.decision is Decision.ROLLBACK
    assert candidate.expected_status is GraderStatus.PASS
    assert "Declared decision:** rollback" in markdown
    assert "Proposed label:** `PASS`" in markdown
