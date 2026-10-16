import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rollouteval.evaluation import (
    GroundednessCalibrationRunner,
    GroundednessJudgment,
    GroundednessReviewDecision,
    GroundingContext,
    GraderStatus,
    HumanReviewedGroundednessReferenceSet,
    load_groundedness_calibration,
    load_human_groundedness_reviews,
    load_human_reviewed_groundedness_reference,
)
from rollouteval.evaluation.providers.openai import _serialize_context


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "calibration" / "groundedness_v1.yaml"
REVIEWS_PATH = ROOT / "calibration" / "groundedness_v1_reviews.yaml"
_CALIBRATION_SHA256 = "256762a6fb7f4c00101108c583faa8884614b8edcba5cac18aa3a737134c5007"


class RecordingJudge:
    def __init__(self):
        self.contexts = []

    def judge(self, context):
        self.contexts.append(context)
        return GroundednessJudgment(status=GraderStatus.PASS, reason="scripted only")


def _review_data():
    return yaml.safe_load(REVIEWS_PATH.read_text(encoding="utf-8"))


def _write_reviews(tmp_path: Path, data) -> Path:
    path = tmp_path / "reviews.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_review_yaml_loads_15_approved_records_matching_candidate_distribution() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reviews = load_human_groundedness_reviews(REVIEWS_PATH, calibration)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)

    assert len(reviews) == len(reference.reviews) == len(calibration.cases) == 15
    assert all(review.decision is GroundednessReviewDecision.APPROVE for review in reviews)
    assert all(review.reviewer == "human-reviewer" for review in reviews)
    assert all(review.reviewer.strip() for review in reviews)
    assert all(
        review.reviewed_at == (
            "2026-10-16" if review.case_id == "evidence-evolved-to-healthy" else "2026-10-15"
        )
        for review in reviews
    )
    assert all(review.reviewer_rationale.strip() for review in reviews)
    statuses = [review.reviewed_status for review in reviews]
    assert statuses.count(GraderStatus.PASS) == 7
    assert statuses.count(GraderStatus.FAIL) == 7
    assert statuses.count(GraderStatus.INCONCLUSIVE) == 1
    assert [review.case_id for review in reviews] == [case.case_id for case in calibration.cases]
    assert all(
        review.reviewed_status is case.expected_status
        for review, case in zip(reviews, calibration.cases)
    )


def test_each_human_approval_validates_against_its_candidate() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reviews = load_human_groundedness_reviews(REVIEWS_PATH, calibration)
    candidates = {case.case_id: case for case in calibration.cases}

    assert all(review.validate_against(candidates[review.case_id]) is review for review in reviews)


def test_review_loader_preserves_review_file_order(tmp_path: Path) -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    data = _review_data()
    data["reviews"].reverse()
    reversed_path = tmp_path / "reviews.yaml"
    reversed_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    reviews = load_human_groundedness_reviews(reversed_path, calibration)

    assert [review.case_id for review in reviews] == [
        review["case_id"] for review in data["reviews"]
    ]


def test_reference_set_is_immutable_ordered_and_supports_status_lookup() -> None:
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    calibration = load_groundedness_calibration(CALIBRATION_PATH)

    assert [review.case_id for review in reference.reviews] == [case.case_id for case in calibration.cases]
    assert reference.reviewed_status_for("observed-metrics-supported") is GraderStatus.PASS
    assert reference.reviewed_status_for("exact-metrics-unobserved") is GraderStatus.FAIL
    assert reference.reviewed_status_for("ambiguous-meaningful-instability") is GraderStatus.INCONCLUSIVE
    with pytest.raises(KeyError):
        reference.reviewed_status_for("unknown")
    with pytest.raises(ValidationError):
        reference.reviews = ()


def test_reference_set_rejects_non_approved_duplicate_and_empty_reviews() -> None:
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    with pytest.raises(ValidationError, match="APPROVE or CHANGE"):
        HumanReviewedGroundednessReferenceSet(
            calibration_id=reference.calibration_id,
            reviews=(reference.reviews[0].model_copy(update={"decision": GroundednessReviewDecision.FLAG}),),
        )
    with pytest.raises(ValidationError, match="unique"):
        HumanReviewedGroundednessReferenceSet(
            calibration_id=reference.calibration_id,
            reviews=(reference.reviews[0], reference.reviews[0]),
        )
    with pytest.raises(ValidationError):
        HumanReviewedGroundednessReferenceSet(calibration_id="set", reviews=())


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "missing", "mismatch"])
def test_review_loader_rejects_inconsistent_review_file(tmp_path: Path, mutation: str) -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    data = _review_data()
    if mutation == "duplicate":
        data["reviews"].append(dict(data["reviews"][0]))
    elif mutation == "unknown":
        data["reviews"][0]["case_id"] = "not-a-calibration-case"
    elif mutation == "missing":
        data["reviews"].pop()
    else:
        data["reviews"][0]["reviewed_status"] = "FAIL"
    path = _write_reviews(tmp_path, data)

    with pytest.raises((ValueError, ValidationError)):
        load_human_groundedness_reviews(path, calibration)


def test_review_loader_rejects_calibration_id_mismatch(tmp_path: Path) -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    data = _review_data()
    data["calibration_id"] = "different-set"
    path = _write_reviews(tmp_path, data)

    with pytest.raises(ValueError, match="calibration_id"):
        load_human_groundedness_reviews(path, calibration)


def test_context_and_openai_serialization_contain_no_review_metadata() -> None:
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    context = calibration.cases[0].context
    payload = _serialize_context(context)
    forbidden = {"reviewed_status", "reviewer", "reviewer_rationale", "reviewed_at"}

    assert set(GroundingContext.model_fields) == {"task", "decision", "explanation", "evidence"}
    assert not forbidden.intersection(payload)
    assert not forbidden.intersection(json.dumps(payload))
    assert reference.reviewed_status_for(calibration.cases[0].case_id) is GraderStatus.PASS


def test_calibration_judge_receives_only_context_not_reference_metadata() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    judge = RecordingJudge()

    GroundednessCalibrationRunner(judge).run(calibration)

    assert len(judge.contexts) == len(calibration.cases)
    assert all(type(context) is GroundingContext for context in judge.contexts)
    assert all(context is case.context for context, case in zip(judge.contexts, calibration.cases))
    for context in judge.contexts:
        serialized = context.model_dump_json()
        assert "reviewed_status" not in serialized
        assert "reviewer" not in serialized
        assert "reviewer_rationale" not in serialized
        assert "reviewed_at" not in serialized
    assert len(reference.reviews) == 15


def test_candidate_yaml_remains_unchanged() -> None:
    assert hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest() == _CALIBRATION_SHA256


def test_combined_reference_loader_does_not_mutate_calibration_cases() -> None:
    calibration_before = load_groundedness_calibration(CALIBRATION_PATH)
    before = tuple(case.model_dump(mode="json") for case in calibration_before.cases)

    load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)

    calibration_after = load_groundedness_calibration(CALIBRATION_PATH)
    assert tuple(case.model_dump(mode="json") for case in calibration_after.cases) == before
