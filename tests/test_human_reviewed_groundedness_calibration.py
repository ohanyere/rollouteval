import json
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval.domain.scenario import Decision
from rollouteval.evaluation import (
    DeterministicEvaluation,
    GroundednessCalibrationCase,
    GroundednessCalibrationReport,
    GroundednessCalibrationResult,
    GroundednessCalibrationSet,
    GroundednessJudgment,
    GroundednessReviewDecision,
    GroundingContext,
    GraderStatus,
    HumanGroundednessReview,
    HumanReviewedGroundednessCalibrationRunner,
    HumanReviewedGroundednessReferenceSet,
    load_groundedness_calibration,
    load_human_reviewed_groundedness_reference,
)
from rollouteval.evaluation.providers.openai import _serialize_context
from rollouteval.orchestration import ObservableRun

ROOT = Path(__file__).resolve().parents[1]
_CLI_SPEC = importlib.util.spec_from_file_location(
    "run_groundedness_calibration",
    ROOT / "scripts" / "run_groundedness_calibration.py",
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
_CLI_MODULE = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(_CLI_MODULE)
build_parser = _CLI_MODULE.build_parser
format_groundedness_calibration_report = _CLI_MODULE.format_groundedness_calibration_report
CALIBRATION_PATH = ROOT / "calibration" / "groundedness_v1.yaml"
REVIEWS_PATH = ROOT / "calibration" / "groundedness_v1_reviews.yaml"


class ScriptedJudge:
    def __init__(self, statuses, *, error_at=None, error=None):
        self.statuses = tuple(statuses)
        self.error_at = error_at
        self.error = error
        self.contexts = []

    def judge(self, context):
        self.contexts.append(context)
        index = len(self.contexts) - 1
        if self.error is not None and index == self.error_at:
            raise self.error
        status = self.statuses[index]
        return GroundednessJudgment(status=status, reason=f"Reason for call {index + 1}.")


def _reference_review(case_id: str, status: GraderStatus, decision=GroundednessReviewDecision.APPROVE):
    return HumanGroundednessReview(
        case_id=case_id,
        decision=decision,
        reviewed_status=status,
        reviewer_rationale="Reviewed candidate explanation and evidence.",
        reviewer="test-reviewer",
        reviewed_at="2026-10-15",
    )


def _synthetic_sets():
    context = GroundingContext(
        task="Evaluate the observed rollout.",
        decision=Decision.PAUSE,
        explanation="The rollout is progressing.",
        evidence=(),
    )
    case = GroundednessCalibrationCase(
        case_id="synthetic-change",
        description="Synthetic answer-key authority test.",
        context=context,
        expected_status=GraderStatus.PASS,
        rationale="Candidate label intentionally differs from reviewed status.",
    )
    calibration = GroundednessCalibrationSet(calibration_id="synthetic-set", cases=(case,))
    review = _reference_review(
        case.case_id,
        GraderStatus.FAIL,
        decision=GroundednessReviewDecision.CHANGE,
    )
    reference = HumanReviewedGroundednessReferenceSet(
        calibration_id=calibration.calibration_id,
        reviews=(review,),
    )
    return calibration, reference


def test_reviewed_runner_accepts_protocol_and_canonical_sets() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    judge = ScriptedJudge([GraderStatus.PASS] * len(calibration.cases))

    report = HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert isinstance(report, GroundednessCalibrationReport)
    assert len(judge.contexts) == len(calibration.cases)


def test_synthetic_change_review_overrides_candidate_expected_status() -> None:
    calibration, reference = _synthetic_sets()
    judge = ScriptedJudge([GraderStatus.FAIL])

    report = HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    result = report.results[0]
    assert calibration.cases[0].expected_status is GraderStatus.PASS
    assert reference.reviewed_status_for("synthetic-change") is GraderStatus.FAIL
    assert result.expected_status is GraderStatus.FAIL
    assert result.actual_status is GraderStatus.FAIL
    assert result.matched is True


@pytest.mark.parametrize("status", list(GraderStatus))
def test_reviewed_runner_matches_each_reference_status(status: GraderStatus) -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    case = calibration.cases[0].model_copy(update={"expected_status": status})
    synthetic = GroundednessCalibrationSet(calibration_id="status-set", cases=(case,))
    reference = HumanReviewedGroundednessReferenceSet(
        calibration_id="status-set",
        reviews=(_reference_review(case.case_id, status),),
    )

    report = HumanReviewedGroundednessCalibrationRunner(ScriptedJudge([status])).run(
        synthetic,
        reference,
    )

    assert report.results[0].expected_status is status
    assert report.results[0].actual_status is status
    assert report.results[0].matched


def test_runner_calls_once_per_case_with_only_context_in_dataset_order() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    judge = ScriptedJudge([review.reviewed_status for review in reference.reviews])

    HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert len(judge.contexts) == 15
    assert all(type(context) is GroundingContext for context in judge.contexts)
    assert all(context is case.context for context, case in zip(judge.contexts, calibration.cases))


def test_reviewed_reference_distribution_and_full_agreement() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    statuses = [reference.reviewed_status_for(case.case_id) for case in calibration.cases]
    judge = ScriptedJudge(statuses)

    report = HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert len(reference.reviews) == report.total_cases == 15
    assert statuses.count(GraderStatus.PASS) == 7
    assert statuses.count(GraderStatus.FAIL) == 7
    assert statuses.count(GraderStatus.INCONCLUSIVE) == 1
    assert report.matched_cases == 15
    assert report.mismatched_cases == 0
    assert report.exact_match_rate == 1.0
    assert report.mismatches == ()
    assert [
        (item.status, item.total, item.matched)
        for item in report.status_agreement
    ] == [
        (GraderStatus.PASS, 7, 7),
        (GraderStatus.FAIL, 7, 7),
        (GraderStatus.INCONCLUSIVE, 1, 1),
    ]


def test_one_wrong_status_identifies_ordered_mismatch_and_status_counts() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    statuses = [review.reviewed_status for review in reference.reviews]
    statuses[3] = GraderStatus.FAIL if statuses[3] is not GraderStatus.FAIL else GraderStatus.PASS
    judge = ScriptedJudge(statuses)

    report = HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert report.matched_cases == 14
    assert report.mismatched_cases == 1
    assert report.exact_match_rate == pytest.approx(14 / 15)
    assert [result.case_id for result in report.mismatches] == [calibration.cases[3].case_id]
    assert report.status_agreement[0].status is GraderStatus.PASS
    assert report.status_agreement[0].total == 7
    assert report.status_agreement[0].matched == 6


def test_report_preserves_exact_judge_reasons_and_reference_statuses() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    judge = ScriptedJudge([GraderStatus.FAIL] * len(calibration.cases))

    report = HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert report.results[0].judge_reason == "Reason for call 1."
    assert report.results[0].expected_status is reference.reviewed_status_for(report.results[0].case_id)


def test_judge_exception_propagates_without_retry_or_fabricated_results() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    error = RuntimeError("scripted failure")
    judge = ScriptedJudge([GraderStatus.PASS] * 15, error_at=2, error=error)

    with pytest.raises(RuntimeError) as caught:
        HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert caught.value is error
    assert len(judge.contexts) == 3


def test_runner_rejects_reference_set_mismatch_before_judge_calls() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    other_reference = reference.model_copy(update={"calibration_id": "other"})
    judge = ScriptedJudge([GraderStatus.PASS] * 15)

    with pytest.raises(ValueError, match="IDs do not match"):
        HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, other_reference)
    assert judge.contexts == []


def test_reviewed_runner_does_not_mutate_inputs() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    case_snapshot = tuple(case.model_dump(mode="json") for case in calibration.cases)
    review_snapshot = tuple(review.model_dump(mode="json") for review in reference.reviews)
    judge = ScriptedJudge([GraderStatus.PASS] * 15)

    HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    assert tuple(case.model_dump(mode="json") for case in calibration.cases) == case_snapshot
    assert tuple(review.model_dump(mode="json") for review in reference.reviews) == review_snapshot


def test_answer_key_metadata_never_reaches_judge_or_context_serialization() -> None:
    calibration = load_groundedness_calibration(CALIBRATION_PATH)
    reference = load_human_reviewed_groundedness_reference(CALIBRATION_PATH, REVIEWS_PATH)
    judge = ScriptedJudge([review.reviewed_status for review in reference.reviews])

    HumanReviewedGroundednessCalibrationRunner(judge).run(calibration, reference)

    forbidden = (
        "reviewed_status",
        "reviewer",
        "reviewer_rationale",
        "reviewed_at",
        "expected_status",
        "case_id",
        "description",
        "rationale",
    )
    for case, context in zip(calibration.cases, judge.contexts):
        assert context is case.context
        serialized = context.model_dump_json()
        assert all(field not in serialized for field in forbidden)
        payload = json.dumps(_serialize_context(context))
        assert all(field not in payload for field in forbidden)


def test_cli_parser_requires_model_and_has_no_default() -> None:
    parser = build_parser()
    model_action = next(action for action in parser._actions if action.dest == "model")

    assert model_action.required is True
    assert model_action.default is None
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(["--model", "test-model"]).model == "test-model"


def test_cli_references_canonical_files_without_credential_inspection() -> None:
    source = (ROOT / "scripts" / "run_groundedness_calibration.py").read_text(encoding="utf-8")

    assert '"groundedness_v1.yaml"' in source
    assert '"groundedness_v1_reviews.yaml"' in source
    assert "OPENAI_API_KEY" not in source
    assert "os.environ" not in source


def test_cli_formatter_contains_summary_and_case_diagnostics() -> None:
    report = GroundednessCalibrationReport(
        calibration_id="reviewed-set",
        results=(
            GroundednessCalibrationResult(
                case_id="case-match",
                expected_status=GraderStatus.PASS,
                actual_status=GraderStatus.PASS,
                judge_reason="Supported by supplied metrics.",
            ),
            GroundednessCalibrationResult(
                case_id="case-mismatch",
                expected_status=GraderStatus.FAIL,
                actual_status=GraderStatus.PASS,
                judge_reason="Unsupported claim appears grounded.",
            ),
        ),
    )

    output = format_groundedness_calibration_report(report, model="caller-model")

    assert "Calibration ID: reviewed-set" in output
    assert "Judge model: caller-model" in output
    assert "Total cases: 2" in output
    assert "Matched: 1" in output
    assert "Mismatched: 1" in output
    assert "Exact agreement: 0.500000" in output
    assert "case-match\nReference: PASS\nJudge: PASS\nResult: MATCH" in output
    assert "case-mismatch\nReference: FAIL\nJudge: PASS\nResult: MISMATCH" in output
    assert "Reason: Unsupported claim appears grounded." in output
    assert "OPENAI_API_KEY" not in output


def test_cli_source_never_formats_credentials_or_provider_objects() -> None:
    source = (ROOT / "scripts" / "run_groundedness_calibration.py").read_text(encoding="utf-8")
    assert "headers" not in source.lower()
    assert "client.__dict__" not in source


def test_existing_artifact_contracts_remain_separate() -> None:
    assert set(DeterministicEvaluation.model_fields) == {
        "decision",
        "final_state",
        "evidence_sufficiency",
        "trajectory",
        "safety",
        "abstention",
    }
    assert set(ObservableRun.model_fields) == {
        "agent_run",
        "runtime_snapshot",
        "trace",
        "evaluation",
        "cost_estimate",
    }
    assert "groundedness" not in DeterministicEvaluation.model_fields
    assert "reviewed_status" not in GroundingContext.model_fields
