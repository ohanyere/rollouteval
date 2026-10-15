import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rollouteval.domain.scenario import Decision
from rollouteval.evaluation import (
    GroundednessCalibrationCase,
    GroundednessCalibrationReport,
    GroundednessCalibrationResult,
    GroundednessCalibrationRunner,
    GroundednessCalibrationSet,
    GroundednessJudgment,
    GraderStatus,
    load_groundedness_calibration,
)
from rollouteval.evaluation.grounding import GroundingContext, GroundingEvidenceItem


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "calibration" / "groundedness_v1.yaml"


def _context(content: str = '{"phase":"Progressing","candidate_weight":20}'):
    return GroundingContext(
        task="Assess the rollout using observed evidence.",
        decision=Decision.PAUSE,
        explanation="The rollout is currently progressing.",
        evidence=(
            GroundingEvidenceItem(
                interaction_index=0,
                evidence_id="rollout_state",
                source_tool="get_rollout_state",
                status="OBSERVED",
                content=content,
            ),
        ),
    )


def _case(case_id="case-a", expected=GraderStatus.PASS, context=None):
    return GroundednessCalibrationCase(
        case_id=case_id,
        description="A reviewable test case.",
        context=context or _context(),
        expected_status=expected,
        rationale="The narrow factual claim is supported by the observation.",
    )


class ScriptedJudge:
    """Offline test double that returns queued labels without judging semantics."""

    def __init__(self, judgments, error_at: int | None = None, error=None):
        self.judgments = tuple(judgments)
        self.error_at = error_at
        self.error = error
        self.contexts = []

    def judge(self, context):
        self.contexts.append(context)
        index = len(self.contexts) - 1
        if self.error is not None and index == self.error_at:
            raise self.error
        return self.judgments[index]


def test_calibration_case_accepts_context_and_existing_status() -> None:
    case = _case()
    assert case.expected_status is GraderStatus.PASS
    assert case.context == _context()


@pytest.mark.parametrize("field", ["case_id", "description", "rationale"])
def test_calibration_case_rejects_blank_text(field: str) -> None:
    values = {
        "case_id": "id",
        "description": "description",
        "context": _context(),
        "expected_status": GraderStatus.PASS,
        "rationale": "rationale",
    }
    values[field] = " \n"
    with pytest.raises(ValidationError):
        GroundednessCalibrationCase(**values)


def test_calibration_case_is_immutable() -> None:
    case = _case()
    with pytest.raises(ValidationError):
        case.expected_status = GraderStatus.FAIL


def test_calibration_set_preserves_order_and_is_immutable_copy_safe() -> None:
    cases = [_case("first"), _case("second", GraderStatus.FAIL)]
    calibration = GroundednessCalibrationSet(calibration_id="set-v1", cases=cases)
    cases.reverse()

    assert [case.case_id for case in calibration.cases] == ["first", "second"]
    with pytest.raises(ValidationError):
        calibration.cases = ()


@pytest.mark.parametrize("calibration_id", ["", "  ", "\t"])
def test_calibration_set_rejects_blank_id(calibration_id: str) -> None:
    with pytest.raises(ValidationError):
        GroundednessCalibrationSet(calibration_id=calibration_id, cases=[_case()])


def test_calibration_set_rejects_duplicate_ids_and_empty_cases() -> None:
    with pytest.raises(ValidationError, match="unique"):
        GroundednessCalibrationSet(
            calibration_id="set",
            cases=[_case("duplicate"), _case("duplicate", GraderStatus.FAIL)],
        )
    with pytest.raises(ValidationError):
        GroundednessCalibrationSet(calibration_id="set", cases=[])


def test_yaml_dataset_loads_in_file_order_and_is_balanced_reviewable_candidates() -> None:
    dataset = load_groundedness_calibration(DATASET)
    raw = yaml.safe_load(DATASET.read_text(encoding="utf-8"))

    assert dataset.calibration_id == "groundedness-v1-candidates"
    assert [case.case_id for case in dataset.cases] == [
        item["case_id"] for item in raw["cases"]
    ]
    assert 12 <= len(dataset.cases) <= 16
    labels = [case.expected_status for case in dataset.cases]
    assert labels.count(GraderStatus.PASS) >= 2
    assert labels.count(GraderStatus.FAIL) >= 2
    assert GraderStatus.INCONCLUSIVE in labels
    assert all(case.rationale.strip() for case in dataset.cases)
    assert all(isinstance(case.context, GroundingContext) for case in dataset.cases)


def test_yaml_dataset_contains_no_rollout_answer_key_fields() -> None:
    raw = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    forbidden = {
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
    }

    def keys_recursively(value):
        if isinstance(value, dict):
            yield from value.keys()
            for nested in value.values():
                yield from keys_recursively(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys_recursively(nested)

    assert not forbidden.intersection(keys_recursively(raw))


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("observed-metrics-supported", GraderStatus.PASS),
        ("exact-metrics-unobserved", GraderStatus.FAIL),
        ("metric-claim-contradicted", GraderStatus.FAIL),
        ("unavailable-metrics-acknowledged", GraderStatus.PASS),
        ("metrics-invented-when-unavailable", GraderStatus.FAIL),
        ("observation-error-acknowledged", GraderStatus.PASS),
        ("successful-metrics-called-error", GraderStatus.FAIL),
        ("limited-claim-with-missing-metrics", GraderStatus.PASS),
        ("wrong-action-grounded-facts", GraderStatus.PASS),
        ("unsupported-database-causality", GraderStatus.FAIL),
        ("runtime-instability-observed", GraderStatus.PASS),
        ("runtime-events-unobserved", GraderStatus.FAIL),
        ("evidence-evolved-to-healthy", GraderStatus.PASS),
        ("evidence-evolution-misrepresented", GraderStatus.FAIL),
        ("ambiguous-meaningful-instability", GraderStatus.INCONCLUSIVE),
    ],
)
def test_dataset_contains_required_semantic_case(case_id, expected) -> None:
    dataset = load_groundedness_calibration(DATASET)
    cases = {case.case_id: case for case in dataset.cases}
    assert cases[case_id].expected_status is expected


def test_yaml_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    data = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    data["cases"].append(copy.deepcopy(data["cases"][0]))
    malformed = tmp_path / "duplicate.yaml"
    malformed.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError, match="unique"):
        load_groundedness_calibration(malformed)


def test_yaml_loader_rejects_invalid_grounding_context(tmp_path: Path) -> None:
    data = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    data["cases"][0]["context"]["decision"] = "delete_everything"
    malformed = tmp_path / "invalid-context.yaml"
    malformed.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_groundedness_calibration(malformed)


def test_runner_calls_judge_once_with_context_only_and_preserves_case_order() -> None:
    cases = (_case("first"), _case("second", GraderStatus.FAIL))
    calibration = GroundednessCalibrationSet(calibration_id="set", cases=cases)
    judge = ScriptedJudge(
        [
            GroundednessJudgment(status=GraderStatus.PASS, reason="First reason."),
            GroundednessJudgment(status=GraderStatus.PASS, reason="Second reason."),
        ]
    )

    report = GroundednessCalibrationRunner(judge).run(calibration)

    assert len(judge.contexts) == len(cases)
    assert all(context is case.context for context, case in zip(judge.contexts, cases))
    assert all(type(context) is GroundingContext for context in judge.contexts)
    assert [result.case_id for result in report.results] == ["first", "second"]
    assert [result.judge_reason for result in report.results] == ["First reason.", "Second reason."]
    for context in judge.contexts:
        assert not {"expected_status", "rationale", "case_id", "description"}.intersection(
            type(context).model_fields
        )


def test_runner_derives_exact_matches_and_report_counts() -> None:
    calibration = GroundednessCalibrationSet(
        calibration_id="set",
        cases=(_case("yes", GraderStatus.PASS), _case("no", GraderStatus.FAIL)),
    )
    judge = ScriptedJudge(
        [
            GroundednessJudgment(status=GraderStatus.PASS, reason="match"),
            GroundednessJudgment(status=GraderStatus.INCONCLUSIVE, reason="mismatch"),
        ]
    )

    report = GroundednessCalibrationRunner(judge).run(calibration)

    assert [result.matched for result in report.results] == [True, False]
    assert (report.total_cases, report.matched_cases, report.mismatched_cases) == (2, 1, 1)
    assert report.exact_match_rate == 0.5
    assert calibration.cases[1].expected_status is GraderStatus.FAIL


def test_scripted_full_agreement_is_reported_without_quality_claim() -> None:
    calibration = load_groundedness_calibration(DATASET)
    judge = ScriptedJudge(
        [
            GroundednessJudgment(status=case.expected_status, reason="scripted label")
            for case in calibration.cases
        ]
    )

    report = GroundednessCalibrationRunner(judge).run(calibration)

    assert report.total_cases == len(calibration.cases)
    assert report.matched_cases == len(calibration.cases)
    assert report.mismatched_cases == 0
    assert report.exact_match_rate == 1.0


def test_one_scripted_disagreement_produces_one_mismatch() -> None:
    calibration = load_groundedness_calibration(DATASET)
    labels = [case.expected_status for case in calibration.cases]
    labels[3] = GraderStatus.FAIL
    judge = ScriptedJudge(
        [GroundednessJudgment(status=label, reason="scripted") for label in labels]
    )

    report = GroundednessCalibrationRunner(judge).run(calibration)

    assert report.mismatched_cases == 1
    assert report.results[3].matched is False
    assert calibration.cases[3].expected_status is GraderStatus.PASS


def test_judge_exception_propagates_without_retry_or_conversion() -> None:
    cases = (_case("first"), _case("second"), _case("third"))
    calibration = GroundednessCalibrationSet(calibration_id="set", cases=cases)
    error = RuntimeError("scripted judge failure")
    judge = ScriptedJudge(
        [GroundednessJudgment(status=GraderStatus.PASS, reason="first")],
        error_at=1,
        error=error,
    )

    with pytest.raises(RuntimeError) as caught:
        GroundednessCalibrationRunner(judge).run(calibration)

    assert caught.value is error
    assert len(judge.contexts) == 2


def test_calibration_run_does_not_mutate_set_or_context() -> None:
    context = _context()
    case = _case(context=context)
    calibration = GroundednessCalibrationSet(calibration_id="set", cases=(case,))
    before = context.model_dump(mode="python")
    judge = ScriptedJudge(
        [GroundednessJudgment(status=GraderStatus.PASS, reason="Supported.")]
    )

    GroundednessCalibrationRunner(judge).run(calibration)

    assert context.model_dump(mode="python") == before
    assert calibration.cases[0] is case
    assert judge.contexts[0] is context


def test_calibration_result_and_report_are_immutable() -> None:
    result = GroundednessCalibrationResult(
        case_id="case",
        expected_status=GraderStatus.PASS,
        actual_status=GraderStatus.FAIL,
        judge_reason="Different label.",
    )
    report = GroundednessCalibrationReport(calibration_id="set", results=(result,))

    with pytest.raises(ValidationError):
        result.actual_status = GraderStatus.PASS
    with pytest.raises(ValidationError):
        report.results = ()


def test_calibration_runner_uses_only_the_scripted_test_double() -> None:
    judge = ScriptedJudge(
        [GroundednessJudgment(status=GraderStatus.PASS, reason="scripted")]
    )
    report = GroundednessCalibrationRunner(judge).run(
        GroundednessCalibrationSet(calibration_id="set", cases=(_case(),))
    )

    assert len(judge.contexts) == 1
    assert report.results[0].actual_status is GraderStatus.PASS
