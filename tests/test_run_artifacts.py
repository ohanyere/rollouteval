import json
import shutil

import pytest

from rollouteval.agent import AgentFinalAnswer, AgentRunResult, AgentTask
from rollouteval.domain.scenario import (
    Decision,
    InitialEnvironment,
    RolloutState,
)
from rollouteval.evaluation import DeterministicEvaluation, GraderResult, GraderStatus
from rollouteval.observability.trace import RunTrace
from rollouteval.orchestration import ObservableRun
from rollouteval.run_artifacts import (
    InvalidRunIdentityError,
    RunArtifactConflictError,
    RunArtifactCorruptError,
    RunArtifactEnvelope,
    RunArtifactSecurityError,
    RunArtifactStore,
    RunArtifactWriteError,
    RunReportService,
    UnsupportedRunArtifactVersionError,
)


def _run(
    run_id="artifact-run-1",
    *,
    groundedness_status=GraderStatus.PASS,
    objective_status=GraderStatus.PASS,
    reason="Explanation supported by observed metrics.",
    judge_error=None,
):
    evaluation = DeterministicEvaluation(
        decision=GraderResult(
            grader="decision",
            status=objective_status,
            reason="Expected decision mismatch." if objective_status is GraderStatus.FAIL else "Decision matched.",
        ),
        final_state=GraderResult(grader="final_state", status=GraderStatus.PASS, reason="Final state matched."),
        evidence_sufficiency=GraderResult(grader="evidence_sufficiency", status=GraderStatus.PASS, reason="Evidence observed."),
        trajectory=GraderResult(grader="trajectory", status=GraderStatus.PASS, reason="Trajectory valid."),
        safety=GraderResult(grader="safety", status=GraderStatus.PASS, reason="No safety violations."),
        abstention=GraderResult(grader="abstention", status=GraderStatus.PASS, reason="Abstention matched."),
        groundedness=GraderResult(
            grader="groundedness",
            status=groundedness_status,
            reason=reason,
            evaluator_version="1",
            criterion_id="groundedness",
            evidence_references=("service_metrics@0", "runtime_events@2"),
            judge_provider="groq",
            judge_model="openai/gpt-oss-120b",
            prompt_sha256="a" * 64,
            judge_error=judge_error,
        ),
    )
    return ObservableRun(
        agent_run=AgentRunResult(
            final_answer=AgentFinalAnswer(
                kind="final",
                decision=Decision.PROMOTE,
                explanation="Observed metrics support promotion.",
            ),
            interactions=(),
            steps=1,
        ),
        task=AgentTask(task="Evaluate the canary rollout."),
        runtime_snapshot=InitialEnvironment(
            rollout=RolloutState(
                stable_version="v1",
                candidate_version="v2",
                candidate_weight=50,
                phase="Progressing",
            ),
            metrics=None,
            events=(),
        ),
        trace=RunTrace(run_id=run_id, scenario_id="canary-check", events=()),
        evaluation=evaluation,
    )


def _artifact_path(root, run_id="artifact-run-1"):
    return next(path for path in root.glob("run-*.json") if json.loads(path.read_text())["run_id"] == run_id)


def test_normal_run_artifact_save_load_is_semantically_lossless(tmp_path):
    run = _run()
    store = RunArtifactStore(tmp_path / "runs")

    path = store.save(run)
    restored = store.load(run.trace.run_id)

    assert path.is_file()
    assert restored == run
    envelope = RunArtifactEnvelope.model_validate_json(path.read_text())
    assert envelope.artifact_schema_version == 1
    assert envelope.rollouteval_version
    assert envelope.run_id == run.trace.run_id
    assert envelope.scenario_id == run.trace.scenario_id
    assert envelope.task == run.task.task
    assert envelope.created_at.tzinfo is not None


def test_groundedness_pass_provenance_and_report_round_trip(tmp_path):
    run = _run()
    store = RunArtifactStore(tmp_path)
    store.save(run)
    report = RunReportService(store).get(run.trace.run_id)
    grounded = next(item for item in report.criteria if item.criterion_id == "groundedness")

    assert report.run_id == run.trace.run_id
    assert report.scenario_id == run.trace.scenario_id
    assert report.task == run.task.task
    assert grounded.status is GraderStatus.PASS
    assert grounded.rationale == "Explanation supported by observed metrics."
    assert grounded.evaluator_name == "groundedness"
    assert grounded.evaluator_version == "1"
    assert grounded.evidence_references == ("service_metrics@0", "runtime_events@2")
    assert grounded.provider == "groq"
    assert grounded.model == "openai/gpt-oss-120b"
    assert grounded.prompt_sha256 == "a" * 64
    assert grounded.judge_error is None


def test_groundedness_fail_round_trip_is_visible_in_report(tmp_path):
    run = _run(
        groundedness_status=GraderStatus.FAIL,
        reason="The unsupported database outage claim is not in observed evidence.",
    )
    store = RunArtifactStore(tmp_path)
    store.save(run)

    report = RunReportService(store).get(run.trace.run_id)
    grounded = next(item for item in report.criteria if item.criterion_id == "groundedness")

    assert grounded.status is GraderStatus.FAIL
    assert "unsupported database outage" in grounded.rationale
    assert grounded.evidence_references == ("service_metrics@0", "runtime_events@2")


def test_inconclusive_provider_error_round_trip_has_no_credentials(tmp_path):
    run = _run(
        groundedness_status=GraderStatus.INCONCLUSIVE,
        reason="Judge provider failed safely.",
        judge_error="TimeoutError",
    )
    store = RunArtifactStore(tmp_path)
    path = store.save(run)
    report_json = RunReportService(store).get(run.trace.run_id).model_dump_json()
    artifact_json = path.read_text()

    assert store.load(run.trace.run_id).evaluation.groundedness.judge_error == "TimeoutError"
    assert "TimeoutError" in report_json
    assert "GROQ_API_KEY" not in artifact_json + report_json
    assert "OPENAI_API_KEY" not in artifact_json + report_json
    assert "Authorization" not in artifact_json + report_json
    assert "Bearer " not in artifact_json + report_json


def test_objective_failure_and_groundedness_pass_remain_distinct_in_report(tmp_path):
    run = _run(objective_status=GraderStatus.FAIL, groundedness_status=GraderStatus.PASS)
    store = RunArtifactStore(tmp_path)
    store.save(run)

    report = RunReportService(store).get(run.trace.run_id)
    by_name = {item.evaluator_name: item for item in report.criteria}

    assert by_name["decision"].status is GraderStatus.FAIL
    assert by_name["groundedness"].status is GraderStatus.PASS
    assert not hasattr(report, "overall_status")
    assert len(report.criteria) == 7


@pytest.mark.parametrize(
    ("mutate", "error_type"),
    [
        (lambda text: "{broken", RunArtifactCorruptError),
        (
            lambda text: json.dumps({**json.loads(text), "artifact_schema_version": 999}),
            UnsupportedRunArtifactVersionError,
        ),
        (
            lambda text: json.dumps(
                {
                    **json.loads(text),
                    "observable_run": {
                        key: value
                        for key, value in json.loads(text)["observable_run"].items()
                        if key != "evaluation"
                    },
                }
            ),
            RunArtifactCorruptError,
        ),
        (
            lambda text: json.dumps(
                {
                    **json.loads(text),
                    "observable_run": {
                        **json.loads(text)["observable_run"],
                        "evaluation": {"decision": {"status": "MAYBE"}},
                    },
                }
            ),
            RunArtifactCorruptError,
        ),
    ],
)
def test_corrupt_or_unsupported_artifacts_fail_without_fabricated_results(
    tmp_path, mutate, error_type
):
    store = RunArtifactStore(tmp_path)
    path = store.save(_run())
    path.write_text(mutate(path.read_text()), encoding="utf-8")

    with pytest.raises(error_type):
        store.load("artifact-run-1")


def test_discovery_is_ordered_ignores_unrelated_files_and_rejects_duplicate_identity(tmp_path):
    store = RunArtifactStore(tmp_path)
    store.save(_run("z-run"))
    store.save(_run("a-run"))
    (tmp_path / "notes.json").write_text("not a run", encoding="utf-8")

    assert [run.trace.run_id for run in store.list()] == ["a-run", "z-run"]
    report_service = RunReportService(store)
    assert [report.run_id for report in report_service.list()] == ["a-run", "z-run"]

    artifact = _artifact_path(tmp_path, "a-run")
    duplicate = tmp_path / f"run-{'f' * 64}.json"
    shutil.copyfile(artifact, duplicate)
    with pytest.raises(RunArtifactCorruptError, match="filename"):
        store.list()


def test_identity_is_confined_and_traversal_is_rejected(tmp_path):
    store = RunArtifactStore(tmp_path / "runs")
    with pytest.raises(InvalidRunIdentityError):
        store.load("../outside")
    with pytest.raises(InvalidRunIdentityError):
        store.save(_run("../outside"))
    assert not (tmp_path / "outside.json").exists()


def test_writes_are_atomic_and_leave_no_partial_final_or_temp_file(tmp_path, monkeypatch):
    store = RunArtifactStore(tmp_path)

    def fail_link(source, destination):
        raise OSError("simulated atomic link failure")

    monkeypatch.setattr("rollouteval.run_artifacts.os.link", fail_link)
    with pytest.raises(RunArtifactWriteError, match="written atomically"):
        store.save(_run())

    assert not tuple(tmp_path.glob("run-*.json"))
    assert not tuple(tmp_path.glob(".run-artifact-*.tmp"))


def test_identical_save_is_idempotent_and_different_run_conflicts(tmp_path):
    store = RunArtifactStore(tmp_path)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("leave unchanged", encoding="utf-8")
    run = _run()
    first_path = store.save(run)

    assert store.save(run) == first_path
    changed = _run(reason="Changed rationale with same run identity.")
    with pytest.raises(RunArtifactConflictError):
        store.save(changed)
    assert store.load(run.trace.run_id) == run
    assert unrelated.read_text(encoding="utf-8") == "leave unchanged"


@pytest.mark.parametrize(
    "reason",
    [
        "OPENAI_API_KEY=gsk_test_secret_value",
        "Authorization: Bearer dummy_bearer_token_123456",
        '"Authorization": "Basic dummy_basic_credential_123456"',
        "provider returned sk-proj-fakecredential123456",
    ],
)
def test_credential_like_content_is_rejected_before_any_artifact_is_written(
    tmp_path, reason
):
    store = RunArtifactStore(tmp_path)
    unsafe = _run(reason=reason)

    with pytest.raises(RunArtifactSecurityError):
        store.save(unsafe)
    assert not tuple(tmp_path.glob("run-*.json"))


def test_load_rejects_symlink_artifact(tmp_path):
    store = RunArtifactStore(tmp_path)
    destination = store.save(_run())
    outside = tmp_path / "outside.json"
    outside.write_text(destination.read_text(), encoding="utf-8")
    destination.unlink()
    destination.symlink_to(outside)

    with pytest.raises(RunArtifactCorruptError, match="regular file"):
        store.load("artifact-run-1")
