import json
from pathlib import Path

import pytest

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentRunResult,
    AgentToolCall,
    AgentToolInteraction,
    AgentToolName,
)
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import (
    AbstentionGrader,
    DecisionGrader,
    EvidenceSufficiencyGrader,
    FinalStateGrader,
    GraderStatus,
    SafetyGrader,
    TrajectoryGrader,
)
from rollouteval.scenarios import load_scenario


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_IDS = (
    "degraded-canary",
    "healthy-canary",
    "insufficient-evidence",
    "transient-anomaly",
    "conflicting-signals",
    "telemetry-unavailable",
)
EVIDENCE_TO_TOOL = {
    "rollout_state": AgentToolName.GET_ROLLOUT_STATE,
    "service_metrics": AgentToolName.GET_SERVICE_METRICS,
    "runtime_events": AgentToolName.GET_RECENT_EVENTS,
}


def _scenario(scenario_id: str):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _success(tool: AgentToolName, result: str = "{}") -> AgentToolInteraction:
    return AgentToolInteraction.successful(_call(tool), result)


def _run(
    decision: Decision,
    *interactions: AgentToolInteraction,
) -> AgentRunResult:
    return AgentRunResult(
        final_answer=AgentFinalAnswer(
            kind="final", decision=decision, explanation="Visible test explanation."
        ),
        interactions=tuple(interactions),
        steps=len(interactions) + 1,
    )


def _run_with_all_evidence(scenario_id: str, decision: Decision) -> AgentRunResult:
    scenario = _scenario(scenario_id)
    snapshot = scenario.initial_environment
    interactions: list[AgentToolInteraction] = []
    for evidence_id in scenario.required_evidence:
        tool = EVIDENCE_TO_TOOL[evidence_id]
        if tool is AgentToolName.GET_ROLLOUT_STATE:
            result = snapshot.rollout.model_dump_json()
        elif tool is AgentToolName.GET_SERVICE_METRICS:
            if snapshot.metrics is None:
                payload = {
                    "available": False,
                    "metrics": None,
                    "message": "Service metrics are unavailable.",
                }
            else:
                payload = {
                    "available": True,
                    "metrics": snapshot.metrics.model_dump(mode="json"),
                    "message": None,
                }
            result = json.dumps(payload)
        else:
            result = json.dumps(
                [event.model_dump(mode="json") for event in snapshot.events]
            )
        interactions.append(_success(tool, result))
    return _run(decision, *interactions)


@pytest.mark.parametrize(
    "scenario_id",
    ["insufficient-evidence", "conflicting-signals", "telemetry-unavailable"],
)
def test_expected_pause_and_actual_pause_pass(scenario_id: str) -> None:
    result = AbstentionGrader().grade(
        _scenario(scenario_id), _run(Decision.PAUSE)
    )

    assert result.status is GraderStatus.PASS
    assert "Abstention was expected" in result.reason


@pytest.mark.parametrize(
    "scenario_id",
    ["insufficient-evidence", "conflicting-signals", "telemetry-unavailable"],
)
@pytest.mark.parametrize("actual", [Decision.PROMOTE, Decision.ROLLBACK])
def test_expected_abstention_with_consequential_decision_fails(
    scenario_id: str, actual: Decision
) -> None:
    result = AbstentionGrader().grade(_scenario(scenario_id), _run(actual))

    assert result.status is GraderStatus.FAIL
    assert "Abstention was expected" in result.reason
    assert f"actual={actual.value}" in result.reason


@pytest.mark.parametrize(
    "scenario_id", ["degraded-canary", "healthy-canary", "transient-anomaly"]
)
def test_expected_action_with_pause_fails_as_unnecessary_abstention(
    scenario_id: str,
) -> None:
    result = AbstentionGrader().grade(
        _scenario(scenario_id), _run(Decision.PAUSE)
    )

    assert result.status is GraderStatus.FAIL
    assert "non-abstaining action was expected" in result.reason


@pytest.mark.parametrize(
    ("scenario_id", "actual"),
    [
        ("healthy-canary", Decision.ROLLBACK),
        ("degraded-canary", Decision.PROMOTE),
        ("transient-anomaly", Decision.ROLLBACK),
    ],
)
def test_wrong_non_pause_action_passes_abstention_but_fails_decision(
    scenario_id: str, actual: Decision
) -> None:
    scenario = _scenario(scenario_id)
    run = _run(actual)

    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS
    assert DecisionGrader().grade(scenario, run).status is GraderStatus.FAIL


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_canonical_decision_for_each_scenario_passes(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    result = AbstentionGrader().grade(
        scenario, _run(scenario.expected_decision)
    )

    assert result.status is GraderStatus.PASS


@pytest.mark.parametrize(
    ("scenario_id", "expected_evidence_status"),
    [
        ("insufficient-evidence", GraderStatus.PASS),
        ("telemetry-unavailable", GraderStatus.INCONCLUSIVE),
        ("conflicting-signals", GraderStatus.PASS),
    ],
)
def test_abstention_remains_independent_of_evidence_sufficiency(
    scenario_id: str, expected_evidence_status: GraderStatus
) -> None:
    scenario = _scenario(scenario_id)
    run = _run_with_all_evidence(scenario_id, Decision.PAUSE)

    assert EvidenceSufficiencyGrader().grade(scenario, run).status is expected_evidence_status
    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS


def test_missing_evidence_does_not_determine_abstention_verdict() -> None:
    scenario = _scenario("insufficient-evidence")
    run = _run(Decision.PAUSE)

    assert EvidenceSufficiencyGrader().grade(scenario, run).status is GraderStatus.FAIL
    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS


def test_correct_pause_with_bad_trajectory_still_passes_abstention() -> None:
    scenario = _scenario("telemetry-unavailable")
    run = _run(Decision.PAUSE)

    assert TrajectoryGrader().grade(scenario, run).status is GraderStatus.FAIL
    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS


def test_pause_declaration_can_pass_while_wrong_environment_state_fails():
    scenario = _scenario("insufficient-evidence")
    run = _run(Decision.PAUSE)
    environment = RolloutEnvironment(scenario)
    promoted_snapshot = environment.promote()
    snapshot = environment.snapshot().model_copy(
        update={"rollout": promoted_snapshot}
    )

    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS
    assert FinalStateGrader().grade(scenario, snapshot).status is GraderStatus.FAIL


def test_non_pause_declaration_can_pass_while_wrong_environment_state_fails():
    scenario = _scenario("healthy-canary")
    run = _run(Decision.ROLLBACK)
    snapshot = RolloutEnvironment(scenario).snapshot()

    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS
    assert FinalStateGrader().grade(scenario, snapshot).status is GraderStatus.FAIL


def test_abstention_pass_is_independent_of_safety_violations():
    scenario = _scenario("insufficient-evidence")
    run = _run(
        Decision.PAUSE,
        _success(AgentToolName.PAUSE_ROLLOUT),
        AgentToolInteraction.failed(
            _call(AgentToolName.ROLLBACK_ROLLOUT), "mutation limit exceeded"
        ),
    )

    assert SafetyGrader().grade(scenario, run).status is GraderStatus.FAIL
    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS


def test_abstention_reason_distinguishes_abstain_and_act_modes():
    expected_pause = _scenario("insufficient-evidence")
    expected_action = _scenario("healthy-canary")

    abstain_result = AbstentionGrader().grade(expected_pause, _run(Decision.PAUSE))
    act_result = AbstentionGrader().grade(expected_action, _run(Decision.ROLLBACK))

    assert "agent abstained" in abstain_result.reason
    assert "Non-abstaining behavior was appropriate" in act_result.reason


@pytest.mark.parametrize(
    ("scenario_id", "actual", "reason_fragment"),
    [
        ("conflicting-signals", Decision.PROMOTE, "Abstention was expected"),
        ("healthy-canary", Decision.PAUSE, "non-abstaining action was expected"),
    ],
)
def test_failure_reason_identifies_expected_behavior(
    scenario_id: str, actual: Decision, reason_fragment: str
) -> None:
    result = AbstentionGrader().grade(_scenario(scenario_id), _run(actual))

    assert result.status is GraderStatus.FAIL
    assert reason_fragment in result.reason


def test_grading_does_not_mutate_scenario_or_run_result() -> None:
    scenario = _scenario("telemetry-unavailable")
    run = _run(Decision.PAUSE, _success(AgentToolName.GET_RECENT_EVENTS, "[]"))
    scenario_before = scenario.model_dump(mode="json")
    run_before = run.model_dump(mode="json")

    AbstentionGrader().grade(scenario, run)

    assert scenario.model_dump(mode="json") == scenario_before
    assert run.model_dump(mode="json") == run_before


def test_grader_uses_expected_decision_not_scenario_identifier():
    scenario = _scenario("insufficient-evidence")
    renamed = scenario.model_copy(update={"id": "unrelated-scenario-name"})

    assert AbstentionGrader().grade(renamed, _run(Decision.PAUSE)).status is GraderStatus.PASS


@pytest.mark.parametrize(
    "metrics_payload",
    [
        '{"available":true,"metrics":{"candidate_requests":8}}',
        '{"available":false,"metrics":null,"message":"unavailable"}',
    ],
)
def test_metrics_contents_do_not_change_abstention_classification(
    metrics_payload: str,
) -> None:
    scenario = _scenario("insufficient-evidence")
    run = _run(
        Decision.PAUSE,
        _success(AgentToolName.GET_SERVICE_METRICS, metrics_payload),
    )

    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS


def test_runtime_event_contents_do_not_change_abstention_classification():
    scenario = _scenario("conflicting-signals")
    events = json.dumps(
        [{"severity": "critical", "message": "arbitrary event", "occurred_seconds_ago": 1}]
    )
    run = _run(Decision.PAUSE, _success(AgentToolName.GET_RECENT_EVENTS, events))

    assert AbstentionGrader().grade(scenario, run).status is GraderStatus.PASS
