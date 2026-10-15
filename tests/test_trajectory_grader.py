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
from rollouteval.evaluation import (
    EvidenceSufficiencyGrader,
    GraderStatus,
    TrajectoryGrader,
)
from rollouteval.scenarios import load_scenario


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
EVIDENCE_TO_TOOL = {
    "rollout_state": AgentToolName.GET_ROLLOUT_STATE,
    "service_metrics": AgentToolName.GET_SERVICE_METRICS,
    "runtime_events": AgentToolName.GET_RECENT_EVENTS,
}
DECISION_TO_TOOL = {
    Decision.PROMOTE: AgentToolName.PROMOTE_ROLLOUT,
    Decision.PAUSE: AgentToolName.PAUSE_ROLLOUT,
    Decision.ROLLBACK: AgentToolName.ROLLBACK_ROLLOUT,
}


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _success(tool: AgentToolName, result: str = "{}") -> AgentToolInteraction:
    return AgentToolInteraction.successful(_call(tool), result)


def _error(tool: AgentToolName) -> AgentToolInteraction:
    return AgentToolInteraction.failed(_call(tool), "tool failed")


def _run(
    *interactions: AgentToolInteraction,
    decision: Decision = Decision.ROLLBACK,
    explanation: str = "Visible test explanation.",
) -> AgentRunResult:
    return AgentRunResult(
        final_answer=AgentFinalAnswer(
            kind="final", decision=decision, explanation=explanation
        ),
        interactions=tuple(interactions),
        steps=len(interactions) + 1,
    )


def _available_observation(tool: AgentToolName) -> AgentToolInteraction:
    if tool is AgentToolName.GET_SERVICE_METRICS:
        result = json.dumps({"available": True, "metrics": {"candidate_requests": 8}})
    elif tool is AgentToolName.GET_RECENT_EVENTS:
        result = "[]"
    else:
        result = '{"phase":"Progressing","candidate_weight":20}'
    return _success(tool, result)


def _canonical_run(scenario_id: str) -> AgentRunResult:
    scenario = _scenario(scenario_id)
    evidence = [
        _available_observation(EVIDENCE_TO_TOOL[evidence_id])
        for evidence_id in scenario.required_evidence
    ]
    mutation = DECISION_TO_TOOL[scenario.expected_decision]
    return _run(
        *evidence,
        _success(mutation, '{"phase":"Changed"}'),
        _success(AgentToolName.GET_ROLLOUT_STATE, '{"phase":"Changed"}'),
        decision=scenario.expected_decision,
    )


@pytest.mark.parametrize(
    "scenario_id",
    [
        "degraded-canary",
        "healthy-canary",
        "insufficient-evidence",
        "transient-anomaly",
        "conflicting-signals",
        "telemetry-unavailable",
    ],
)
def test_canonical_trajectory_passes_for_all_scenarios(scenario_id: str) -> None:
    result = TrajectoryGrader().grade(_scenario(scenario_id), _canonical_run(scenario_id))

    assert result.status is GraderStatus.PASS


def test_mutation_before_any_required_evidence_fails() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            _success(AgentToolName.GET_ROLLOUT_STATE),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "rollout_state" in result.reason
    assert "service_metrics" in result.reason


def test_mutation_after_only_rollout_state_fails_missing_metrics_ordering() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            _success(AgentToolName.GET_ROLLOUT_STATE),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "service_metrics" in result.reason


@pytest.mark.parametrize("scenario_id", ["transient-anomaly", "conflicting-signals"])
def test_mutation_before_required_runtime_events_fails(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    earlier = [
        _available_observation(EVIDENCE_TO_TOOL[evidence_id])
        for evidence_id in scenario.required_evidence
        if evidence_id != "runtime_events"
    ]
    run = _run(
        *earlier,
        _success(AgentToolName.PROMOTE_ROLLOUT),
        _success(AgentToolName.GET_RECENT_EVENTS, "[]"),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        decision=Decision.PROMOTE,
    )

    result = TrajectoryGrader().grade(scenario, run)

    assert result.status is GraderStatus.FAIL
    assert "runtime_events" in result.reason


def test_later_evidence_does_not_repair_pre_mutation_ordering() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "not attempted before mutation" in result.reason


@pytest.mark.parametrize(
    "interaction",
    [
        _error(AgentToolName.GET_SERVICE_METRICS),
        _success(
            AgentToolName.GET_SERVICE_METRICS,
            '{"available":false,"metrics":null,"message":"offline"}',
        ),
    ],
    ids=["error", "unavailable"],
)
def test_error_or_unavailable_evidence_before_mutation_counts_as_attempt(
    interaction: AgentToolInteraction,
) -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            interaction,
            _success(AgentToolName.PAUSE_ROLLOUT),
            _success(AgentToolName.GET_ROLLOUT_STATE),
            decision=Decision.PAUSE,
        ),
    )

    assert result.status is GraderStatus.PASS


def test_successful_mutation_followed_by_successful_state_observation_passes() -> None:
    result = TrajectoryGrader().grade(
        _scenario(), _canonical_run("degraded-canary")
    )

    assert result.status is GraderStatus.PASS


def test_successful_mutation_without_later_state_observation_fails() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "verification" in result.reason


def test_pre_mutation_state_observation_does_not_verify_mutation() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            decision=Decision.ROLLBACK,
        ),
    )

    assert "verification" in result.reason


def test_failed_post_mutation_state_observation_does_not_verify_mutation() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            _error(AgentToolName.GET_ROLLOUT_STATE),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "verification" in result.reason


def test_repeated_observations_are_allowed() -> None:
    run = _run(
        _success(AgentToolName.GET_ROLLOUT_STATE),
        _success(AgentToolName.GET_SERVICE_METRICS),
        _success(AgentToolName.GET_SERVICE_METRICS),
        _success(AgentToolName.ROLLBACK_ROLLOUT),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        decision=Decision.ROLLBACK,
    )

    assert TrajectoryGrader().grade(_scenario(), run).status is GraderStatus.PASS


@pytest.mark.parametrize(
    ("tool", "decision"),
    [
        (AgentToolName.PROMOTE_ROLLOUT, Decision.PROMOTE),
        (AgentToolName.PAUSE_ROLLOUT, Decision.PAUSE),
        (AgentToolName.ROLLBACK_ROLLOUT, Decision.ROLLBACK),
    ],
)
def test_mutation_matching_final_decision_passes(
    tool: AgentToolName, decision: Decision
) -> None:
    run = _run(
        _success(AgentToolName.GET_ROLLOUT_STATE),
        _success(AgentToolName.GET_SERVICE_METRICS),
        _success(tool),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        decision=decision,
    )

    assert TrajectoryGrader().grade(_scenario(), run).status is GraderStatus.PASS


@pytest.mark.parametrize(
    ("tool", "decision", "action"),
    [
        (AgentToolName.ROLLBACK_ROLLOUT, Decision.PROMOTE, "rollback"),
        (AgentToolName.PROMOTE_ROLLOUT, Decision.ROLLBACK, "promote"),
        (AgentToolName.PAUSE_ROLLOUT, Decision.ROLLBACK, "pause"),
    ],
)
def test_mutation_conflicting_with_final_decision_fails(
    tool: AgentToolName, decision: Decision, action: str
) -> None:
    run = _run(
        _success(AgentToolName.GET_ROLLOUT_STATE),
        _success(AgentToolName.GET_SERVICE_METRICS),
        _success(tool),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        decision=decision,
    )

    result = TrajectoryGrader().grade(_scenario(), run)

    assert result.status is GraderStatus.FAIL
    assert action in result.reason
    assert decision.value in result.reason


def test_no_successful_mutation_fails_even_after_evidence_was_read() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "no successful rollout mutation" in result.reason


def test_multiple_successful_mutations_fail() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            _success(AgentToolName.PROMOTE_ROLLOUT),
            _success(AgentToolName.GET_ROLLOUT_STATE),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "multiple successful rollout mutations" in result.reason


def test_failed_second_mutation_is_not_a_second_successful_mutation() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(
            _success(AgentToolName.GET_ROLLOUT_STATE),
            _success(AgentToolName.GET_SERVICE_METRICS),
            _success(AgentToolName.ROLLBACK_ROLLOUT),
            _error(AgentToolName.PROMOTE_ROLLOUT),
            _success(AgentToolName.GET_ROLLOUT_STATE),
            decision=Decision.ROLLBACK,
        ),
    )

    assert result.status is GraderStatus.PASS


@pytest.mark.parametrize(
    ("scenario_id", "tool", "decision"),
    [
        ("healthy-canary", AgentToolName.ROLLBACK_ROLLOUT, Decision.ROLLBACK),
        ("degraded-canary", AgentToolName.PROMOTE_ROLLOUT, Decision.PROMOTE),
    ],
)
def test_coherent_wrong_golden_action_still_passes_trajectory(
    scenario_id: str, tool: AgentToolName, decision: Decision
) -> None:
    scenario = _scenario(scenario_id)
    run = _run(
        *[
            _available_observation(EVIDENCE_TO_TOOL[evidence_id])
            for evidence_id in scenario.required_evidence
        ],
        _success(tool),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        decision=decision,
    )

    assert TrajectoryGrader().grade(scenario, run).status is GraderStatus.PASS


def test_telemetry_unavailable_can_pass_trajectory_but_be_inconclusive_for_evidence():
    scenario = _scenario("telemetry-unavailable")
    run = _run(
        _success(AgentToolName.GET_ROLLOUT_STATE),
        _success(
            AgentToolName.GET_SERVICE_METRICS,
            '{"available":false,"metrics":null,"message":"offline"}',
        ),
        _success(AgentToolName.PAUSE_ROLLOUT),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        decision=Decision.PAUSE,
    )

    assert TrajectoryGrader().grade(scenario, run).status is GraderStatus.PASS
    assert EvidenceSufficiencyGrader().grade(scenario, run).status is GraderStatus.INCONCLUSIVE


def test_evidence_only_after_mutation_can_pass_sufficiency_but_fail_trajectory():
    scenario = _scenario()
    run = _run(
        _success(AgentToolName.ROLLBACK_ROLLOUT),
        _success(AgentToolName.GET_ROLLOUT_STATE),
        _available_observation(AgentToolName.GET_SERVICE_METRICS),
        decision=Decision.ROLLBACK,
    )

    assert EvidenceSufficiencyGrader().grade(scenario, run).status is GraderStatus.PASS
    assert TrajectoryGrader().grade(scenario, run).status is GraderStatus.FAIL


def test_multiple_trajectory_violations_are_reported_together() -> None:
    result = TrajectoryGrader().grade(
        _scenario(),
        _run(_success(AgentToolName.PROMOTE_ROLLOUT), decision=Decision.ROLLBACK),
    )

    assert result.status is GraderStatus.FAIL
    assert "not attempted before mutation" in result.reason
    assert "verification" in result.reason
    assert "conflicts with final decision" in result.reason


def test_grading_does_not_mutate_scenario_or_run_result() -> None:
    scenario = _scenario()
    run = _canonical_run("degraded-canary")
    scenario_before = scenario.model_dump(mode="json")
    run_before = run.model_dump(mode="json")

    TrajectoryGrader().grade(scenario, run)

    assert scenario.model_dump(mode="json") == scenario_before
    assert run.model_dump(mode="json") == run_before
