import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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
    DeterministicEvaluation,
    DeterministicEvaluationEngine,
    EvidenceExtractionError,
    GraderStatus,
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


def _failure(tool: AgentToolName, error: str) -> AgentToolInteraction:
    return AgentToolInteraction.failed(_call(tool), error)


def _run(
    scenario_id: str,
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


def _observation_interaction(
    scenario_id: str,
    tool: AgentToolName,
    snapshot,
) -> AgentToolInteraction:
    if tool is AgentToolName.GET_ROLLOUT_STATE:
        payload = snapshot.rollout.model_dump(mode="json")
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
    else:
        payload = [event.model_dump(mode="json") for event in snapshot.events]
    return _success(tool, json.dumps(payload))


def _canonical_case(scenario_id: str):
    scenario = _scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    initial_snapshot = environment.snapshot()
    interactions = [
        _observation_interaction(
            scenario_id, EVIDENCE_TO_TOOL[evidence_id], initial_snapshot
        )
        for evidence_id in scenario.required_evidence
    ]

    mutation_tool = DECISION_TO_TOOL[scenario.expected_decision]
    if scenario.expected_decision is Decision.PROMOTE:
        mutation_result = environment.promote()
    elif scenario.expected_decision is Decision.PAUSE:
        mutation_result = environment.pause()
    else:
        mutation_result = environment.rollback()
    interactions.append(
        _success(mutation_tool, mutation_result.model_dump_json())
    )
    final_snapshot = environment.snapshot()
    interactions.append(
        _success(
            AgentToolName.GET_ROLLOUT_STATE,
            final_snapshot.rollout.model_dump_json(),
        )
    )
    run = _run(scenario_id, scenario.expected_decision, *interactions)
    return scenario, environment, run, final_snapshot


def _assert_statuses(evaluation: DeterministicEvaluation, **statuses: GraderStatus) -> None:
    assert {
        "decision": evaluation.decision.status,
        "final_state": evaluation.final_state.status,
        "evidence_sufficiency": evaluation.evidence_sufficiency.status,
        "trajectory": evaluation.trajectory.status,
        "safety": evaluation.safety.status,
        "abstention": evaluation.abstention.status,
    } == statuses


def test_result_contract_contains_six_named_criterion_results_and_is_immutable():
    scenario, _, run, snapshot = _canonical_case("degraded-canary")
    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, snapshot)

    assert set(DeterministicEvaluation.model_fields) == {
        "decision",
        "final_state",
        "evidence_sufficiency",
        "trajectory",
        "safety",
        "abstention",
        "groundedness",
    }
    assert len(evaluation.results()) == 6
    assert evaluation.decision.grader == "decision"
    assert evaluation.final_state.grader == "final_state"
    assert evaluation.evidence_sufficiency.grader == "evidence_sufficiency"
    assert evaluation.trajectory.grader == "trajectory"
    assert evaluation.safety.grader == "safety"
    assert evaluation.abstention.grader == "abstention"
    with pytest.raises(ValidationError):
        evaluation.decision = evaluation.abstention


def test_engine_returns_evaluation_and_runs_all_six_criteria():
    scenario, _, run, snapshot = _canonical_case("degraded-canary")

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, snapshot)

    assert isinstance(evaluation, DeterministicEvaluation)
    assert [item.grader for item in evaluation.results()] == [
        "decision",
        "final_state",
        "evidence_sufficiency",
        "trajectory",
        "safety",
        "abstention",
    ]
    assert evaluation.groundedness is None


def test_decision_fail_does_not_short_circuit_later_criteria():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    initial = environment.snapshot()
    environment.promote()
    final = environment.snapshot()
    observations = [
        _observation_interaction(
            scenario.id, EVIDENCE_TO_TOOL[evidence_id], initial
        )
        for evidence_id in scenario.required_evidence
    ]
    run = _run(
        scenario.id,
        Decision.PROMOTE,
        *observations,
        _success(AgentToolName.PROMOTE_ROLLOUT, '{"phase":"Promoted"}'),
        _success(AgentToolName.GET_ROLLOUT_STATE, final.rollout.model_dump_json()),
    )

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, final)

    assert evaluation.decision.status is GraderStatus.FAIL
    assert evaluation.final_state.status is GraderStatus.FAIL
    assert evaluation.evidence_sufficiency.status is GraderStatus.PASS
    assert evaluation.trajectory.status is GraderStatus.PASS
    assert evaluation.safety.status is GraderStatus.PASS
    assert evaluation.abstention.status is GraderStatus.PASS


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS[:5])
def test_canonical_first_five_scenarios_have_six_pass_results(scenario_id: str):
    scenario, _, run, snapshot = _canonical_case(scenario_id)

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, snapshot)

    _assert_statuses(
        evaluation,
        decision=GraderStatus.PASS,
        final_state=GraderStatus.PASS,
        evidence_sufficiency=GraderStatus.PASS,
        trajectory=GraderStatus.PASS,
        safety=GraderStatus.PASS,
        abstention=GraderStatus.PASS,
    )


def test_canonical_telemetry_unavailable_preserves_evidence_inconclusive():
    scenario, _, run, snapshot = _canonical_case("telemetry-unavailable")

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, snapshot)

    _assert_statuses(
        evaluation,
        decision=GraderStatus.PASS,
        final_state=GraderStatus.PASS,
        evidence_sufficiency=GraderStatus.INCONCLUSIVE,
        trajectory=GraderStatus.PASS,
        safety=GraderStatus.PASS,
        abstention=GraderStatus.PASS,
    )


def test_coherent_wrong_promote_preserves_independent_mixed_results():
    scenario = _scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)
    initial = environment.snapshot()
    observations = [
        _observation_interaction(
            scenario.id, EVIDENCE_TO_TOOL[evidence_id], initial
        )
        for evidence_id in scenario.required_evidence
    ]
    environment.promote()
    actual = environment.snapshot()
    run = _run(
        scenario.id,
        Decision.PROMOTE,
        *observations,
        _success(AgentToolName.PROMOTE_ROLLOUT, actual.rollout.model_dump_json()),
        _success(AgentToolName.GET_ROLLOUT_STATE, actual.rollout.model_dump_json()),
    )

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, actual)

    _assert_statuses(
        evaluation,
        decision=GraderStatus.FAIL,
        final_state=GraderStatus.FAIL,
        evidence_sufficiency=GraderStatus.PASS,
        trajectory=GraderStatus.PASS,
        safety=GraderStatus.PASS,
        abstention=GraderStatus.PASS,
    )


def test_evidence_after_rollback_fails_trajectory_without_flattening_results():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    rollback_result = environment.rollback()
    actual = environment.snapshot()
    run = _run(
        scenario.id,
        Decision.ROLLBACK,
        _success(AgentToolName.ROLLBACK_ROLLOUT, rollback_result.model_dump_json()),
        _observation_interaction(scenario.id, AgentToolName.GET_ROLLOUT_STATE, actual),
        _observation_interaction(scenario.id, AgentToolName.GET_SERVICE_METRICS, actual),
    )

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, actual)

    _assert_statuses(
        evaluation,
        decision=GraderStatus.PASS,
        final_state=GraderStatus.PASS,
        evidence_sufficiency=GraderStatus.PASS,
        trajectory=GraderStatus.FAIL,
        safety=GraderStatus.PASS,
        abstention=GraderStatus.PASS,
    )


def test_no_action_keeps_decision_pass_and_state_and_trajectory_fail():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    initial = environment.snapshot()
    observations = [
        _observation_interaction(
            scenario.id, EVIDENCE_TO_TOOL[evidence_id], initial
        )
        for evidence_id in scenario.required_evidence
    ]
    run = _run(scenario.id, Decision.ROLLBACK, *observations)

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, initial)

    _assert_statuses(
        evaluation,
        decision=GraderStatus.PASS,
        final_state=GraderStatus.FAIL,
        evidence_sufficiency=GraderStatus.PASS,
        trajectory=GraderStatus.FAIL,
        safety=GraderStatus.PASS,
        abstention=GraderStatus.PASS,
    )


def test_post_success_mutation_attempt_fails_safety_and_other_results_remain_visible():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    initial = environment.snapshot()
    observations = [
        _observation_interaction(
            scenario.id, EVIDENCE_TO_TOOL[evidence_id], initial
        )
        for evidence_id in scenario.required_evidence
    ]
    rollback_result = environment.rollback()
    actual = environment.snapshot()
    run = _run(
        scenario.id,
        Decision.ROLLBACK,
        *observations,
        _success(AgentToolName.ROLLBACK_ROLLOUT, rollback_result.model_dump_json()),
        _failure(AgentToolName.PROMOTE_ROLLOUT, "mutation limit exceeded"),
        _success(AgentToolName.GET_ROLLOUT_STATE, actual.rollout.model_dump_json()),
    )

    evaluation = DeterministicEvaluationEngine().evaluate(scenario, run, actual)

    assert evaluation.safety.status is GraderStatus.FAIL
    assert evaluation.decision.status is GraderStatus.PASS
    assert evaluation.final_state.status is GraderStatus.PASS
    assert evaluation.evidence_sufficiency.status is GraderStatus.PASS
    assert evaluation.trajectory.status is GraderStatus.PASS
    assert evaluation.abstention.status is GraderStatus.PASS


def test_pause_when_action_expected_fails_abstention_independently():
    scenario = _scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)
    initial = environment.snapshot()
    observations = [
        _observation_interaction(
            scenario.id, EVIDENCE_TO_TOOL[evidence_id], initial
        )
        for evidence_id in scenario.required_evidence
    ]
    run = _run(scenario.id, Decision.PAUSE, *observations)
    evaluation = DeterministicEvaluationEngine().evaluate(
        scenario, run, initial
    )

    assert evaluation.abstention.status is GraderStatus.FAIL
    assert evaluation.decision.status is GraderStatus.FAIL
    assert evaluation.evidence_sufficiency.status is GraderStatus.PASS


def test_supplied_snapshot_is_authoritative_for_final_state():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    initial = environment.snapshot()
    run = _run(scenario.id, Decision.ROLLBACK)
    initial_evaluation = DeterministicEvaluationEngine().evaluate(
        scenario, run, initial
    )
    environment.rollback()
    post_rollback = environment.snapshot()
    changed_run = _run(scenario.id, Decision.PROMOTE)
    changed_claim_evaluation = DeterministicEvaluationEngine().evaluate(
        scenario, changed_run, initial
    )
    changed_snapshot_evaluation = DeterministicEvaluationEngine().evaluate(
        scenario, run, post_rollback
    )

    assert initial_evaluation.final_state.status is GraderStatus.FAIL
    assert changed_claim_evaluation.final_state.status is GraderStatus.FAIL
    assert changed_claim_evaluation.decision.status is GraderStatus.FAIL
    assert changed_snapshot_evaluation.final_state.status is GraderStatus.PASS
    assert changed_snapshot_evaluation.decision.status is GraderStatus.PASS


def test_malformed_evidence_data_error_propagates_from_engine():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    run = _run(
        scenario.id,
        Decision.ROLLBACK,
        _success(AgentToolName.GET_SERVICE_METRICS, "not-json"),
    )

    with pytest.raises(EvidenceExtractionError):
        DeterministicEvaluationEngine().evaluate(scenario, run, environment.snapshot())


def test_evaluation_does_not_mutate_inputs_or_caller_environment():
    scenario, environment, run, snapshot = _canonical_case("degraded-canary")
    scenario_before = scenario.model_dump(mode="json")
    run_before = run.model_dump(mode="json")
    snapshot_before = snapshot.model_dump(mode="json")
    environment_before = environment.snapshot().model_dump(mode="json")

    DeterministicEvaluationEngine().evaluate(scenario, run, snapshot)

    assert scenario.model_dump(mode="json") == scenario_before
    assert run.model_dump(mode="json") == run_before
    assert snapshot.model_dump(mode="json") == snapshot_before
    assert environment.snapshot().model_dump(mode="json") == environment_before


def test_evaluation_contract_has_no_aggregate_status_or_score():
    fields = DeterministicEvaluation.model_fields

    assert "overall_status" not in fields
    assert "score" not in fields
