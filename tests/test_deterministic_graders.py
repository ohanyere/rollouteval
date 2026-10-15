import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval.agent import AgentFinalAnswer, AgentRunResult
from rollouteval.domain.scenario import Decision, InitialEnvironment, RolloutState
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import (
    DecisionGrader,
    FinalStateGrader,
    GraderResult,
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


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _run_result(decision: Decision, explanation: str = "Decision explanation.") -> AgentRunResult:
    return AgentRunResult(
        final_answer=AgentFinalAnswer(
            kind="final", decision=decision, explanation=explanation
        ),
        interactions=(),
        steps=1,
    )


def _snapshot_after_action(scenario_id: str, action: str) -> InitialEnvironment:
    scenario = _scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    getattr(environment, action)()
    return environment.snapshot()


def _snapshot_with_rollout(
    scenario_id: str, phase: str, candidate_weight: int
) -> InitialEnvironment:
    scenario = _scenario(scenario_id)
    snapshot = scenario.initial_environment.model_copy(deep=True)
    rollout = snapshot.rollout.model_copy(
        update={"phase": phase, "candidate_weight": candidate_weight}
    )
    return snapshot.model_copy(update={"rollout": rollout})


def test_grader_status_has_exact_values() -> None:
    assert {status.value for status in GraderStatus} == {
        "PASS",
        "FAIL",
        "INCONCLUSIVE",
    }


@pytest.mark.parametrize("status", list(GraderStatus))
def test_grader_result_accepts_each_status(status: GraderStatus) -> None:
    result = GraderResult(grader="decision", status=status, reason="Comparison completed.")

    assert result.status is status


def test_grader_result_is_immutable() -> None:
    result = GraderResult(grader="decision", status=GraderStatus.PASS, reason="Matched.")

    with pytest.raises(ValidationError):
        result.status = GraderStatus.FAIL


@pytest.mark.parametrize("decision", list(Decision))
def test_matching_decision_passes(decision: Decision) -> None:
    scenario_id = {
        Decision.ROLLBACK: "degraded-canary",
        Decision.PROMOTE: "healthy-canary",
        Decision.PAUSE: "insufficient-evidence",
    }[decision]

    result = DecisionGrader().grade(_scenario(scenario_id), _run_result(decision))

    assert result.status is GraderStatus.PASS


@pytest.mark.parametrize(
    "expected, actual",
    [
        (Decision.ROLLBACK, Decision.PROMOTE),
        (Decision.PROMOTE, Decision.ROLLBACK),
        (Decision.PAUSE, Decision.PROMOTE),
    ],
)
def test_mismatching_decision_fails(expected: Decision, actual: Decision) -> None:
    scenario_id = {
        Decision.ROLLBACK: "degraded-canary",
        Decision.PROMOTE: "healthy-canary",
        Decision.PAUSE: "insufficient-evidence",
    }[expected]

    result = DecisionGrader().grade(_scenario(scenario_id), _run_result(actual))

    assert result.status is GraderStatus.FAIL


def test_decision_reason_identifies_expected_and_actual_values() -> None:
    result = DecisionGrader().grade(
        _scenario("degraded-canary"), _run_result(Decision.PROMOTE)
    )

    assert "expected=rollback" in result.reason
    assert "actual=promote" in result.reason


@pytest.mark.parametrize(
    "scenario_id, action, expected_phase, expected_weight",
    [
        ("healthy-canary", "promote", "Promoted", 100),
        ("insufficient-evidence", "pause", "Paused", 20),
        ("degraded-canary", "rollback", "RolledBack", 0),
    ],
)
def test_matching_final_phase_and_weight_pass(
    scenario_id: str, action: str, expected_phase: str, expected_weight: int
) -> None:
    scenario = _scenario(scenario_id)
    snapshot = _snapshot_after_action(scenario_id, action)

    result = FinalStateGrader().grade(scenario, snapshot)

    assert result.status is GraderStatus.PASS
    assert snapshot.rollout.phase == expected_phase
    assert snapshot.rollout.candidate_weight == expected_weight


def test_incorrect_final_phase_fails() -> None:
    scenario = _scenario("degraded-canary")
    snapshot = _snapshot_with_rollout("degraded-canary", "Promoted", 0)

    result = FinalStateGrader().grade(scenario, snapshot)

    assert result.status is GraderStatus.FAIL
    assert "phase expected=RolledBack, actual=Promoted" in result.reason


def test_incorrect_candidate_weight_fails() -> None:
    scenario = _scenario("degraded-canary")
    snapshot = _snapshot_with_rollout("degraded-canary", "RolledBack", 20)

    result = FinalStateGrader().grade(scenario, snapshot)

    assert result.status is GraderStatus.FAIL
    assert "candidate_weight expected=0, actual=20" in result.reason


def test_both_incorrect_final_state_fields_fail_and_are_identified() -> None:
    scenario = _scenario("degraded-canary")
    snapshot = _snapshot_with_rollout("degraded-canary", "Promoted", 100)

    result = FinalStateGrader().grade(scenario, snapshot)

    assert result.status is GraderStatus.FAIL
    assert "phase expected=RolledBack, actual=Promoted" in result.reason
    assert "candidate_weight expected=0, actual=100" in result.reason


@pytest.mark.parametrize(
    "decision, action, decision_status, state_status",
    [
        (Decision.ROLLBACK, "rollback", GraderStatus.PASS, GraderStatus.PASS),
        (Decision.PROMOTE, "promote", GraderStatus.FAIL, GraderStatus.FAIL),
        (Decision.ROLLBACK, "promote", GraderStatus.PASS, GraderStatus.FAIL),
        (Decision.PROMOTE, "rollback", GraderStatus.FAIL, GraderStatus.PASS),
    ],
)
def test_decision_and_final_state_graders_are_independent(
    decision: Decision,
    action: str,
    decision_status: GraderStatus,
    state_status: GraderStatus,
) -> None:
    scenario = _scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)
    getattr(environment, action)()
    run_result = _run_result(decision)

    decision_result = DecisionGrader().grade(scenario, run_result)
    final_state_result = FinalStateGrader().grade(scenario, environment.snapshot())

    assert decision_result.status is decision_status
    assert final_state_result.status is state_status


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_canonical_execution_for_each_scenario_passes_both_graders(
    scenario_id: str,
) -> None:
    scenario = _scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    action_by_decision = {
        Decision.PROMOTE: environment.promote,
        Decision.PAUSE: environment.pause,
        Decision.ROLLBACK: environment.rollback,
    }
    action_by_decision[scenario.expected_decision]()
    run_result = _run_result(scenario.expected_decision)

    decision_result = DecisionGrader().grade(scenario, run_result)
    final_state_result = FinalStateGrader().grade(scenario, environment.snapshot())

    assert decision_result.status is GraderStatus.PASS
    assert final_state_result.status is GraderStatus.PASS


def test_decision_grader_does_not_mutate_scenario_or_run_result() -> None:
    scenario = _scenario()
    run_result = _run_result(Decision.ROLLBACK)
    scenario_before = scenario.model_dump(mode="json")
    run_before = run_result.model_dump(mode="json")

    DecisionGrader().grade(scenario, run_result)

    assert scenario.model_dump(mode="json") == scenario_before
    assert run_result.model_dump(mode="json") == run_before


def test_final_state_grader_does_not_mutate_snapshot_or_environment() -> None:
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    environment.rollback()
    snapshot = environment.snapshot()
    snapshot_before = snapshot.model_dump(mode="json")
    environment_before = environment.snapshot()
    mutation_count_before = environment.mutation_count

    FinalStateGrader().grade(scenario, snapshot)

    assert snapshot.model_dump(mode="json") == snapshot_before
    assert environment.snapshot() == environment_before
    assert environment.mutation_count == mutation_count_before


def test_final_state_grader_uses_state_even_when_answer_claim_disagrees() -> None:
    scenario = _scenario("degraded-canary")
    snapshot = _snapshot_after_action("degraded-canary", "promote")
    run_result = _run_result(Decision.ROLLBACK, "I rolled back the candidate.")

    result = FinalStateGrader().grade(scenario, snapshot)

    assert run_result.final_answer.explanation
    assert result.status is GraderStatus.FAIL
    assert "actual=Promoted" in result.reason


def test_final_state_grader_does_not_accept_agent_claim_as_input() -> None:
    parameters = tuple(inspect.signature(FinalStateGrader.grade).parameters)

    assert parameters == ("self", "scenario", "runtime_snapshot")


def test_decision_grader_uses_decision_even_when_environment_disagrees() -> None:
    scenario = _scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)
    environment.promote()

    result = DecisionGrader().grade(scenario, _run_result(Decision.ROLLBACK))

    assert environment.snapshot().rollout.phase == "Promoted"
    assert result.status is GraderStatus.PASS
