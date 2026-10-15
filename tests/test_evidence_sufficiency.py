import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval.agent import AgentFinalAnswer, AgentRunResult, AgentToolCall, AgentToolInteraction, AgentToolName
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import (
    EvidenceExtractionError,
    EvidenceId,
    EvidenceObservation,
    EvidenceStatus,
    EvidenceSufficiencyGrader,
    GraderStatus,
    extract_evidence,
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
OBSERVATION_TO_EVIDENCE = {
    AgentToolName.GET_ROLLOUT_STATE: EvidenceId.ROLLOUT_STATE,
    AgentToolName.GET_SERVICE_METRICS: EvidenceId.SERVICE_METRICS,
    AgentToolName.GET_RECENT_EVENTS: EvidenceId.RUNTIME_EVENTS,
}


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _run_result(
    *interactions: AgentToolInteraction,
    decision: Decision = Decision.PAUSE,
) -> AgentRunResult:
    return AgentRunResult(
        final_answer=AgentFinalAnswer(
            kind="final", decision=decision, explanation="A test-only final explanation."
        ),
        interactions=tuple(interactions),
        steps=len(interactions) + 1,
    )


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _observed(tool: AgentToolName, *, candidate_requests: int = 450) -> AgentToolInteraction:
    if tool is AgentToolName.GET_ROLLOUT_STATE:
        payload = {"phase": "Progressing", "candidate_weight": 20}
    elif tool is AgentToolName.GET_SERVICE_METRICS:
        payload = {
            "available": True,
            "metrics": {"candidate_requests": candidate_requests},
            "message": None,
        }
    elif tool is AgentToolName.GET_RECENT_EVENTS:
        payload = []
    else:
        payload = {"phase": tool.value}
    return AgentToolInteraction.successful(_call(tool), json.dumps(payload))


def _unavailable_metrics() -> AgentToolInteraction:
    return AgentToolInteraction.successful(
        _call(AgentToolName.GET_SERVICE_METRICS),
        json.dumps(
            {
                "available": False,
                "metrics": None,
                "message": "Service metrics are unavailable.",
            }
        ),
    )


def _errored(tool: AgentToolName, error: str = "observation failed") -> AgentToolInteraction:
    return AgentToolInteraction.failed(_call(tool), error)


def _canonical_evidence_result(scenario_id: str) -> AgentRunResult:
    scenario = _scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    snapshot = environment.snapshot()
    interactions: list[AgentToolInteraction] = []
    for evidence_id in scenario.required_evidence:
        if evidence_id == EvidenceId.ROLLOUT_STATE.value:
            interactions.append(
                AgentToolInteraction.successful(
                    _call(AgentToolName.GET_ROLLOUT_STATE),
                    snapshot.rollout.model_dump_json(),
                )
            )
        elif evidence_id == EvidenceId.SERVICE_METRICS.value:
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
            interactions.append(
                AgentToolInteraction.successful(
                    _call(AgentToolName.GET_SERVICE_METRICS), json.dumps(payload)
                )
            )
        elif evidence_id == EvidenceId.RUNTIME_EVENTS.value:
            interactions.append(
                AgentToolInteraction.successful(
                    _call(AgentToolName.GET_RECENT_EVENTS),
                    json.dumps([event.model_dump(mode="json") for event in snapshot.events]),
                )
            )
    return _run_result(*interactions, decision=scenario.expected_decision)


def test_evidence_status_has_exact_values() -> None:
    assert {status.value for status in EvidenceStatus} == {
        "OBSERVED",
        "UNAVAILABLE",
        "ERROR",
    }


def test_evidence_observation_can_represent_observed_evidence_and_is_immutable() -> None:
    observation = EvidenceObservation(
        evidence_id=EvidenceId.ROLLOUT_STATE,
        source_tool="get_rollout_state",
        interaction_index=0,
        status=EvidenceStatus.OBSERVED,
    )

    assert observation.status is EvidenceStatus.OBSERVED
    with pytest.raises(ValidationError):
        observation.status = EvidenceStatus.ERROR


def test_evidence_observation_rejects_invalid_identifier() -> None:
    with pytest.raises(ValidationError):
        EvidenceObservation(
            evidence_id="decision",
            source_tool="get_rollout_state",
            interaction_index=0,
            status=EvidenceStatus.OBSERVED,
        )


def test_successful_rollout_state_produces_observed_evidence() -> None:
    observations = extract_evidence(_run_result(_observed(AgentToolName.GET_ROLLOUT_STATE)))

    assert observations == (
        EvidenceObservation(
            evidence_id=EvidenceId.ROLLOUT_STATE,
            source_tool="get_rollout_state",
            interaction_index=0,
            status=EvidenceStatus.OBSERVED,
        ),
    )


def test_rollout_state_error_produces_error_evidence() -> None:
    observations = extract_evidence(
        _run_result(_errored(AgentToolName.GET_ROLLOUT_STATE))
    )

    assert observations[0].evidence_id is EvidenceId.ROLLOUT_STATE
    assert observations[0].status is EvidenceStatus.ERROR


def test_available_service_metrics_produce_observed_evidence() -> None:
    observations = extract_evidence(
        _run_result(_observed(AgentToolName.GET_SERVICE_METRICS))
    )

    assert observations[0].evidence_id is EvidenceId.SERVICE_METRICS
    assert observations[0].status is EvidenceStatus.OBSERVED


def test_unavailable_service_metrics_produce_unavailable_evidence() -> None:
    observations = extract_evidence(_run_result(_unavailable_metrics()))

    assert observations[0].status is EvidenceStatus.UNAVAILABLE


def test_service_metrics_error_produces_error_evidence() -> None:
    observations = extract_evidence(
        _run_result(_errored(AgentToolName.GET_SERVICE_METRICS))
    )

    assert observations[0].status is EvidenceStatus.ERROR


def test_eight_candidate_requests_are_still_observed_metrics() -> None:
    observations = extract_evidence(
        _run_result(_observed(AgentToolName.GET_SERVICE_METRICS, candidate_requests=8))
    )

    assert observations[0].status is EvidenceStatus.OBSERVED


@pytest.mark.parametrize(
    "result",
    [
        "not json",
        "{}",
        '{"available":"yes","metrics":{}}',
        '{"available":true,"metrics":null}',
        '{"available":false,"metrics":{},"message":"missing"}',
        '{"available":false,"metrics":null}',
    ],
)
def test_malformed_successful_metrics_result_raises_extraction_error(result: str) -> None:
    interaction = AgentToolInteraction.successful(
        _call(AgentToolName.GET_SERVICE_METRICS), result
    )

    with pytest.raises(EvidenceExtractionError):
        extract_evidence(_run_result(interaction))


def test_successful_recent_events_produce_observed_evidence() -> None:
    observations = extract_evidence(_run_result(_observed(AgentToolName.GET_RECENT_EVENTS)))

    assert observations[0].evidence_id is EvidenceId.RUNTIME_EVENTS
    assert observations[0].status is EvidenceStatus.OBSERVED


def test_recent_events_error_produces_error_evidence() -> None:
    observations = extract_evidence(
        _run_result(_errored(AgentToolName.GET_RECENT_EVENTS))
    )

    assert observations[0].status is EvidenceStatus.ERROR


@pytest.mark.parametrize(
    "tool",
    [
        AgentToolName.PROMOTE_ROLLOUT,
        AgentToolName.PAUSE_ROLLOUT,
        AgentToolName.ROLLBACK_ROLLOUT,
    ],
)
def test_mutation_tools_do_not_create_evidence(tool: AgentToolName) -> None:
    mutation = AgentToolInteraction.successful(_call(tool), '{"phase":"Changed"}')

    assert extract_evidence(_run_result(mutation)) == ()


def test_extracted_evidence_preserves_interaction_order_and_indices() -> None:
    run = _run_result(
        _observed(AgentToolName.GET_RECENT_EVENTS),
        AgentToolInteraction.successful(_call(AgentToolName.PROMOTE_ROLLOUT), "{}"),
        _observed(AgentToolName.GET_ROLLOUT_STATE),
    )

    observations = extract_evidence(run)

    assert [(item.evidence_id, item.interaction_index) for item in observations] == [
        (EvidenceId.RUNTIME_EVENTS, 0),
        (EvidenceId.ROLLOUT_STATE, 2),
    ]


def test_repeated_observations_produce_repeated_records() -> None:
    observations = extract_evidence(
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _observed(AgentToolName.GET_ROLLOUT_STATE),
        )
    )

    assert [item.interaction_index for item in observations] == [0, 1]
    assert all(item.status is EvidenceStatus.OBSERVED for item in observations)


def test_unavailable_then_observed_metrics_retains_both_records() -> None:
    observations = extract_evidence(
        _run_result(_unavailable_metrics(), _observed(AgentToolName.GET_SERVICE_METRICS))
    )

    assert [item.status for item in observations] == [
        EvidenceStatus.UNAVAILABLE,
        EvidenceStatus.OBSERVED,
    ]


@pytest.mark.parametrize(
    "scenario_id",
    [
        "degraded-canary",
        "healthy-canary",
        "insufficient-evidence",
        "transient-anomaly",
        "conflicting-signals",
    ],
)
def test_available_required_evidence_passes_for_golden_scenarios(
    scenario_id: str,
) -> None:
    scenario = _scenario(scenario_id)

    result = EvidenceSufficiencyGrader().grade(
        scenario, _canonical_evidence_result(scenario_id)
    )

    assert result.status is GraderStatus.PASS


def test_degraded_canary_fails_when_metrics_never_requested() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(_observed(AgentToolName.GET_ROLLOUT_STATE)),
    )

    assert result.status is GraderStatus.FAIL
    assert "service_metrics" in result.reason


def test_missing_rollout_state_fails() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(_observed(AgentToolName.GET_SERVICE_METRICS)),
    )

    assert result.status is GraderStatus.FAIL
    assert "rollout_state" in result.reason


def test_missing_required_runtime_events_fails() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("transient-anomaly"),
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _observed(AgentToolName.GET_SERVICE_METRICS),
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "runtime_events" in result.reason


def test_telemetry_unavailable_is_inconclusive_not_missing() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("telemetry-unavailable"),
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _unavailable_metrics(),
        ),
    )

    assert result.status is GraderStatus.INCONCLUSIVE
    assert "service_metrics (unavailable)" in result.reason


def test_required_observation_error_is_inconclusive_when_every_source_attempted() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _errored(AgentToolName.GET_SERVICE_METRICS),
        ),
    )

    assert result.status is GraderStatus.INCONCLUSIVE
    assert "service_metrics (error)" in result.reason


def test_unavailable_or_error_reason_identifies_evidence_id() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(
            _errored(AgentToolName.GET_ROLLOUT_STATE),
            _unavailable_metrics(),
        ),
    )

    assert result.status is GraderStatus.INCONCLUSIVE
    assert "rollout_state (error)" in result.reason
    assert "service_metrics (unavailable)" in result.reason


def test_missing_evidence_takes_precedence_over_unavailable_evidence() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("transient-anomaly"),
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _unavailable_metrics(),
        ),
    )

    assert result.status is GraderStatus.FAIL
    assert "runtime_events" in result.reason


def test_unavailable_metrics_followed_by_observed_metrics_satisfies_requirement() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _unavailable_metrics(),
            _observed(AgentToolName.GET_SERVICE_METRICS),
        ),
    )

    assert result.status is GraderStatus.PASS


def test_observed_evidence_passes_despite_wrong_final_decision() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _observed(AgentToolName.GET_SERVICE_METRICS),
            decision=Decision.PROMOTE,
        ),
    )

    assert result.status is GraderStatus.PASS


def test_observed_evidence_passes_despite_wrong_environment_state() -> None:
    scenario = _scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)
    environment.promote()
    run = _run_result(
        _observed(AgentToolName.GET_ROLLOUT_STATE),
        _observed(AgentToolName.GET_SERVICE_METRICS),
    )

    result = EvidenceSufficiencyGrader().grade(scenario, run)

    assert environment.snapshot().rollout.phase == "Promoted"
    assert result.status is GraderStatus.PASS


def test_evidence_observed_after_mutation_can_pass_sufficiency() -> None:
    result = EvidenceSufficiencyGrader().grade(
        _scenario("degraded-canary"),
        _run_result(
            AgentToolInteraction.successful(
                _call(AgentToolName.ROLLBACK_ROLLOUT), '{"phase":"RolledBack"}'
            ),
            _observed(AgentToolName.GET_ROLLOUT_STATE),
            _observed(AgentToolName.GET_SERVICE_METRICS),
        ),
    )

    assert result.status is GraderStatus.PASS


def test_evidence_extraction_does_not_mutate_run_result() -> None:
    run = _run_result(
        _observed(AgentToolName.GET_ROLLOUT_STATE),
        _observed(AgentToolName.GET_SERVICE_METRICS),
    )
    before = run.model_dump(mode="json")

    extract_evidence(run)

    assert run.model_dump(mode="json") == before


def test_evidence_grading_does_not_mutate_scenario_or_run_result() -> None:
    scenario = _scenario()
    run = _run_result(
        _observed(AgentToolName.GET_ROLLOUT_STATE),
        _observed(AgentToolName.GET_SERVICE_METRICS),
    )
    scenario_before = scenario.model_dump(mode="json")
    run_before = run.model_dump(mode="json")

    EvidenceSufficiencyGrader().grade(scenario, run)

    assert scenario.model_dump(mode="json") == scenario_before
    assert run.model_dump(mode="json") == run_before


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_evidence_extraction_operates_for_all_golden_scenarios(
    scenario_id: str,
) -> None:
    scenario = _scenario(scenario_id)

    observations = extract_evidence(_canonical_evidence_result(scenario_id))

    assert tuple(item.evidence_id.value for item in observations) == scenario.required_evidence
    assert all(item.status in EvidenceStatus for item in observations)
