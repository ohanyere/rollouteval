from pathlib import Path

import pytest

from rollouteval import Decision, Scenario, load_scenario
from rollouteval.environment import RolloutEnvironment
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_MATRIX = (
    ("degraded-canary", Decision.ROLLBACK, "RolledBack", 0),
    ("healthy-canary", Decision.PROMOTE, "Promoted", 100),
    ("insufficient-evidence", Decision.PAUSE, "Paused", 20),
    ("transient-anomaly", Decision.PROMOTE, "Promoted", 100),
    ("conflicting-signals", Decision.PAUSE, "Paused", 20),
    ("telemetry-unavailable", Decision.PAUSE, "Paused", 20),
)
def _episode(
    scenario_id: str,
) -> tuple[Scenario, RolloutEnvironment, RolloutObservationTools, RolloutMutationTools]:
    scenario = load_scenario(SCENARIOS / f"{scenario_id}.yaml")
    environment = RolloutEnvironment(scenario)
    return scenario, environment, RolloutObservationTools(environment), RolloutMutationTools(environment)


def test_scenario_evidence_identifiers_have_observation_capabilities() -> None:
    scenario_ids = {scenario_id for scenario_id, *_ in SCENARIO_MATRIX}

    for scenario_id in scenario_ids:
        scenario, _, observations, _ = _episode(scenario_id)
        observation_capabilities = {
            "rollout_state": observations.get_rollout_state,
            "service_metrics": observations.get_service_metrics,
            "runtime_events": observations.get_recent_events,
        }
        for evidence_id in scenario.required_evidence:
            capability = observation_capabilities.get(evidence_id)
            assert capability is not None, f"unsupported scenario evidence identifier: {evidence_id!r}"
            assert callable(capability)
            capability()


def _assert_initial_observations(
    scenario: Scenario,
    environment: RolloutEnvironment,
    tools: RolloutObservationTools,
) -> None:
    initial_state = scenario.initial_environment
    assert tools.get_rollout_state() == initial_state.rollout

    metrics_observation = tools.get_service_metrics()
    assert metrics_observation.available is (initial_state.metrics is not None)
    assert metrics_observation.metrics == initial_state.metrics
    if initial_state.metrics is None:
        assert metrics_observation.message is not None
        assert "unavailable" in metrics_observation.message.lower()

    assert tools.get_recent_events() == initial_state.events
    observed = {"rollout_state", "service_metrics", "runtime_events"}
    assert set(scenario.required_evidence) <= observed


@pytest.mark.parametrize(
    "scenario_id, expected_decision, expected_phase, expected_weight",
    SCENARIO_MATRIX,
)
def test_complete_environment_workflow_for_each_golden_scenario(
    scenario_id: str,
    expected_decision: Decision,
    expected_phase: str,
    expected_weight: int,
) -> None:
    scenario, environment, observations, mutations = _episode(scenario_id)
    assert environment.snapshot() == scenario.initial_environment
    _assert_initial_observations(scenario, environment, observations)

    assert scenario.expected_decision is expected_decision
    assert scenario.expected_final_state.phase == expected_phase
    assert scenario.expected_final_state.candidate_weight == expected_weight

    action_by_decision = {
        Decision.PROMOTE: mutations.promote_rollout,
        Decision.PAUSE: mutations.pause_rollout,
        Decision.ROLLBACK: mutations.rollback_rollout,
    }
    resulting_state = action_by_decision[expected_decision]()
    observed_final_state = observations.get_rollout_state()
    final_metrics_observation = observations.get_service_metrics()
    final_events_observation = observations.get_recent_events()

    assert resulting_state.phase == expected_phase
    assert resulting_state.candidate_weight == expected_weight
    assert observed_final_state == resulting_state
    assert environment.snapshot().rollout == observed_final_state
    assert environment.snapshot().metrics == scenario.initial_environment.metrics
    assert environment.snapshot().events == scenario.initial_environment.events
    assert final_events_observation == scenario.initial_environment.events
    if scenario_id == "transient-anomaly":
        assert [
            (event.severity.value, event.occurred_seconds_ago)
            for event in final_events_observation
        ] == [("warning", 600), ("info", 480)]
    if scenario_id == "conflicting-signals":
        final_metrics = final_metrics_observation.metrics
        assert final_metrics is not None
        assert final_metrics.candidate_error_rate < 0.01
        assert final_metrics.candidate_p95_ms <= final_metrics.stable_p95_ms * 1.10
        assert any(event.severity.value == "warning" for event in final_events_observation)
        assert any(
            event.severity.value == "critical"
            and event.occurred_seconds_ago is not None
            and event.occurred_seconds_ago <= 60
            for event in final_events_observation
        )
    if scenario_id == "insufficient-evidence":
        final_metrics = final_metrics_observation.metrics
        assert final_metrics is not None
        assert final_metrics.candidate_requests == 8
    assert final_metrics_observation.available is (
        scenario.initial_environment.metrics is not None
    )
    assert final_metrics_observation.metrics == scenario.initial_environment.metrics
    if scenario.initial_environment.metrics is None:
        assert final_metrics_observation.message is not None
        assert "unavailable" in final_metrics_observation.message.lower()
    assert environment.mutation_count == 1

    environment.reset()

    assert environment.snapshot() == scenario.initial_environment
    reset_observation = observations.get_rollout_state()
    assert reset_observation.phase == scenario.initial_environment.rollout.phase
    assert (
        reset_observation.candidate_weight
        == scenario.initial_environment.rollout.candidate_weight
    )
    reset_events_observation = observations.get_recent_events()
    assert reset_events_observation == scenario.initial_environment.events
    if scenario_id == "transient-anomaly":
        assert [
            (event.severity.value, event.occurred_seconds_ago)
            for event in reset_events_observation
        ] == [("warning", 600), ("info", 480)]
    assert environment.mutation_count == 0


def test_wrong_action_executes_without_being_blocked_by_scenario_expectation() -> None:
    scenario, environment, observations, mutations = _episode("degraded-canary")
    _assert_initial_observations(scenario, environment, observations)

    resulting_state = mutations.promote_rollout()

    assert scenario.expected_decision == Decision.ROLLBACK
    assert resulting_state.phase == "Promoted"
    assert resulting_state.candidate_weight == 100
    assert environment.snapshot().rollout == resulting_state
    assert environment.mutation_count == 1
