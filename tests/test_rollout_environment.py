from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval import Scenario, load_scenario
from rollouteval.domain.scenario import InitialEnvironment, ServiceMetrics
from rollouteval.environment import RolloutEnvironment


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _load_scenario(scenario_id: str) -> Scenario:
    return load_scenario(SCENARIOS / f"{scenario_id}.yaml")


def _replace_rollout(environment: RolloutEnvironment, **updates: object) -> None:
    state = environment._state
    environment._state = state.model_copy(
        update={"rollout": state.rollout.model_copy(update=updates)}
    )


def test_degraded_canary_creates_rollout_environment() -> None:
    assert isinstance(RolloutEnvironment(_load_scenario("degraded-canary")), RolloutEnvironment)


def test_initial_rollout_state_matches_scenario() -> None:
    scenario = _load_scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)

    assert environment.snapshot().rollout == scenario.initial_environment.rollout


def test_initial_metrics_match_scenario() -> None:
    scenario = _load_scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)

    assert environment.snapshot().metrics == scenario.initial_environment.metrics


def test_initial_runtime_events_match_scenario() -> None:
    scenario = _load_scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)

    assert environment.snapshot().events == scenario.initial_environment.events


def test_snapshot_returns_current_rollout_state() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))

    assert environment.snapshot().rollout.candidate_weight == 20
    assert environment.snapshot().rollout.phase == "Progressing"


def test_snapshot_returns_metrics() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))

    assert isinstance(environment.snapshot().metrics, ServiceMetrics)


def test_snapshot_returns_runtime_events() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))

    assert environment.snapshot().events


def test_telemetry_unavailable_environment_has_no_metrics() -> None:
    environment = RolloutEnvironment(_load_scenario("telemetry-unavailable"))

    assert environment.snapshot().metrics is None


def test_transient_anomaly_preserves_event_recency() -> None:
    environment = RolloutEnvironment(_load_scenario("transient-anomaly"))

    assert [event.occurred_seconds_ago for event in environment.snapshot().events] == [600, 480]


def test_conflicting_signals_preserves_warning_and_critical_events() -> None:
    environment = RolloutEnvironment(_load_scenario("conflicting-signals"))
    severities = {event.severity for event in environment.snapshot().events}

    assert {"warning", "critical"} <= {severity.value for severity in severities}


def test_runtime_state_mutation_does_not_mutate_scenario() -> None:
    scenario = _load_scenario("degraded-canary")
    environment = RolloutEnvironment(scenario)

    _replace_rollout(environment, candidate_weight=0, phase="RolledBack")

    assert scenario.initial_environment.rollout.candidate_weight == 20
    assert scenario.initial_environment.rollout.phase == "Progressing"


def test_mutating_returned_snapshot_does_not_mutate_environment() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))
    returned_snapshot = environment.snapshot()

    with pytest.raises(ValidationError):
        returned_snapshot.rollout.candidate_weight = 0
    assert environment.snapshot().rollout.candidate_weight == 20


def test_reset_restores_candidate_weight() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))
    _replace_rollout(environment, candidate_weight=0)

    environment.reset()

    assert environment.snapshot().rollout.candidate_weight == 20


def test_reset_restores_phase() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))
    _replace_rollout(environment, phase="RolledBack")

    environment.reset()

    assert environment.snapshot().rollout.phase == "Progressing"


def test_reset_restores_metrics_after_runtime_mutation() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))
    environment._state = environment._state.model_copy(update={"metrics": None})

    environment.reset()

    assert environment.snapshot().metrics == _load_scenario("degraded-canary").initial_environment.metrics


def test_reset_restores_runtime_events_after_runtime_mutation() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))
    environment._state = environment._state.model_copy(update={"events": ()})

    environment.reset()

    assert environment.snapshot().events == _load_scenario("degraded-canary").initial_environment.events


def test_reset_restores_unavailable_metrics_to_none() -> None:
    environment = RolloutEnvironment(_load_scenario("telemetry-unavailable"))
    environment._state = environment._state.model_copy(
        update={"metrics": _load_scenario("degraded-canary").initial_environment.metrics}
    )

    environment.reset()

    assert environment.snapshot().metrics is None


def test_environments_from_same_scenario_have_isolated_state() -> None:
    scenario = _load_scenario("degraded-canary")
    first = RolloutEnvironment(scenario)
    second = RolloutEnvironment(scenario)
    _replace_rollout(first, candidate_weight=0)

    assert second.snapshot().rollout.candidate_weight == 20


def test_reset_is_repeatable() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))
    _replace_rollout(environment, candidate_weight=0, phase="RolledBack")

    environment.reset()
    first_reset = environment.snapshot()
    _replace_rollout(environment, candidate_weight=100, phase="Promoted")
    environment.reset()

    assert environment.snapshot() == first_reset


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
def test_all_scenarios_instantiate_environments(scenario_id: str) -> None:
    environment = RolloutEnvironment(_load_scenario(scenario_id))

    assert isinstance(environment.snapshot(), InitialEnvironment)
