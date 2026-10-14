from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval import Scenario, load_scenario
from rollouteval.domain.scenario import RuntimeEvent, ServiceMetrics
from rollouteval.environment import RolloutEnvironment
from rollouteval.tools import RolloutObservationTools, ServiceMetricsObservation


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _load_scenario(scenario_id: str) -> Scenario:
    return load_scenario(SCENARIOS / f"{scenario_id}.yaml")


def _tools(scenario_id: str = "degraded-canary") -> tuple[RolloutEnvironment, RolloutObservationTools]:
    environment = RolloutEnvironment(_load_scenario(scenario_id))
    return environment, RolloutObservationTools(environment)


def test_observation_tools_can_be_created_for_environment() -> None:
    environment = RolloutEnvironment(_load_scenario("degraded-canary"))

    assert isinstance(RolloutObservationTools(environment), RolloutObservationTools)


def test_get_rollout_state_returns_stable_version() -> None:
    _, tools = _tools()

    assert tools.get_rollout_state().stable_version == "v1"


def test_get_rollout_state_returns_candidate_version() -> None:
    _, tools = _tools()

    assert tools.get_rollout_state().candidate_version == "v2"


def test_get_rollout_state_returns_candidate_weight() -> None:
    _, tools = _tools()

    assert tools.get_rollout_state().candidate_weight == 20


def test_get_rollout_state_returns_phase() -> None:
    _, tools = _tools()

    assert tools.get_rollout_state().phase == "Progressing"


def test_mutating_returned_rollout_state_does_not_mutate_environment() -> None:
    environment, tools = _tools()
    observation = tools.get_rollout_state()

    with pytest.raises(ValidationError):
        observation.candidate_weight = 0
    assert environment.snapshot().rollout.candidate_weight == 20


def test_get_service_metrics_reports_available_for_degraded_canary() -> None:
    _, tools = _tools()

    assert tools.get_service_metrics().available is True


def test_available_metrics_match_environment_metrics() -> None:
    environment, tools = _tools()
    observation = tools.get_service_metrics()

    assert observation.metrics == environment.snapshot().metrics


def test_mutating_returned_metrics_does_not_mutate_environment() -> None:
    environment, tools = _tools()
    observation = tools.get_service_metrics()

    with pytest.raises(ValidationError):
        observation.metrics.candidate_requests = 0
    assert environment.snapshot().metrics.candidate_requests == 450


def test_telemetry_unavailable_reports_metrics_unavailable() -> None:
    _, tools = _tools("telemetry-unavailable")

    observation = tools.get_service_metrics()

    assert observation.available is False
    assert observation.metrics is None


def test_telemetry_unavailable_does_not_fabricate_metrics() -> None:
    _, tools = _tools("telemetry-unavailable")

    observation = tools.get_service_metrics()

    assert observation.metrics is None
    assert not isinstance(observation.metrics, ServiceMetrics)


def test_unavailable_metrics_response_has_explicit_message() -> None:
    _, tools = _tools("telemetry-unavailable")

    assert "unavailable" in tools.get_service_metrics().message.lower()


def test_get_recent_events_returns_runtime_events() -> None:
    environment, tools = _tools()

    assert tools.get_recent_events() == environment.snapshot().events


def test_transient_anomaly_event_recency_is_preserved() -> None:
    _, tools = _tools("transient-anomaly")

    assert [event.occurred_seconds_ago for event in tools.get_recent_events()] == [600, 480]


def test_conflicting_signals_warning_and_critical_events_are_preserved() -> None:
    _, tools = _tools("conflicting-signals")

    assert [event.severity.value for event in tools.get_recent_events()] == ["warning", "critical"]


def test_mutating_returned_events_does_not_mutate_environment() -> None:
    environment, tools = _tools()
    events = tools.get_recent_events()

    with pytest.raises(ValidationError):
        events[0].message = "changed"
    assert environment.snapshot().events[0].message == "Checkout canary is receiving production traffic."


def test_get_rollout_state_is_side_effect_free() -> None:
    environment, tools = _tools()
    before = environment.snapshot()

    tools.get_rollout_state()

    assert environment.snapshot() == before


def test_get_service_metrics_is_side_effect_free() -> None:
    environment, tools = _tools()
    before = environment.snapshot()

    tools.get_service_metrics()

    assert environment.snapshot() == before


def test_get_recent_events_is_side_effect_free() -> None:
    environment, tools = _tools()
    before = environment.snapshot()

    tools.get_recent_events()

    assert environment.snapshot() == before


def test_repeated_observation_calls_do_not_change_environment() -> None:
    environment, tools = _tools()
    before = environment.snapshot()

    tools.get_rollout_state()
    tools.get_service_metrics()
    tools.get_recent_events()
    tools.get_rollout_state()
    tools.get_service_metrics()
    tools.get_recent_events()

    assert environment.snapshot() == before


def test_tool_outputs_do_not_contain_expected_decision() -> None:
    scenario = _load_scenario("degraded-canary")
    _, tools = _tools()
    outputs = [
        tools.get_rollout_state().model_dump(mode="json"),
        tools.get_service_metrics().model_dump(mode="json"),
        [event.model_dump(mode="json") for event in tools.get_recent_events()],
    ]

    assert all("expected_decision" not in str(output) for output in outputs)
    assert scenario.expected_decision.value not in str(outputs)


def test_tool_outputs_do_not_contain_expected_final_state() -> None:
    scenario = _load_scenario("degraded-canary")
    _, tools = _tools()
    outputs = [
        tools.get_rollout_state().model_dump(mode="json"),
        tools.get_service_metrics().model_dump(mode="json"),
        [event.model_dump(mode="json") for event in tools.get_recent_events()],
    ]

    assert all("expected_final_state" not in str(output) for output in outputs)
    assert scenario.expected_final_state.phase not in str(outputs)


def test_tool_outputs_do_not_contain_required_evidence() -> None:
    scenario = _load_scenario("degraded-canary")
    _, tools = _tools()
    outputs = [
        tools.get_rollout_state().model_dump(mode="json"),
        tools.get_service_metrics().model_dump(mode="json"),
        [event.model_dump(mode="json") for event in tools.get_recent_events()],
    ]

    assert all("required_evidence" not in str(output) for output in outputs)
    assert not any(item in str(outputs) for item in scenario.required_evidence)


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
def test_observation_tools_work_across_all_scenarios(scenario_id: str) -> None:
    _, tools = _tools(scenario_id)

    assert tools.get_rollout_state().candidate_weight == 20
    assert isinstance(tools.get_service_metrics(), ServiceMetricsObservation)
    assert all(isinstance(event, RuntimeEvent) for event in tools.get_recent_events())


def test_available_metrics_observation_requires_metrics() -> None:
    with pytest.raises(ValidationError):
        ServiceMetricsObservation(available=True, metrics=None)


def test_unavailable_metrics_observation_rejects_metrics() -> None:
    metrics = _load_scenario("degraded-canary").initial_environment.metrics

    with pytest.raises(ValidationError):
        ServiceMetricsObservation(available=False, metrics=metrics, message="Unavailable")


def test_unavailable_metrics_observation_requires_message() -> None:
    with pytest.raises(ValidationError):
        ServiceMetricsObservation(available=False, metrics=None)
