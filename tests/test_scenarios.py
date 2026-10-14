from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval import Decision, Scenario, load_scenario
from rollouteval.domain.scenario import InitialEnvironment, ServiceMetrics


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SCENARIO = ROOT / "scenarios" / "degraded-canary.yaml"
HEALTHY_SCENARIO = ROOT / "scenarios" / "healthy-canary.yaml"
INSUFFICIENT_EVIDENCE_SCENARIO = ROOT / "scenarios" / "insufficient-evidence.yaml"
TRANSIENT_ANOMALY_SCENARIO = ROOT / "scenarios" / "transient-anomaly.yaml"
CONFLICTING_SIGNALS_SCENARIO = ROOT / "scenarios" / "conflicting-signals.yaml"
TELEMETRY_UNAVAILABLE_SCENARIO = ROOT / "scenarios" / "telemetry-unavailable.yaml"


def _scenario_data() -> dict[str, object]:
    return load_scenario(GOLDEN_SCENARIO).model_dump(mode="python")


def test_degraded_canary_loads_successfully() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert isinstance(scenario, Scenario)


def test_degraded_canary_id_is_correct() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert scenario.id == "degraded-canary"


def test_expected_decision_is_rollback() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert scenario.expected_decision is Decision.ROLLBACK


def test_required_evidence_includes_rollout_state() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert "rollout_state" in scenario.required_evidence


def test_required_evidence_includes_service_metrics() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert "service_metrics" in scenario.required_evidence


def test_expected_final_state_is_rolled_back_with_zero_weight() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert scenario.expected_final_state.phase == "RolledBack"
    assert scenario.expected_final_state.candidate_weight == 0


def test_candidate_request_count_is_450() -> None:
    scenario = load_scenario(GOLDEN_SCENARIO)

    assert scenario.initial_environment.metrics.candidate_requests == 450


def test_candidate_weight_above_100_is_rejected() -> None:
    data = _scenario_data()
    data["initial_environment"]["rollout"]["candidate_weight"] = 101

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_negative_candidate_weight_is_rejected() -> None:
    data = _scenario_data()
    data["initial_environment"]["rollout"]["candidate_weight"] = -1

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_negative_stable_error_rate_is_rejected() -> None:
    data = _scenario_data()
    data["initial_environment"]["metrics"]["stable_error_rate"] = -0.001

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_negative_candidate_error_rate_is_rejected() -> None:
    data = _scenario_data()
    data["initial_environment"]["metrics"]["candidate_error_rate"] = -0.001

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_invalid_decision_value_is_rejected() -> None:
    data = _scenario_data()
    data["expected_decision"] = "continue"

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_healthy_canary_loads_successfully() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert isinstance(scenario, Scenario)


def test_healthy_canary_id_is_correct() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert scenario.id == "healthy-canary"


def test_healthy_canary_expected_decision_is_promote() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert scenario.expected_decision is Decision.PROMOTE


def test_healthy_canary_requires_rollout_state_evidence() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert "rollout_state" in scenario.required_evidence


def test_healthy_canary_requires_service_metrics_evidence() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert "service_metrics" in scenario.required_evidence


def test_healthy_canary_expected_final_phase_is_promoted() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert scenario.expected_final_state.phase == "Promoted"


def test_healthy_canary_expected_final_weight_is_100() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert scenario.expected_final_state.candidate_weight == 100


def test_healthy_canary_candidate_request_count_is_500() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert scenario.initial_environment.metrics.candidate_requests == 500


def test_healthy_canary_candidate_error_rate_is_below_one_percent() -> None:
    scenario = load_scenario(HEALTHY_SCENARIO)

    assert scenario.initial_environment.metrics.candidate_error_rate < 0.01


def test_healthy_canary_candidate_p95_is_within_ten_percent_of_stable() -> None:
    metrics = load_scenario(HEALTHY_SCENARIO).initial_environment.metrics

    assert metrics.candidate_p95_ms <= metrics.stable_p95_ms * 1.10


def test_insufficient_evidence_loads_successfully() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert isinstance(scenario, Scenario)


def test_insufficient_evidence_id_is_correct() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert scenario.id == "insufficient-evidence"


def test_insufficient_evidence_expected_decision_is_pause() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert scenario.expected_decision is Decision.PAUSE


def test_insufficient_evidence_requires_rollout_state_evidence() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert "rollout_state" in scenario.required_evidence


def test_insufficient_evidence_requires_service_metrics_evidence() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert "service_metrics" in scenario.required_evidence


def test_insufficient_evidence_expected_final_phase_is_paused() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert scenario.expected_final_state.phase == "Paused"


def test_insufficient_evidence_expected_final_weight_remains_20() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert scenario.expected_final_state.candidate_weight == 20


def test_insufficient_evidence_candidate_request_count_is_8() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)

    assert scenario.initial_environment.metrics.candidate_requests == 8


def test_insufficient_evidence_requests_are_lower_than_decisive_cases() -> None:
    request_count = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO).initial_environment.metrics.candidate_requests
    degraded_requests = load_scenario(GOLDEN_SCENARIO).initial_environment.metrics.candidate_requests
    healthy_requests = load_scenario(HEALTHY_SCENARIO).initial_environment.metrics.candidate_requests

    assert request_count < degraded_requests
    assert request_count < healthy_requests


def test_insufficient_evidence_candidate_error_rate_is_materially_higher() -> None:
    metrics = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO).initial_environment.metrics

    assert metrics.candidate_error_rate > metrics.stable_error_rate * 10


def test_insufficient_evidence_candidate_p95_is_materially_higher() -> None:
    metrics = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO).initial_environment.metrics

    assert metrics.candidate_p95_ms > metrics.stable_p95_ms * 4


def test_degraded_metrics_with_small_sample_still_expect_pause() -> None:
    scenario = load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO)
    metrics = scenario.initial_environment.metrics

    assert metrics.candidate_error_rate > metrics.stable_error_rate
    assert metrics.candidate_p95_ms > metrics.stable_p95_ms
    assert metrics.candidate_requests == 8
    assert scenario.expected_decision is Decision.PAUSE


def test_transient_anomaly_loads_successfully() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert isinstance(scenario, Scenario)


def test_transient_anomaly_id_is_correct() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert scenario.id == "transient-anomaly"


def test_transient_anomaly_expected_decision_is_promote() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert scenario.expected_decision is Decision.PROMOTE


def test_transient_anomaly_requires_rollout_state_evidence() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert "rollout_state" in scenario.required_evidence


def test_transient_anomaly_requires_service_metrics_evidence() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert "service_metrics" in scenario.required_evidence


def test_transient_anomaly_requires_runtime_events_evidence() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert "runtime_events" in scenario.required_evidence


def test_transient_anomaly_expected_final_phase_is_promoted() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert scenario.expected_final_state.phase == "Promoted"


def test_transient_anomaly_expected_final_weight_is_100() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert scenario.expected_final_state.candidate_weight == 100


def test_transient_anomaly_candidate_request_count_is_900() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)

    assert scenario.initial_environment.metrics.candidate_requests == 900


def test_transient_anomaly_candidate_error_rate_is_below_one_percent() -> None:
    metrics = load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.metrics

    assert metrics.candidate_error_rate < 0.01


def test_transient_anomaly_candidate_p95_is_within_ten_percent_of_stable() -> None:
    metrics = load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.metrics

    assert metrics.candidate_p95_ms <= metrics.stable_p95_ms * 1.10


def test_transient_anomaly_contains_historical_warning_event() -> None:
    events = load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.events

    assert any(
        event.severity.value == "warning" and event.occurred_seconds_ago == 600
        for event in events
    )


def test_transient_anomaly_contains_later_recovery_event() -> None:
    events = load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.events

    assert any(
        event.severity.value == "info"
        and "returned to the healthy range" in event.message
        and event.occurred_seconds_ago == 480
        for event in events
    )


def test_transient_anomaly_warning_precedes_recovery_event() -> None:
    events = load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.events
    warning = next(event for event in events if event.severity.value == "warning")
    recovery = next(event for event in events if event.severity.value == "info")

    assert warning.occurred_seconds_ago > recovery.occurred_seconds_ago


def test_transient_anomaly_has_no_current_critical_event() -> None:
    events = load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.events

    assert not any(event.severity.value == "critical" for event in events)


def test_transient_warning_still_expects_promotion() -> None:
    scenario = load_scenario(TRANSIENT_ANOMALY_SCENARIO)
    has_historical_warning = any(
        event.severity.value == "warning" and event.occurred_seconds_ago is not None
        for event in scenario.initial_environment.events
    )

    assert has_historical_warning
    assert scenario.expected_decision is Decision.PROMOTE


def test_negative_runtime_event_age_is_rejected() -> None:
    data = load_scenario(TRANSIENT_ANOMALY_SCENARIO).model_dump(mode="python")
    data["initial_environment"]["events"][0]["occurred_seconds_ago"] = -1

    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_existing_scenarios_without_event_age_still_load() -> None:
    existing_scenarios = (
        GOLDEN_SCENARIO,
        HEALTHY_SCENARIO,
        INSUFFICIENT_EVIDENCE_SCENARIO,
    )

    for path in existing_scenarios:
        scenario = load_scenario(path)
        assert all(event.occurred_seconds_ago is None for event in scenario.initial_environment.events)


def test_conflicting_signals_loads_successfully() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert isinstance(scenario, Scenario)


def test_conflicting_signals_id_is_correct() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert scenario.id == "conflicting-signals"


def test_conflicting_signals_expected_decision_is_pause() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert scenario.expected_decision is Decision.PAUSE


def test_conflicting_signals_requires_rollout_state_evidence() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert "rollout_state" in scenario.required_evidence


def test_conflicting_signals_requires_service_metrics_evidence() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert "service_metrics" in scenario.required_evidence


def test_conflicting_signals_requires_runtime_events_evidence() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert "runtime_events" in scenario.required_evidence


def test_conflicting_signals_expected_final_phase_is_paused() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert scenario.expected_final_state.phase == "Paused"


def test_conflicting_signals_expected_final_weight_remains_20() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert scenario.expected_final_state.candidate_weight == 20


def test_conflicting_signals_candidate_request_count_is_700() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)

    assert scenario.initial_environment.metrics.candidate_requests == 700


def test_conflicting_signals_candidate_error_rate_is_below_one_percent() -> None:
    metrics = load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.metrics

    assert metrics.candidate_error_rate < 0.01


def test_conflicting_signals_candidate_p95_is_within_ten_percent_of_stable() -> None:
    metrics = load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.metrics

    assert metrics.candidate_p95_ms <= metrics.stable_p95_ms * 1.10


def test_conflicting_signals_contains_warning_event() -> None:
    events = load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.events

    assert any(event.severity.value == "warning" for event in events)


def test_conflicting_signals_contains_critical_event() -> None:
    events = load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.events

    assert any(event.severity.value == "critical" for event in events)


def test_conflicting_signals_has_recent_critical_event() -> None:
    events = load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.events

    assert any(
        event.severity.value == "critical"
        and event.occurred_seconds_ago is not None
        and event.occurred_seconds_ago <= 60
        for event in events
    )


def test_conflicting_signals_event_text_indicates_candidate_instability() -> None:
    messages = [
        event.message.lower()
        for event in load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.events
    ]

    assert any("restarted unexpectedly" in message for message in messages)
    assert any("readiness probe failed" in message for message in messages)


def test_conflicting_signals_healthy_metrics_coexist_with_runtime_instability() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)
    metrics = scenario.initial_environment.metrics
    events = scenario.initial_environment.events

    assert metrics.candidate_error_rate < 0.01
    assert metrics.candidate_p95_ms <= metrics.stable_p95_ms * 1.10
    assert any(event.severity.value == "critical" for event in events)


def test_conflicting_signals_healthy_metrics_still_expect_pause() -> None:
    scenario = load_scenario(CONFLICTING_SIGNALS_SCENARIO)
    metrics = scenario.initial_environment.metrics

    assert metrics.candidate_error_rate < 0.01
    assert metrics.candidate_p95_ms <= metrics.stable_p95_ms * 1.10
    assert scenario.expected_decision is Decision.PAUSE


def test_previous_golden_scenarios_still_load() -> None:
    previous_scenarios = (
        GOLDEN_SCENARIO,
        HEALTHY_SCENARIO,
        INSUFFICIENT_EVIDENCE_SCENARIO,
        TRANSIENT_ANOMALY_SCENARIO,
    )

    for path in previous_scenarios:
        assert isinstance(load_scenario(path), Scenario)


def test_telemetry_unavailable_loads_successfully() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert isinstance(scenario, Scenario)


def test_telemetry_unavailable_id_is_correct() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.id == "telemetry-unavailable"


def test_telemetry_unavailable_expected_decision_is_pause() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.expected_decision is Decision.PAUSE


def test_telemetry_unavailable_requires_rollout_state_evidence() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert "rollout_state" in scenario.required_evidence


def test_telemetry_unavailable_requires_service_metrics_evidence() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert "service_metrics" in scenario.required_evidence


def test_telemetry_unavailable_expected_final_phase_is_paused() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.expected_final_state.phase == "Paused"


def test_telemetry_unavailable_expected_final_weight_remains_20() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.expected_final_state.candidate_weight == 20


def test_telemetry_unavailable_metrics_are_none() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.initial_environment.metrics is None


def test_telemetry_unavailable_has_warning_about_missing_metrics() -> None:
    events = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO).initial_environment.events

    assert any(
        event.severity.value == "warning"
        and "metrics" in event.message.lower()
        and "unavailable" in event.message.lower()
        for event in events
    )


def test_telemetry_unavailable_warning_is_recent() -> None:
    events = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO).initial_environment.events

    assert any(
        event.severity.value == "warning" and event.occurred_seconds_ago == 15
        for event in events
    )


def test_telemetry_unavailable_rollout_is_still_progressing() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.initial_environment.rollout.phase == "Progressing"


def test_telemetry_unavailable_has_no_fabricated_metrics_object() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert not isinstance(scenario.initial_environment.metrics, ServiceMetrics)


def test_telemetry_unavailable_still_expects_pause() -> None:
    scenario = load_scenario(TELEMETRY_UNAVAILABLE_SCENARIO)

    assert scenario.initial_environment.metrics is None
    assert "service_metrics" in scenario.required_evidence
    assert scenario.expected_decision is Decision.PAUSE


def test_scenarios_with_metrics_still_load() -> None:
    scenarios_with_metrics = (
        GOLDEN_SCENARIO,
        HEALTHY_SCENARIO,
        INSUFFICIENT_EVIDENCE_SCENARIO,
        TRANSIENT_ANOMALY_SCENARIO,
        CONFLICTING_SIGNALS_SCENARIO,
    )

    for path in scenarios_with_metrics:
        assert isinstance(load_scenario(path).initial_environment.metrics, ServiceMetrics)


def test_degraded_canary_metrics_remain_available() -> None:
    assert isinstance(load_scenario(GOLDEN_SCENARIO).initial_environment.metrics, ServiceMetrics)


def test_healthy_canary_metrics_remain_available() -> None:
    assert isinstance(load_scenario(HEALTHY_SCENARIO).initial_environment.metrics, ServiceMetrics)


def test_insufficient_evidence_metrics_remain_available() -> None:
    assert isinstance(
        load_scenario(INSUFFICIENT_EVIDENCE_SCENARIO).initial_environment.metrics,
        ServiceMetrics,
    )


def test_transient_anomaly_metrics_remain_available() -> None:
    assert isinstance(
        load_scenario(TRANSIENT_ANOMALY_SCENARIO).initial_environment.metrics,
        ServiceMetrics,
    )


def test_conflicting_signals_metrics_remain_available() -> None:
    assert isinstance(
        load_scenario(CONFLICTING_SIGNALS_SCENARIO).initial_environment.metrics,
        ServiceMetrics,
    )


def test_initial_environment_accepts_null_metrics() -> None:
    environment = InitialEnvironment.model_validate(
        {
            "rollout": {
                "stable_version": "v1",
                "candidate_version": "v2",
                "candidate_weight": 20,
                "phase": "Progressing",
            },
            "metrics": None,
            "events": [],
        }
    )

    assert environment.metrics is None
