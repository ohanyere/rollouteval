from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval import Scenario, load_scenario
from rollouteval.environment import MutationLimitExceeded, RolloutEnvironment
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _load_scenario(scenario_id: str) -> Scenario:
    return load_scenario(SCENARIOS / f"{scenario_id}.yaml")


def _runtime(
    scenario_id: str = "degraded-canary",
) -> tuple[Scenario, RolloutEnvironment, RolloutMutationTools, RolloutObservationTools]:
    scenario = _load_scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    return (
        scenario,
        environment,
        RolloutMutationTools(environment),
        RolloutObservationTools(environment),
    )


def test_promote_changes_phase_to_promoted() -> None:
    _, environment, tools, _ = _runtime()

    tools.promote_rollout()

    assert environment.snapshot().rollout.phase == "Promoted"


def test_promote_changes_candidate_weight_to_100() -> None:
    _, environment, tools, _ = _runtime()

    tools.promote_rollout()

    assert environment.snapshot().rollout.candidate_weight == 100


def test_promote_preserves_stable_version() -> None:
    _, environment, tools, _ = _runtime()
    original_version = environment.snapshot().rollout.stable_version

    tools.promote_rollout()

    assert environment.snapshot().rollout.stable_version == original_version


def test_promote_preserves_candidate_version() -> None:
    _, environment, tools, _ = _runtime()
    original_version = environment.snapshot().rollout.candidate_version

    tools.promote_rollout()

    assert environment.snapshot().rollout.candidate_version == original_version


def test_promote_does_not_change_metrics() -> None:
    _, environment, tools, _ = _runtime()
    original_metrics = environment.snapshot().metrics

    tools.promote_rollout()

    assert environment.snapshot().metrics == original_metrics


def test_promote_does_not_change_runtime_events() -> None:
    _, environment, tools, _ = _runtime()
    original_events = environment.snapshot().events

    tools.promote_rollout()

    assert environment.snapshot().events == original_events


def test_pause_changes_phase_to_paused() -> None:
    _, environment, tools, _ = _runtime()

    tools.pause_rollout()

    assert environment.snapshot().rollout.phase == "Paused"


def test_pause_preserves_candidate_weight() -> None:
    _, environment, tools, _ = _runtime()

    tools.pause_rollout()

    assert environment.snapshot().rollout.candidate_weight == 20


def test_pause_does_not_change_metrics() -> None:
    _, environment, tools, _ = _runtime()
    original_metrics = environment.snapshot().metrics

    tools.pause_rollout()

    assert environment.snapshot().metrics == original_metrics


def test_pause_does_not_change_runtime_events() -> None:
    _, environment, tools, _ = _runtime()
    original_events = environment.snapshot().events

    tools.pause_rollout()

    assert environment.snapshot().events == original_events


def test_rollback_changes_phase_to_rolled_back() -> None:
    _, environment, tools, _ = _runtime()

    tools.rollback_rollout()

    assert environment.snapshot().rollout.phase == "RolledBack"


def test_rollback_changes_candidate_weight_to_zero() -> None:
    _, environment, tools, _ = _runtime()

    tools.rollback_rollout()

    assert environment.snapshot().rollout.candidate_weight == 0


def test_rollback_preserves_stable_version() -> None:
    _, environment, tools, _ = _runtime()
    original_version = environment.snapshot().rollout.stable_version

    tools.rollback_rollout()

    assert environment.snapshot().rollout.stable_version == original_version


def test_rollback_does_not_change_metrics() -> None:
    _, environment, tools, _ = _runtime()
    original_metrics = environment.snapshot().metrics

    tools.rollback_rollout()

    assert environment.snapshot().metrics == original_metrics


def test_rollback_does_not_change_runtime_events() -> None:
    _, environment, tools, _ = _runtime()
    original_events = environment.snapshot().events

    tools.rollback_rollout()

    assert environment.snapshot().events == original_events


def test_observation_tools_report_promoted_state_after_mutation() -> None:
    _, _, mutations, observations = _runtime()

    mutations.promote_rollout()

    state = observations.get_rollout_state()
    assert state.phase == "Promoted"
    assert state.candidate_weight == 100


def test_observation_tools_report_paused_state_after_mutation() -> None:
    _, _, mutations, observations = _runtime()

    mutations.pause_rollout()

    state = observations.get_rollout_state()
    assert state.phase == "Paused"
    assert state.candidate_weight == 20


def test_observation_tools_report_rolled_back_state_after_mutation() -> None:
    _, _, mutations, observations = _runtime()

    mutations.rollback_rollout()

    state = observations.get_rollout_state()
    assert state.phase == "RolledBack"
    assert state.candidate_weight == 0


def test_first_mutation_is_allowed() -> None:
    _, environment, tools, _ = _runtime()

    result = tools.pause_rollout()

    assert result.phase == "Paused"
    assert environment.mutation_count == 1


def test_second_mutation_exceeds_limit() -> None:
    _, _, tools, _ = _runtime()
    tools.pause_rollout()

    with pytest.raises(MutationLimitExceeded):
        tools.rollback_rollout()


def test_rejected_second_mutation_leaves_state_unchanged() -> None:
    _, environment, tools, _ = _runtime()
    tools.pause_rollout()
    state_after_first_mutation = environment.snapshot()

    with pytest.raises(MutationLimitExceeded):
        tools.rollback_rollout()

    assert environment.snapshot() == state_after_first_mutation


def test_mutation_count_increments_only_after_successful_mutation() -> None:
    _, environment, tools, _ = _runtime()
    assert environment.mutation_count == 0
    tools.pause_rollout()
    assert environment.mutation_count == 1

    with pytest.raises(MutationLimitExceeded):
        tools.promote_rollout()

    assert environment.mutation_count == 1


def test_reset_restores_mutation_count_to_zero() -> None:
    _, environment, tools, _ = _runtime()
    tools.pause_rollout()

    environment.reset()

    assert environment.mutation_count == 0


def test_mutation_is_allowed_again_after_reset() -> None:
    _, environment, tools, _ = _runtime()
    tools.pause_rollout()
    environment.reset()

    result = tools.rollback_rollout()

    assert result.phase == "RolledBack"
    assert environment.mutation_count == 1


def test_reset_after_promote_restores_original_phase_and_weight() -> None:
    _, environment, tools, _ = _runtime()
    tools.promote_rollout()

    environment.reset()

    assert environment.snapshot().rollout.phase == "Progressing"
    assert environment.snapshot().rollout.candidate_weight == 20


def test_reset_after_pause_restores_original_phase_and_weight() -> None:
    _, environment, tools, _ = _runtime()
    tools.pause_rollout()

    environment.reset()

    assert environment.snapshot().rollout.phase == "Progressing"
    assert environment.snapshot().rollout.candidate_weight == 20


def test_reset_after_rollback_restores_original_phase_and_weight() -> None:
    _, environment, tools, _ = _runtime()
    tools.rollback_rollout()

    environment.reset()

    assert environment.snapshot().rollout.phase == "Progressing"
    assert environment.snapshot().rollout.candidate_weight == 20


def test_mutation_does_not_change_original_scenario() -> None:
    scenario, _, tools, _ = _runtime()
    tools.rollback_rollout()

    assert scenario.initial_environment.rollout.phase == "Progressing"
    assert scenario.initial_environment.rollout.candidate_weight == 20


def test_mutation_result_cannot_change_environment_state() -> None:
    _, environment, tools, _ = _runtime()
    result = tools.promote_rollout()

    with pytest.raises(ValidationError):
        result.candidate_weight = 0

    assert environment.snapshot().rollout.candidate_weight == 100


def test_separate_environments_do_not_share_mutation_state() -> None:
    scenario = _load_scenario("degraded-canary")
    first_environment = RolloutEnvironment(scenario)
    second_environment = RolloutEnvironment(scenario)
    first_tools = RolloutMutationTools(first_environment)

    first_tools.rollback_rollout()

    assert first_environment.mutation_count == 1
    assert second_environment.mutation_count == 0
    assert second_environment.snapshot().rollout.candidate_weight == 20


def test_promote_executes_even_when_degraded_canary_expects_rollback() -> None:
    scenario, environment, tools, _ = _runtime("degraded-canary")

    result = tools.promote_rollout()

    assert scenario.expected_decision.value == "rollback"
    assert result.phase == "Promoted"
    assert environment.snapshot().rollout.candidate_weight == 100


def test_rollback_executes_even_when_healthy_canary_expects_promote() -> None:
    scenario, environment, tools, _ = _runtime("healthy-canary")

    result = tools.rollback_rollout()

    assert scenario.expected_decision.value == "promote"
    assert result.phase == "RolledBack"
    assert environment.snapshot().rollout.candidate_weight == 0


def test_telemetry_unavailable_can_be_paused_without_fabricating_metrics() -> None:
    _, environment, tools, _ = _runtime("telemetry-unavailable")
    assert environment.snapshot().metrics is None

    tools.pause_rollout()

    assert environment.snapshot().metrics is None


def test_transient_anomaly_events_remain_intact_after_mutation() -> None:
    scenario, environment, tools, _ = _runtime("transient-anomaly")
    original_events = scenario.initial_environment.events

    tools.pause_rollout()

    assert environment.snapshot().events == original_events


def test_conflicting_signals_events_remain_intact_after_mutation() -> None:
    scenario, environment, tools, _ = _runtime("conflicting-signals")
    original_events = scenario.initial_environment.events

    tools.pause_rollout()

    assert environment.snapshot().events == original_events
