from pathlib import Path
from typing import Callable

import pytest
from pydantic import ValidationError

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentModel,
    AgentModelResponse,
    AgentOutput,
    AgentRunner,
    AgentStepLimitExceeded,
    AgentTask,
    AgentToolCall,
    AgentToolInteraction,
    AgentToolName,
)
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.observability import (
    TraceEvent,
    TraceEventType,
    TraceRecorder,
)
from rollouteval.scenarios import load_scenario
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_IDS = (
    "degraded-canary",
    "healthy-canary",
    "insufficient-evidence",
    "transient-anomaly",
    "conflicting-signals",
    "telemetry-unavailable",
)
DECISION_TO_MUTATION = {
    Decision.PROMOTE: AgentToolName.PROMOTE_ROLLOUT,
    Decision.PAUSE: AgentToolName.PAUSE_ROLLOUT,
    Decision.ROLLBACK: AgentToolName.ROLLBACK_ROLLOUT,
}
EVIDENCE_TO_TOOL = {
    "rollout_state": AgentToolName.GET_ROLLOUT_STATE,
    "service_metrics": AgentToolName.GET_SERVICE_METRICS,
    "runtime_events": AgentToolName.GET_RECENT_EVENTS,
}
COMPLETION_OR_FAILURE = (
    TraceEventType.MODEL_COMPLETED,
    TraceEventType.MODEL_FAILED,
    TraceEventType.TOOL_COMPLETED,
    TraceEventType.TOOL_FAILED,
    TraceEventType.RUN_COMPLETED,
    TraceEventType.RUN_FAILED,
)
START_EVENTS = (
    TraceEventType.RUN_STARTED,
    TraceEventType.MODEL_STARTED,
    TraceEventType.TOOL_STARTED,
)


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self.values)


class IncrementingClock:
    def __init__(self, increment: float = 0.01) -> None:
        self.value = 0.0
        self.increment = increment
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        self.value += self.increment
        return self.value


class ScriptedModel:
    def __init__(self, outputs: list[AgentOutput]) -> None:
        self.outputs = iter(outputs)
        self.histories: list[tuple[AgentToolInteraction, ...]] = []

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentModelResponse:
        self.histories.append(interactions)
        return AgentModelResponse(output=next(self.outputs))


class FailingModel:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentModelResponse:
        raise self.error


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _answer(decision: Decision = Decision.PAUSE) -> AgentFinalAnswer:
    return AgentFinalAnswer(
        kind="final", decision=decision, explanation="Deterministic timing test."
    )


def _runner(
    model: AgentModel,
    *,
    scenario_id: str = "degraded-canary",
    max_steps: int = 8,
    clock: Callable[[], float] = lambda: -1.0,
) -> tuple[AgentRunner, RolloutEnvironment]:
    environment = RolloutEnvironment(_scenario(scenario_id))
    runner = AgentRunner(
        model,
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
        max_steps=max_steps,
        clock=clock,
    )
    return runner, environment


def _outputs_for_scenario(scenario_id: str) -> list[AgentOutput]:
    scenario = _scenario(scenario_id)
    outputs: list[AgentOutput] = [
        _call(EVIDENCE_TO_TOOL[evidence_id])
        for evidence_id in scenario.required_evidence
    ]
    outputs.extend(
        [
            _call(DECISION_TO_MUTATION[scenario.expected_decision]),
            _call(AgentToolName.GET_ROLLOUT_STATE),
            _answer(scenario.expected_decision),
        ]
    )
    return outputs


def _duration_event(
    event_type: TraceEventType,
    duration_ms: float | None,
    *,
    tool_name: AgentToolName | None = None,
) -> TraceEvent:
    return TraceEvent(
        sequence=0,
        event_type=event_type,
        step=1 if event_type not in {TraceEventType.RUN_STARTED, TraceEventType.RUN_COMPLETED, TraceEventType.RUN_FAILED} else None,
        tool_name=tool_name,
        duration_ms=duration_ms,
    )


@pytest.mark.parametrize("event_type", COMPLETION_OR_FAILURE)
def test_completed_and_failed_events_accept_nonnegative_duration(
    event_type: TraceEventType,
) -> None:
    tool_name = (
        AgentToolName.GET_ROLLOUT_STATE
        if event_type in {TraceEventType.TOOL_COMPLETED, TraceEventType.TOOL_FAILED}
        else None
    )

    event = _duration_event(event_type, 0.0, tool_name=tool_name)

    assert event.duration_ms == 0.0


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _duration_event(TraceEventType.MODEL_COMPLETED, -0.01)


@pytest.mark.parametrize("event_type", START_EVENTS)
def test_start_events_reject_duration(event_type: TraceEventType) -> None:
    tool_name = AgentToolName.GET_ROLLOUT_STATE if event_type is TraceEventType.TOOL_STARTED else None

    with pytest.raises(ValidationError, match="duration_ms is only valid"):
        _duration_event(event_type, 1.0, tool_name=tool_name)


def test_trace_event_remains_immutable_with_duration():
    event = _duration_event(TraceEventType.MODEL_COMPLETED, 12.5)

    with pytest.raises(ValidationError):
        event.duration_ms = 9.0


def test_recorder_accepts_and_preserves_producer_supplied_duration():
    recorder = TraceRecorder("run-001", "scenario")

    recorded = recorder.record(TraceEventType.MODEL_COMPLETED, step=1, duration_ms=24.75)

    assert recorded.duration_ms == 24.75
    assert recorder.snapshot().events[0].duration_ms == 24.75


def test_recorder_does_not_calculate_duration_itself():
    recorder = TraceRecorder("run-001", "scenario")

    event = recorder.record(TraceEventType.MODEL_COMPLETED, step=1)

    assert event.duration_ms is None


def test_immediate_final_records_model_and_total_run_durations_without_order_change():
    clock = FakeClock([2.0, 2.1, 2.225, 2.5])
    runner, _ = _runner(ScriptedModel([_answer()]), clock=clock)
    recorder = TraceRecorder("run-001", "degraded-canary")

    runner.run(AgentTask(task="test"), trace_recorder=recorder)
    events = recorder.snapshot().events

    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]
    assert events[2].duration_ms == pytest.approx(125.0)
    assert events[3].duration_ms == pytest.approx(500.0)
    assert clock.calls == 4


def test_one_tool_run_records_independent_model_tool_and_total_durations():
    clock = FakeClock([0.0, 0.01, 0.04, 0.05, 0.09, 0.10, 0.16, 0.20])
    runner, _ = _runner(
        ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()]),
        clock=clock,
    )
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(AgentTask(task="test"), trace_recorder=recorder)
    events = recorder.snapshot().events

    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]
    assert events[2].duration_ms == pytest.approx(30.0)
    assert events[4].duration_ms == pytest.approx(40.0)
    assert events[6].duration_ms == pytest.approx(60.0)
    assert events[7].duration_ms == pytest.approx(200.0)
    assert events[3].step == events[4].step == 1
    assert events[1].step == events[2].step == 1
    assert events[5].step == events[6].step == result.steps == 2


def test_injected_fake_clock_controls_timing_without_sleeping():
    clock = FakeClock([10.0, 10.2, 10.25, 10.3])
    runner, _ = _runner(ScriptedModel([_answer()]), clock=clock)
    recorder = TraceRecorder("run-001", "scenario")

    runner.run(AgentTask(task="test"), trace_recorder=recorder)

    assert recorder.snapshot().events[2].duration_ms == pytest.approx(50.0)
    assert recorder.snapshot().events[3].duration_ms == pytest.approx(300.0)


def test_runner_uses_injected_clock_instead_of_wall_clock():
    clock = FakeClock([0.0, 1.0, 1.5, 2.0])
    runner, _ = _runner(ScriptedModel([_answer()]), clock=clock)
    recorder = TraceRecorder("run-001", "scenario")

    runner.run(AgentTask(task="test"), trace_recorder=recorder)

    assert clock.calls == 4
    assert recorder.snapshot().events[-1].duration_ms == pytest.approx(2000.0)


def test_untraced_run_does_not_call_trace_clock_and_preserves_result_and_environment():
    scenario = _scenario()
    environment = RolloutEnvironment(scenario)
    clock = FakeClock([])
    runner = AgentRunner(
        ScriptedModel([_call(AgentToolName.ROLLBACK_ROLLOUT), _answer(Decision.ROLLBACK)]),
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
        clock=clock,
    )

    result = runner.run(AgentTask(task="test"))

    assert clock.calls == 0
    assert result.final_answer.decision is Decision.ROLLBACK
    assert len(result.interactions) == 1
    assert environment.snapshot().rollout.phase == "RolledBack"
    assert environment.mutation_count == 1


def test_expected_mutation_limit_failure_is_timed_and_run_recovers():
    clock = FakeClock(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.5]
    )
    model = ScriptedModel(
        [
            _call(AgentToolName.ROLLBACK_ROLLOUT),
            _call(AgentToolName.PROMOTE_ROLLOUT),
            _answer(Decision.ROLLBACK),
        ]
    )
    runner, _ = _runner(model, clock=clock)
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(AgentTask(task="test"), trace_recorder=recorder)
    events = recorder.snapshot().events
    tool_failure = next(e for e in events if e.event_type is TraceEventType.TOOL_FAILED)

    assert tool_failure.duration_ms == pytest.approx(200.0)
    assert events[-1].event_type is TraceEventType.RUN_COMPLETED
    assert events[-1].duration_ms == pytest.approx(1500.0)
    assert result.interactions[1].error is not None
    assert result.interactions[1].result is None
    assert result.interactions[1].call.tool is AgentToolName.PROMOTE_ROLLOUT


def test_model_failure_has_model_and_run_durations_and_reraises_original():
    original = RuntimeError("provider failure")
    clock = FakeClock([0.0, 1.0, 1.2, 1.5])
    runner, _ = _runner(FailingModel(original), clock=clock)
    recorder = TraceRecorder("run-001", "scenario")

    with pytest.raises(RuntimeError) as raised:
        runner.run(AgentTask(task="test"), trace_recorder=recorder)

    events = recorder.snapshot().events
    assert raised.value is original
    assert events[2].event_type is TraceEventType.MODEL_FAILED
    assert events[2].duration_ms == pytest.approx(200.0)
    assert events[3].event_type is TraceEventType.RUN_FAILED
    assert events[3].duration_ms == pytest.approx(1500.0)


def test_unexpected_tool_failure_has_tool_and_run_durations_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
):
    runner, _ = _runner(
        ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE)]),
        clock=FakeClock([0.0, 0.1, 0.2, 0.3, 0.4, 0.6]),
    )
    original = RuntimeError("tool failure")

    def fail_dispatch(call: AgentToolCall) -> AgentToolInteraction:
        raise original

    monkeypatch.setattr(runner._dispatcher, "execute", fail_dispatch)
    recorder = TraceRecorder("run-001", "scenario")

    with pytest.raises(RuntimeError) as raised:
        runner.run(AgentTask(task="test"), trace_recorder=recorder)

    events = recorder.snapshot().events
    assert raised.value is original
    assert events[4].event_type is TraceEventType.TOOL_FAILED
    assert events[4].duration_ms == pytest.approx(100.0)
    assert events[5].event_type is TraceEventType.RUN_FAILED
    assert events[5].duration_ms == pytest.approx(600.0)


def test_step_limit_run_failed_contains_total_duration_and_keeps_event_order():
    clock = FakeClock([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    runner, _ = _runner(
        ScriptedModel(
            [_call(AgentToolName.GET_ROLLOUT_STATE), _call(AgentToolName.GET_SERVICE_METRICS)]
        ),
        max_steps=2,
        clock=clock,
    )
    recorder = TraceRecorder("run-001", "scenario")

    with pytest.raises(AgentStepLimitExceeded):
        runner.run(AgentTask(task="test"), trace_recorder=recorder)

    events = recorder.snapshot().events
    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
        TraceEventType.RUN_FAILED,
    ]
    assert events[-1].duration_ms == pytest.approx(1000.0)
    assert not any(event.event_type is TraceEventType.RUN_COMPLETED for event in events)


def test_backward_injected_clock_is_not_silently_repaired():
    clock = FakeClock([0.0, 1.0, 0.9])
    runner, _ = _runner(ScriptedModel([_answer()]), clock=clock)
    recorder = TraceRecorder("run-001", "scenario")

    with pytest.raises(ValidationError, match="duration_ms"):
        runner.run(AgentTask(task="test"), trace_recorder=recorder)


def test_all_six_canonical_scenarios_run_with_injected_clock():
    for scenario_id in SCENARIO_IDS:
        clock = IncrementingClock()
        runner, environment = _runner(
            ScriptedModel(_outputs_for_scenario(scenario_id)),
            scenario_id=scenario_id,
            clock=clock,
        )
        recorder = TraceRecorder(f"run-{scenario_id}", scenario_id)

        result = runner.run(
            AgentTask.from_scenario(_scenario(scenario_id)),
            trace_recorder=recorder,
        )
        trace = recorder.snapshot()

        assert result.final_answer.decision is _scenario(scenario_id).expected_decision
        assert trace.events[-1].event_type is TraceEventType.RUN_COMPLETED
        assert trace.events[-1].duration_ms is not None
        assert all(
            event.duration_ms is not None
            for event in trace.events
            if event.event_type in COMPLETION_OR_FAILURE
        )
        assert all(
            event.duration_ms is None
            for event in trace.events
            if event.event_type in START_EVENTS
        )
        assert environment.mutation_count == 1
        assert clock.calls > 0


def test_timing_fields_contain_no_evaluator_private_truth_or_usage_fields():
    runner, _ = _runner(ScriptedModel([_answer()]), clock=IncrementingClock())
    recorder = TraceRecorder("run-001", "degraded-canary")

    runner.run(AgentTask(task="test"), trace_recorder=recorder)

    for event in recorder.snapshot().events:
        fields = set(event.model_dump())
        assert not {
            "expected_decision",
            "expected_final_state",
            "required_evidence",
            "max_mutations",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        } & fields
