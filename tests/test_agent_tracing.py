import json
from pathlib import Path

import pytest

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentModel,
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
from rollouteval.observability import TraceEventType, TraceRecorder
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


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _task(scenario_id: str = "degraded-canary") -> AgentTask:
    return AgentTask.from_scenario(_scenario(scenario_id))


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _answer(decision: Decision = Decision.PAUSE) -> AgentFinalAnswer:
    return AgentFinalAnswer(
        kind="final", decision=decision, explanation="Deterministic test response."
    )


class ScriptedModel:
    def __init__(self, outputs: list[AgentOutput]) -> None:
        self._outputs = iter(outputs)
        self.interaction_histories: list[tuple[AgentToolInteraction, ...]] = []

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentOutput:
        assert isinstance(task, AgentTask)
        self.interaction_histories.append(interactions)
        return next(self._outputs)


class FailingModel:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentOutput:
        self.calls += 1
        raise self.error


def _runner(
    model: AgentModel,
    scenario_id: str = "degraded-canary",
    max_steps: int = 8,
) -> tuple[AgentRunner, RolloutEnvironment]:
    environment = RolloutEnvironment(_scenario(scenario_id))
    runner = AgentRunner(
        model,
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
        max_steps=max_steps,
    )
    return runner, environment


def _observe_tool_result(scenario_id: str, tool: AgentToolName) -> str:
    environment = RolloutEnvironment(_scenario(scenario_id))
    if tool is AgentToolName.GET_ROLLOUT_STATE:
        return environment.snapshot().rollout.model_dump_json()
    if tool is AgentToolName.GET_SERVICE_METRICS:
        snapshot = environment.snapshot()
        if snapshot.metrics is None:
            result = {
                "available": False,
                "metrics": None,
                "message": "Service metrics are unavailable.",
            }
        else:
            result = {
                "available": True,
                "metrics": snapshot.metrics.model_dump(mode="json"),
                "message": None,
            }
        return json.dumps(result)
    return json.dumps(
        [event.model_dump(mode="json") for event in environment.snapshot().events]
    )


def _canonical_outputs(scenario_id: str) -> list[AgentOutput]:
    scenario = _scenario(scenario_id)
    outputs: list[AgentOutput] = [
        _call(EVIDENCE_TO_TOOL[evidence_id])
        for evidence_id in scenario.required_evidence
    ]
    outputs.append(_call(DECISION_TO_MUTATION[scenario.expected_decision]))
    outputs.append(_call(AgentToolName.GET_ROLLOUT_STATE))
    outputs.append(_answer(scenario.expected_decision))
    return outputs


def test_runner_still_runs_without_trace_recorder() -> None:
    model = ScriptedModel([_answer(Decision.ROLLBACK)])
    runner, _ = _runner(model)

    result = runner.run(_task())

    assert result.final_answer.decision is Decision.ROLLBACK
    assert result.interactions == ()
    assert result.steps == 1


def test_untraced_run_preserves_semantic_result_and_environment_mutation() -> None:
    scenario = _scenario()
    plain_environment = RolloutEnvironment(scenario)
    traced_environment = RolloutEnvironment(scenario)
    plain_runner = AgentRunner(
        ScriptedModel([_call(AgentToolName.ROLLBACK_ROLLOUT), _answer(Decision.ROLLBACK)]),
        RolloutObservationTools(plain_environment),
        RolloutMutationTools(plain_environment),
    )
    traced_runner = AgentRunner(
        ScriptedModel([_call(AgentToolName.ROLLBACK_ROLLOUT), _answer(Decision.ROLLBACK)]),
        RolloutObservationTools(traced_environment),
        RolloutMutationTools(traced_environment),
    )
    recorder = TraceRecorder("traced-run", scenario.id)

    plain_result = plain_runner.run(_task())
    traced_result = traced_runner.run(_task(), trace_recorder=recorder)

    assert traced_result == plain_result
    assert traced_environment.snapshot() == plain_environment.snapshot()
    assert traced_environment.mutation_count == plain_environment.mutation_count == 1


def test_immediate_final_answer_emits_exact_lifecycle_and_step_one() -> None:
    runner, _ = _runner(ScriptedModel([_answer()]))
    recorder = TraceRecorder("run-001", "degraded-canary")

    runner.run(_task(), trace_recorder=recorder)
    events = recorder.snapshot().events

    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]
    assert events[1].step == events[2].step == 1
    assert not any(event.event_type.value.startswith("tool.") for event in events)


def test_observation_tool_then_final_has_exact_lifecycle_and_matching_step():
    tool = AgentToolName.GET_ROLLOUT_STATE
    runner, _ = _runner(ScriptedModel([_call(tool), _answer()]))
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(_task(), trace_recorder=recorder)
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
    assert events[3].tool_name is events[4].tool_name is tool
    assert events[3].step == events[4].step == 1
    assert events[5].step == events[6].step == 2
    assert result.steps == 2


def test_multiple_tools_keep_operational_order_contiguous_sequences_and_steps():
    outputs = [
        _call(AgentToolName.GET_ROLLOUT_STATE),
        _call(AgentToolName.GET_SERVICE_METRICS),
        _answer(),
    ]
    runner, _ = _runner(ScriptedModel(outputs))
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(_task(), trace_recorder=recorder)
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
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]
    assert [event.sequence for event in events] == list(range(len(events)))
    assert [event.step for event in events if event.event_type is TraceEventType.MODEL_STARTED] == [1, 2, 3]
    assert [interaction.call.tool for interaction in result.interactions] == [
        AgentToolName.GET_ROLLOUT_STATE,
        AgentToolName.GET_SERVICE_METRICS,
    ]


@pytest.mark.parametrize(
    ("mutation", "decision"),
    [
        (AgentToolName.PROMOTE_ROLLOUT, Decision.PROMOTE),
        (AgentToolName.PAUSE_ROLLOUT, Decision.PAUSE),
        (AgentToolName.ROLLBACK_ROLLOUT, Decision.ROLLBACK),
    ],
)
def test_successful_mutation_emits_tool_started_and_completed(
    mutation: AgentToolName, decision: Decision
) -> None:
    runner, _ = _runner(ScriptedModel([_call(mutation), _answer(decision)]))
    recorder = TraceRecorder("run-001", "scenario")

    runner.run(_task(), trace_recorder=recorder)
    events = recorder.snapshot().events
    tool_events = [event for event in events if event.event_type.value.startswith("tool.")]

    assert [event.event_type for event in tool_events] == [
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
    ]
    assert all(event.tool_name is mutation and event.step == 1 for event in tool_events)


def test_expected_mutation_limit_failure_can_recover_and_finish():
    model = ScriptedModel(
        [
            _call(AgentToolName.ROLLBACK_ROLLOUT),
            _call(AgentToolName.PROMOTE_ROLLOUT),
            _answer(Decision.ROLLBACK),
        ]
    )
    runner, _ = _runner(model)
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(_task(), trace_recorder=recorder)
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
        TraceEventType.TOOL_FAILED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]
    assert not any(event.event_type is TraceEventType.RUN_FAILED for event in events)
    assert [item.call.tool for item in result.interactions] == [
        AgentToolName.ROLLBACK_ROLLOUT,
        AgentToolName.PROMOTE_ROLLOUT,
    ]
    assert result.interactions[0].error is None
    assert result.interactions[1].error is not None
    assert model.interaction_histories[-1] == result.interactions
    assert events[7].tool_name is events[8].tool_name is AgentToolName.PROMOTE_ROLLOUT
    assert events[7].step == events[8].step == 2


def test_model_failure_records_failed_lifecycle_and_reraises_original_exception():
    original = RuntimeError("private provider payload should not be traced")
    model = FailingModel(original)
    runner, _ = _runner(model)
    recorder = TraceRecorder("run-001", "degraded-canary")

    with pytest.raises(RuntimeError) as raised:
        runner.run(_task(), trace_recorder=recorder)

    events = recorder.snapshot().events
    assert raised.value is original
    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_FAILED,
        TraceEventType.RUN_FAILED,
    ]
    assert events[2].message == "Model failed: RuntimeError"
    assert not any(
        event.event_type in {TraceEventType.MODEL_COMPLETED, TraceEventType.RUN_COMPLETED}
        for event in events
    )
    assert model.calls == 1


def test_unexpected_tool_failure_records_failure_and_reraises_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = _runner(ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE)]))
    original = RuntimeError("private tool detail")

    def fail_dispatch(call: AgentToolCall) -> AgentToolInteraction:
        raise original

    monkeypatch.setattr(runner._dispatcher, "execute", fail_dispatch)
    recorder = TraceRecorder("run-001", "degraded-canary")

    with pytest.raises(RuntimeError) as raised:
        runner.run(_task(), trace_recorder=recorder)

    events = recorder.snapshot().events
    assert raised.value is original
    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_FAILED,
        TraceEventType.RUN_FAILED,
    ]
    assert events[4].message == "Tool failed: RuntimeError"
    assert not any(event.event_type is TraceEventType.RUN_COMPLETED for event in events)


def test_step_limit_records_run_failed_without_extra_model_call():
    model = ScriptedModel(
        [_call(AgentToolName.GET_ROLLOUT_STATE), _call(AgentToolName.GET_SERVICE_METRICS)]
    )
    runner, _ = _runner(model, max_steps=2)
    recorder = TraceRecorder("run-001", "degraded-canary")

    with pytest.raises(AgentStepLimitExceeded):
        runner.run(_task(), trace_recorder=recorder)

    events = recorder.snapshot().events
    assert events[-1].event_type is TraceEventType.RUN_FAILED
    assert not any(event.event_type is TraceEventType.RUN_COMPLETED for event in events)
    assert sum(event.event_type is TraceEventType.MODEL_STARTED for event in events) == 2
    assert len(model.interaction_histories) == 2


def test_final_answer_on_last_allowed_step_completes_run():
    runner, _ = _runner(
        ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()]),
        max_steps=2,
    )
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(_task(), trace_recorder=recorder)
    events = recorder.snapshot().events

    assert result.steps == 2
    assert events[-1].event_type is TraceEventType.RUN_COMPLETED
    assert not any(event.event_type is TraceEventType.RUN_FAILED for event in events)


def test_trace_snapshot_uses_caller_identity_and_excludes_evaluator_truth():
    runner, _ = _runner(ScriptedModel([_answer()]))
    recorder = TraceRecorder("caller-run", "caller-scenario")

    runner.run(_task(), trace_recorder=recorder)
    trace = recorder.snapshot()

    assert trace.run_id == "caller-run"
    assert trace.scenario_id == "caller-scenario"
    for event in trace.events:
        assert not {
            "expected_decision",
            "expected_final_state",
            "required_evidence",
            "max_mutations",
        } & set(event.model_dump())


def test_trace_and_semantic_interaction_histories_remain_separate():
    runner, _ = _runner(
        ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()])
    )
    recorder = TraceRecorder("run-001", "degraded-canary")

    result = runner.run(_task(), trace_recorder=recorder)
    trace = recorder.snapshot()

    assert len(result.interactions) == 1
    assert type(result.interactions[0]) is AgentToolInteraction
    assert len([event for event in trace.events if event.event_type is TraceEventType.TOOL_STARTED]) == 1
    assert len([event for event in trace.events if event.event_type is TraceEventType.TOOL_COMPLETED]) == 1
    assert not hasattr(result, "trace")


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_canonical_runner_execution_is_traceable_for_all_scenarios(
    scenario_id: str,
) -> None:
    runner, environment = _runner(
        ScriptedModel(_canonical_outputs(scenario_id)), scenario_id
    )
    recorder = TraceRecorder("run-001", scenario_id)

    result = runner.run(_task(scenario_id), trace_recorder=recorder)
    trace = recorder.snapshot()

    assert result.final_answer.decision is _scenario(scenario_id).expected_decision
    assert result.steps == len(result.interactions) + 1
    assert trace.events[0].event_type is TraceEventType.RUN_STARTED
    assert trace.events[-1].event_type is TraceEventType.RUN_COMPLETED
    assert [event.sequence for event in trace.events] == list(range(len(trace.events)))
    assert trace.scenario_id == scenario_id
    assert environment.mutation_count == 1
