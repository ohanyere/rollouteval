from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentModelResponse,
    AgentRunner,
    AgentStepLimitExceeded,
    AgentTask,
    AgentToolCall,
    AgentToolName,
    ModelUsage,
)
from rollouteval.domain.scenario import Decision, Scenario
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import GraderStatus
from rollouteval.observability import (
    CostEstimationUnavailable,
    ModelTokenPricing,
    PricingSnapshot,
    TraceEventType,
    TraceRecorder,
)
from rollouteval.orchestration import (
    ObservableRun,
    ObservableRunOrchestrator,
    OrchestrationConfigurationError,
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


def _scenario(scenario_id: str = "degraded-canary") -> Scenario:
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _answer(decision: Decision, explanation: str = "Based on observed rollout data."):
    return AgentFinalAnswer(kind="final", decision=decision, explanation=explanation)


def _response(output, *, metadata: bool = False) -> AgentModelResponse:
    if not metadata:
        return AgentModelResponse(output=output)
    return AgentModelResponse(
        output=output,
        model="synthetic-model",
        usage=ModelUsage(
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
            cached_input_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
        ),
    )


class ScriptedModel:
    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self._responses = iter(responses)
        self.tasks: list[AgentTask] = []
        self.histories = []
        self.calls = 0

    def respond(self, task, interactions):
        self.calls += 1
        self.tasks.append(task)
        self.histories.append(interactions)
        return next(self._responses)


class FailingModel:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def respond(self, task, interactions):
        self.calls += 1
        raise self.error


class IncrementingClock:
    def __init__(self, increment: float = 0.002) -> None:
        self.value = 0.0
        self.increment = increment

    def __call__(self) -> float:
        self.value += self.increment
        return self.value


def _canonical_outputs(scenario: Scenario, *, metadata: bool = False):
    outputs = [_call(EVIDENCE_TO_TOOL[item]) for item in scenario.required_evidence]
    outputs.extend(
        [
            _call(DECISION_TO_TOOL[scenario.expected_decision]),
            _call(AgentToolName.GET_ROLLOUT_STATE),
            _answer(scenario.expected_decision),
        ]
    )
    return [_response(output, metadata=metadata) for output in outputs]


def _make_runner(scenario, model, *, environment=None, clock=None, max_steps=8):
    environment = environment or RolloutEnvironment(scenario)
    observations = RolloutObservationTools(environment)
    mutations = RolloutMutationTools(environment)
    runner = AgentRunner(
        model,
        observations,
        mutations,
        max_steps=max_steps,
        clock=clock or IncrementingClock(),
    )
    return runner, environment, observations, mutations


def _execute(scenario, model, *, run_id="run-001", pricing=None, clock=None, max_steps=8):
    runner, environment, _, _ = _make_runner(
        scenario, model, clock=clock, max_steps=max_steps
    )
    recorder = TraceRecorder(run_id=run_id, scenario_id=scenario.id)
    result = ObservableRunOrchestrator().execute(
        scenario=scenario,
        runner=runner,
        environment=environment,
        trace_recorder=recorder,
        pricing=pricing,
    )
    return result, environment, recorder, model


def _statuses(observable: ObservableRun):
    evaluation = observable.evaluation
    return (
        evaluation.decision.status,
        evaluation.final_state.status,
        evaluation.evidence_sufficiency.status,
        evaluation.trajectory.status,
        evaluation.safety.status,
        evaluation.abstention.status,
    )


class ScriptedGroundednessJudge:
    provider_name = "test-provider"
    model = "test-judge"
    prompt_sha256 = "a" * 64

    def __init__(
        self,
        status=GraderStatus.PASS,
        reason="Supported by observed evidence.",
        error=None,
    ):
        from rollouteval.evaluation import GroundednessJudgment

        self.judgment = GroundednessJudgment(status=status, reason=reason)
        self.error = error
        self.contexts = []

    def judge(self, context):
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return self.judgment


def _execute_with_judge(
    scenario_id,
    judge,
    *,
    decision=None,
    explanation="Observed rollout evidence supports this explanation.",
):
    scenario = _scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    outputs = [
        _call(EVIDENCE_TO_TOOL[item]) for item in scenario.required_evidence
    ]
    selected_decision = decision or scenario.expected_decision
    outputs.append(_call(DECISION_TO_TOOL[selected_decision]))
    outputs.append(_call(AgentToolName.GET_ROLLOUT_STATE))
    outputs.append(_answer(selected_decision, explanation))
    model = ScriptedModel([_response(output) for output in outputs])
    runner, _, _, _ = _make_runner(scenario, model, environment=environment)
    trace_recorder = TraceRecorder("groundedness-run", scenario.id)
    result = ObservableRunOrchestrator().execute(
        scenario=scenario,
        runner=runner,
        environment=environment,
        trace_recorder=trace_recorder,
        groundedness_judge=judge,
    )
    return result


def test_observable_run_has_exact_five_artifact_fields_and_is_frozen() -> None:
    assert set(ObservableRun.model_fields) == {
        "agent_run",
        "runtime_snapshot",
        "trace",
        "evaluation",
        "cost_estimate",
    }
    assert not {"overall_status", "score", "scenario"} & set(ObservableRun.model_fields)


def test_canonical_run_constructs_safe_task_and_completes_trace() -> None:
    scenario = _scenario()
    model = ScriptedModel(_canonical_outputs(scenario))

    observable, _, _, _ = _execute(scenario, model)

    assert all(isinstance(task, AgentTask) for task in model.tasks)
    assert all(task.task == scenario.task for task in model.tasks)
    assert "expected_decision" not in AgentTask.model_fields
    assert observable.trace.events[-1].event_type is TraceEventType.RUN_COMPLETED
    assert observable.trace.run_id == "run-001"
    assert observable.trace.scenario_id == scenario.id
    with pytest.raises(ValidationError):
        observable.trace = observable.trace


def test_normal_run_persists_groundedness_provenance_and_reads_it_back() -> None:
    judge = ScriptedGroundednessJudge()
    observable = _execute_with_judge("healthy-canary", judge)

    criterion = observable.evaluation.groundedness
    assert criterion is not None
    assert criterion.status is GraderStatus.PASS
    assert criterion.reason == "Supported by observed evidence."
    assert criterion.grader == "groundedness"
    assert criterion.evaluator_version == "1"
    assert criterion.criterion_id == "groundedness"
    assert criterion.evidence_references
    assert criterion.judge_provider == "test-provider"
    assert criterion.judge_model == "test-judge"
    assert criterion.prompt_sha256 == "a" * 64
    assert observable.evaluation.decision.status is GraderStatus.PASS
    restored = ObservableRun.model_validate_json(observable.model_dump_json())
    restored_criterion = restored.evaluation.groundedness
    assert restored_criterion == criterion
    assert restored.trace.run_id == observable.trace.run_id
    assert restored.trace.scenario_id == observable.trace.scenario_id


def test_unsupported_claim_failure_is_reported_in_normal_evaluation() -> None:
    judge = ScriptedGroundednessJudge(
        GraderStatus.FAIL,
        "The claimed outage is absent from the supplied observations.",
    )
    observable = _execute_with_judge(
        "healthy-canary",
        judge,
        explanation="The database outage caused the rollout failure.",
    )

    assert observable.evaluation.groundedness.status is GraderStatus.FAIL
    reportable = ObservableRun.model_validate_json(observable.model_dump_json())
    assert reportable.evaluation.groundedness.reason.startswith("The claimed outage")


def test_insufficient_evidence_keeps_inconclusive_judgment() -> None:
    judge = ScriptedGroundednessJudge(
        GraderStatus.INCONCLUSIVE,
        "The supplied evidence is insufficient to decide.",
    )
    observable = _execute_with_judge(
        "telemetry-unavailable",
        judge,
        explanation="Service metrics were unavailable in the observed response.",
    )

    assert observable.evaluation.groundedness.status is GraderStatus.INCONCLUSIVE
    assert len(judge.contexts) == 1
    assert any(item.status.value == "UNAVAILABLE" for item in judge.contexts[0].evidence)
    assert set(judge.contexts[0].model_fields_set) == {
        "task",
        "decision",
        "explanation",
        "evidence",
    }


def test_groundedness_pass_does_not_override_objective_failure() -> None:
    observable = _execute_with_judge(
        "degraded-canary",
        ScriptedGroundednessJudge(),
        decision=Decision.PROMOTE,
    )

    assert observable.evaluation.groundedness.status is GraderStatus.PASS
    assert observable.evaluation.decision.status is GraderStatus.FAIL
    assert observable.evaluation.groundedness not in observable.evaluation.results()[:6]


def test_provider_failure_is_inconclusive_and_preserves_objective_results() -> None:
    observable = _execute_with_judge(
        "healthy-canary",
        ScriptedGroundednessJudge(error=RuntimeError("provider unavailable")),
    )

    groundedness = observable.evaluation.groundedness
    assert groundedness.status is GraderStatus.INCONCLUSIVE
    assert groundedness.judge_error == "RuntimeError"
    assert groundedness.judge_provider == "test-provider"
    assert observable.evaluation.decision.status is GraderStatus.PASS
    restored = ObservableRun.model_validate_json(observable.model_dump_json())
    assert restored.evaluation == observable.evaluation


def test_trace_scenario_mismatch_fails_before_model_execution() -> None:
    scenario = _scenario()
    model = ScriptedModel([_response(_answer(scenario.expected_decision))])
    runner, environment, _, _ = _make_runner(scenario, model)
    recorder = TraceRecorder("caller-run-id", "other-scenario")

    with pytest.raises(OrchestrationConfigurationError, match="scenario_id"):
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
        )

    assert model.calls == 0
    assert recorder.snapshot().events == ()


def test_caller_supplied_run_identity_is_preserved_without_generation() -> None:
    scenario = _scenario()
    model = ScriptedModel(_canonical_outputs(scenario))
    observable, _, _, _ = _execute(scenario, model, run_id="externally-chosen-id")

    assert observable.trace.run_id == "externally-chosen-id"
    assert observable.trace.scenario_id == scenario.id


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_canonical_scenarios_produce_independent_expected_evaluations(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)
    model = ScriptedModel(_canonical_outputs(scenario))

    observable, _, _, _ = _execute(scenario, model)

    expected = (GraderStatus.PASS,) * 6
    if scenario_id == "telemetry-unavailable":
        expected = (
            GraderStatus.PASS,
            GraderStatus.PASS,
            GraderStatus.INCONCLUSIVE,
            GraderStatus.PASS,
            GraderStatus.PASS,
            GraderStatus.PASS,
        )
    assert _statuses(observable) == expected
    assert observable.runtime_snapshot.rollout.phase == scenario.expected_final_state.phase
    assert observable.runtime_snapshot.rollout.candidate_weight == scenario.expected_final_state.candidate_weight
    assert observable.agent_run.steps == model.calls
    assert sum(item.call.tool in set(DECISION_TO_TOOL.values()) for item in observable.agent_run.interactions) == 1


def test_wrong_promote_on_degraded_canary_preserves_mixed_results() -> None:
    scenario = _scenario("degraded-canary")
    model = ScriptedModel(
        [
            _response(_call(AgentToolName.GET_ROLLOUT_STATE)),
            _response(_call(AgentToolName.GET_SERVICE_METRICS)),
            _response(_call(AgentToolName.PROMOTE_ROLLOUT)),
            _response(_call(AgentToolName.GET_ROLLOUT_STATE)),
            _response(_answer(Decision.PROMOTE)),
        ]
    )

    observable, environment, _, _ = _execute(scenario, model)

    assert _statuses(observable) == (
        GraderStatus.FAIL,
        GraderStatus.FAIL,
        GraderStatus.PASS,
        GraderStatus.PASS,
        GraderStatus.PASS,
        GraderStatus.PASS,
    )
    assert observable.runtime_snapshot == environment.snapshot()
    assert observable.runtime_snapshot.rollout.phase == "Promoted"
    assert observable.runtime_snapshot.rollout.candidate_weight == 100
    assert observable.agent_run.final_answer.decision is Decision.PROMOTE


def test_evidence_after_rollback_preserves_bad_process_result() -> None:
    scenario = _scenario("degraded-canary")
    model = ScriptedModel(
        [
            _response(_call(AgentToolName.ROLLBACK_ROLLOUT)),
            _response(_call(AgentToolName.GET_ROLLOUT_STATE)),
            _response(_call(AgentToolName.GET_SERVICE_METRICS)),
            _response(_call(AgentToolName.GET_ROLLOUT_STATE)),
            _response(_answer(Decision.ROLLBACK)),
        ]
    )

    observable, _, _, _ = _execute(scenario, model)

    assert _statuses(observable) == (
        GraderStatus.PASS,
        GraderStatus.PASS,
        GraderStatus.PASS,
        GraderStatus.FAIL,
        GraderStatus.PASS,
        GraderStatus.PASS,
    )


def test_declared_rollback_without_mutation_preserves_no_action_result() -> None:
    scenario = _scenario("degraded-canary")
    model = ScriptedModel(
        [
            _response(_call(AgentToolName.GET_ROLLOUT_STATE)),
            _response(_call(AgentToolName.GET_SERVICE_METRICS)),
            _response(_answer(Decision.ROLLBACK)),
        ]
    )

    observable, environment, _, _ = _execute(scenario, model)

    assert _statuses(observable) == (
        GraderStatus.PASS,
        GraderStatus.FAIL,
        GraderStatus.PASS,
        GraderStatus.FAIL,
        GraderStatus.PASS,
        GraderStatus.PASS,
    )
    assert observable.runtime_snapshot == scenario.initial_environment
    assert environment.mutation_count == 0


def test_timing_identity_and_usage_are_preserved_in_trace() -> None:
    scenario = _scenario("degraded-canary")
    model = ScriptedModel(_canonical_outputs(scenario, metadata=True))
    pricing = PricingSnapshot(
        pricing_id="synthetic-test-rates",
        models=(
            ModelTokenPricing(
                model="synthetic-model",
                input_per_million_usd=Decimal("1"),
                cached_input_per_million_usd=Decimal("0"),
                cache_write_per_million_usd=Decimal("0"),
                output_per_million_usd=Decimal("3"),
            ),
        ),
    )

    observable, _, _, _ = _execute(scenario, model, pricing=pricing)
    completed = [
        event for event in observable.trace.events
        if event.event_type is TraceEventType.MODEL_COMPLETED
    ]
    tool_events = [
        event for event in observable.trace.events
        if event.event_type is TraceEventType.TOOL_COMPLETED
    ]

    assert completed
    assert all(event.model == "synthetic-model" and event.model_usage is not None for event in completed)
    assert all(event.duration_ms is not None for event in completed + tool_events)
    assert observable.trace.events[-1].duration_ms is not None
    assert observable.cost_estimate is not None
    assert observable.cost_estimate.pricing_id == "synthetic-test-rates"
    assert observable.cost_estimate.total_cost_usd == Decimal("0.000025")
    assert len(observable.agent_run.interactions) == len(tool_events)


def test_no_pricing_keeps_cost_absent_and_evaluation_unchanged() -> None:
    scenario = _scenario("healthy-canary")
    unpriced_model = ScriptedModel(_canonical_outputs(scenario, metadata=True))
    priced_metadata_model = ScriptedModel(_canonical_outputs(scenario, metadata=True))
    unpriced, _, _, _ = _execute(scenario, unpriced_model)
    priced_model, _, _, _ = _execute(
        scenario,
        priced_metadata_model,
        pricing=PricingSnapshot(
            pricing_id="test-rates",
            models=(
                ModelTokenPricing(
                    model="synthetic-model",
                    input_per_million_usd=Decimal("1"),
                    cached_input_per_million_usd=Decimal("1"),
                    cache_write_per_million_usd=Decimal("1"),
                    output_per_million_usd=Decimal("1"),
                ),
            ),
        ),
    )

    assert unpriced.cost_estimate is None
    assert unpriced.evaluation == priced_model.evaluation
    assert unpriced.agent_run == priced_model.agent_run
    assert unpriced.runtime_snapshot == priced_model.runtime_snapshot
    assert unpriced.trace == priced_model.trace


@pytest.mark.parametrize(
    ("response", "pricing", "message"),
    [
        (
            AgentModelResponse(
                output=_answer(Decision.ROLLBACK),
                model="synthetic-model",
                usage=None,
            ),
            "matching-pricing",
            "model usage",
        ),
        (
            AgentModelResponse(
                output=_answer(Decision.ROLLBACK),
                model="synthetic-model",
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            ),
            "matching-pricing",
            "unknown cached_input_tokens",
        ),
        (
            AgentModelResponse(
                output=_answer(Decision.ROLLBACK),
                model="unpriced-model",
                usage=ModelUsage(
                    input_tokens=2,
                    output_tokens=1,
                    total_tokens=3,
                    cached_input_tokens=0,
                    cache_write_tokens=0,
                ),
            ),
            "matching-pricing",
            "no pricing for exact model",
        ),
    ],
)
def test_requested_cost_failures_propagate_after_run(response, pricing, message: str) -> None:
    scenario = _scenario("degraded-canary")
    model = ScriptedModel([response])
    runner, environment, _, _ = _make_runner(scenario, model)
    recorder = TraceRecorder("cost-error", scenario.id)
    rate_snapshot = None
    if pricing == "matching-pricing":
        rate_snapshot = PricingSnapshot(
            pricing_id="synthetic",
            models=(
                ModelTokenPricing(
                    model="synthetic-model",
                    input_per_million_usd=Decimal("1"),
                    cached_input_per_million_usd=Decimal("1"),
                    cache_write_per_million_usd=Decimal("1"),
                    output_per_million_usd=Decimal("1"),
                ),
            ),
        )

    with pytest.raises(CostEstimationUnavailable, match=message):
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
            pricing=rate_snapshot,
        )

    assert recorder.snapshot().events[-1].event_type is TraceEventType.RUN_COMPLETED


def test_unknown_model_cost_failure_happens_after_evaluation(monkeypatch) -> None:
    import rollouteval.orchestration as orchestration_module

    scenario = _scenario("degraded-canary")
    model = ScriptedModel(
        [
            AgentModelResponse(
                output=_answer(Decision.ROLLBACK),
                model="unpriced",
                usage=ModelUsage(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    cached_input_tokens=0,
                    cache_write_tokens=0,
                ),
            )
        ]
    )
    runner, environment, _, _ = _make_runner(scenario, model)
    recorder = TraceRecorder("cost-order", scenario.id)
    evaluated = []
    original_evaluate = orchestration_module.DeterministicEvaluationEngine.evaluate

    def track_evaluate(self, *args, **kwargs):
        evaluated.append(True)
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(orchestration_module.DeterministicEvaluationEngine, "evaluate", track_evaluate)
    pricing = PricingSnapshot(
        pricing_id="no-unpriced-model",
        models=(
            ModelTokenPricing(
                model="some-other-model",
                input_per_million_usd=Decimal("1"),
                cached_input_per_million_usd=Decimal("1"),
                cache_write_per_million_usd=Decimal("1"),
                output_per_million_usd=Decimal("1"),
            ),
        ),
    )

    with pytest.raises(CostEstimationUnavailable):
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
            pricing=pricing,
        )
    assert evaluated == [True]
    assert recorder.snapshot().events[-1].event_type is TraceEventType.RUN_COMPLETED


def test_returned_runtime_and_trace_snapshots_survive_later_changes() -> None:
    scenario = _scenario("degraded-canary")
    model = ScriptedModel(_canonical_outputs(scenario))
    observable, environment, recorder, _ = _execute(scenario, model)
    original_snapshot = observable.runtime_snapshot
    original_trace = observable.trace

    environment.reset()
    recorder.record(TraceEventType.RUN_FAILED, message="later caller event")

    assert observable.runtime_snapshot == original_snapshot
    assert observable.runtime_snapshot.rollout.phase == "RolledBack"
    assert observable.trace == original_trace
    assert recorder.snapshot().events[-1].event_type is TraceEventType.RUN_FAILED


def test_orchestration_does_not_reset_environment_or_change_scenario() -> None:
    scenario = _scenario("degraded-canary")
    original_scenario = scenario.model_copy(deep=True)
    model = ScriptedModel(_canonical_outputs(scenario))
    observable, environment, _, _ = _execute(scenario, model)

    assert environment.mutation_count == 1
    assert observable.runtime_snapshot.rollout.phase == "RolledBack"
    assert scenario == original_scenario


def test_model_failure_propagates_and_leaves_failure_trace_inspectable() -> None:
    scenario = _scenario()
    failure = RuntimeError("model failed")
    model = FailingModel(failure)
    runner, environment, _, _ = _make_runner(scenario, model)
    recorder = TraceRecorder("model-failure", scenario.id)

    with pytest.raises(RuntimeError) as caught:
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
        )

    assert caught.value is failure
    assert [event.event_type for event in recorder.snapshot().events][-2:] == [
        TraceEventType.MODEL_FAILED,
        TraceEventType.RUN_FAILED,
    ]


def test_step_limit_propagates_and_leaves_run_failed_trace() -> None:
    scenario = _scenario()
    model = ScriptedModel([_response(_call(AgentToolName.GET_ROLLOUT_STATE))])
    runner, environment, _, _ = _make_runner(scenario, model, max_steps=1)
    recorder = TraceRecorder("step-limit", scenario.id)

    with pytest.raises(AgentStepLimitExceeded):
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
        )

    assert recorder.snapshot().events[-1].event_type is TraceEventType.RUN_FAILED
    assert model.calls == 1


def test_unexpected_tool_failure_propagates_and_leaves_failure_trace(monkeypatch) -> None:
    scenario = _scenario()
    model = ScriptedModel([_response(_call(AgentToolName.GET_ROLLOUT_STATE))])
    environment = RolloutEnvironment(scenario)
    observations = RolloutObservationTools(environment)
    mutations = RolloutMutationTools(environment)
    failure = RuntimeError("observation implementation failed")

    def fail_observation():
        raise failure

    monkeypatch.setattr(observations, "get_rollout_state", fail_observation)
    runner = AgentRunner(model, observations, mutations)
    recorder = TraceRecorder("tool-failure", scenario.id)

    with pytest.raises(RuntimeError) as caught:
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
        )

    assert caught.value is failure
    assert [event.event_type for event in recorder.snapshot().events][-2:] == [
        TraceEventType.TOOL_FAILED,
        TraceEventType.RUN_FAILED,
    ]


def test_evaluation_software_error_propagates_without_partial_run(monkeypatch) -> None:
    import rollouteval.orchestration as orchestration_module

    scenario = _scenario()
    model = ScriptedModel(_canonical_outputs(scenario))
    runner, environment, _, _ = _make_runner(scenario, model)
    recorder = TraceRecorder("evaluation-failure", scenario.id)
    failure = RuntimeError("evaluation integrity failure")

    def fail_evaluation(self, *args, **kwargs):
        raise failure

    monkeypatch.setattr(orchestration_module.DeterministicEvaluationEngine, "evaluate", fail_evaluation)

    with pytest.raises(RuntimeError) as caught:
        ObservableRunOrchestrator().execute(
            scenario=scenario,
            runner=runner,
            environment=environment,
            trace_recorder=recorder,
        )

    assert caught.value is failure
    assert recorder.snapshot().events[-1].event_type is TraceEventType.RUN_COMPLETED
