import json
from pathlib import Path

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
from rollouteval.scenarios import load_scenario
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_IDS = (
    "degraded-canary",
    "healthy-canary",
    "insufficient-evidence",
    "transient-anomaly",
    "conflicting-signals",
    "telemetry-unavailable",
)


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIOS / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _answer(
    decision: Decision = Decision.PAUSE, explanation: str = "Continue cautiously."
) -> AgentFinalAnswer:
    return AgentFinalAnswer(kind="final", decision=decision, explanation=explanation)


class ScriptedModel:
    """Test-only deterministic implementation of the AgentModel protocol."""

    def __init__(self, outputs: list[AgentOutput]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[tuple[AgentTask, tuple[AgentToolInteraction, ...]]] = []

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentModelResponse:
        assert isinstance(task, AgentTask)
        self.calls.append((task, interactions))
        return AgentModelResponse(output=next(self._outputs))


def _runner(
    model: AgentModel,
    scenario_id: str = "degraded-canary",
    max_steps: int = 8,
) -> tuple[AgentRunner, RolloutEnvironment, AgentTask]:
    scenario = _scenario(scenario_id)
    environment = RolloutEnvironment(scenario)
    runner = AgentRunner(
        model,
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
        max_steps=max_steps,
    )
    return runner, environment, AgentTask.from_scenario(scenario)


def test_runner_finishes_immediately_with_final_answer() -> None:
    model = ScriptedModel([_answer(Decision.ROLLBACK, "The rollout is degraded.")])
    runner, _, task = _runner(model)

    result = runner.run(task)

    assert result.final_answer.decision is Decision.ROLLBACK
    assert result.final_answer.explanation == "The rollout is degraded."
    assert result.interactions == ()
    assert result.steps == 1


@pytest.mark.parametrize(
    "tool, expected_key",
    [
        (AgentToolName.GET_ROLLOUT_STATE, "candidate_weight"),
        (AgentToolName.GET_SERVICE_METRICS, "available"),
        (AgentToolName.GET_RECENT_EVENTS, None),
    ],
)
def test_runner_returns_observation_tool_result_in_history(
    tool: AgentToolName, expected_key: str | None
) -> None:
    model = ScriptedModel([_call(tool), _answer()])
    runner, _, task = _runner(model)

    result = runner.run(task)

    interaction = result.interactions[0]
    payload = json.loads(interaction.result)
    assert interaction.call.tool is tool
    assert interaction.error is None
    if expected_key is not None:
        assert expected_key in payload
    else:
        assert isinstance(payload, list)
    assert result.steps == 2


def test_telemetry_unavailable_result_is_explicit_in_model_history() -> None:
    model = ScriptedModel([_call(AgentToolName.GET_SERVICE_METRICS), _answer()])
    runner, _, task = _runner(model, "telemetry-unavailable")

    result = runner.run(task)
    metrics = json.loads(result.interactions[0].result)

    assert metrics["available"] is False
    assert metrics["metrics"] is None
    assert "unavailable" in metrics["message"].lower()


@pytest.mark.parametrize(
    "tool, expected_phase, expected_weight",
    [
        (AgentToolName.PROMOTE_ROLLOUT, "Promoted", 100),
        (AgentToolName.PAUSE_ROLLOUT, "Paused", 20),
        (AgentToolName.ROLLBACK_ROLLOUT, "RolledBack", 0),
    ],
)
def test_runner_executes_requested_mutation(
    tool: AgentToolName, expected_phase: str, expected_weight: int
) -> None:
    model = ScriptedModel([_call(tool), _answer()])
    runner, environment, task = _runner(model)

    runner.run(task)

    state = environment.snapshot().rollout
    assert (state.phase, state.candidate_weight) == (expected_phase, expected_weight)
    assert environment.mutation_count == 1


def test_runner_supports_mutation_then_reading_changed_state() -> None:
    model = ScriptedModel(
        [_call(AgentToolName.PROMOTE_ROLLOUT), _call(AgentToolName.GET_ROLLOUT_STATE), _answer()]
    )
    runner, _, task = _runner(model)

    result = runner.run(task)

    assert json.loads(result.interactions[1].result)["phase"] == "Promoted"
    assert json.loads(result.interactions[1].result)["candidate_weight"] == 100


def test_multistep_sequence_preserves_order_and_separates_final_answer() -> None:
    model = ScriptedModel(
        [
            _call(AgentToolName.GET_ROLLOUT_STATE),
            _call(AgentToolName.GET_SERVICE_METRICS),
            _call(AgentToolName.ROLLBACK_ROLLOUT),
            _answer(Decision.ROLLBACK, "The candidate is unhealthy."),
        ]
    )
    runner, _, task = _runner(model)

    result = runner.run(task)

    assert [item.call.tool for item in result.interactions] == [
        AgentToolName.GET_ROLLOUT_STATE,
        AgentToolName.GET_SERVICE_METRICS,
        AgentToolName.ROLLBACK_ROLLOUT,
    ]
    assert all(isinstance(item, AgentToolInteraction) for item in result.interactions)
    assert result.final_answer.explanation == "The candidate is unhealthy."
    assert result.steps == 4
    assert len(model.calls) == 4
    assert model.calls[3][1] == result.interactions


def test_runner_allows_promoting_degraded_canary() -> None:
    model = ScriptedModel([_call(AgentToolName.PROMOTE_ROLLOUT), _answer()])
    runner, environment, task = _runner(model, "degraded-canary")

    runner.run(task)

    assert environment.snapshot().rollout.phase == "Promoted"


def test_runner_allows_rolling_back_healthy_canary() -> None:
    model = ScriptedModel([_call(AgentToolName.ROLLBACK_ROLLOUT), _answer()])
    runner, environment, task = _runner(model, "healthy-canary")

    runner.run(task)

    assert environment.snapshot().rollout.phase == "RolledBack"


def test_runner_allows_final_answer_without_observations() -> None:
    model = ScriptedModel([_answer(Decision.PROMOTE)])
    runner, environment, task = _runner(model)

    result = runner.run(task)

    assert result.final_answer.decision is Decision.PROMOTE
    assert result.interactions == ()
    assert environment.mutation_count == 0


def test_runner_does_not_reconcile_final_decision_with_mutation() -> None:
    model = ScriptedModel([_call(AgentToolName.PROMOTE_ROLLOUT), _answer(Decision.ROLLBACK)])
    runner, environment, task = _runner(model)

    result = runner.run(task)

    assert result.final_answer.decision is Decision.ROLLBACK
    assert environment.snapshot().rollout.phase == "Promoted"


def test_second_mutation_failure_is_returned_to_model_without_state_change() -> None:
    model = ScriptedModel(
        [
            _call(AgentToolName.PROMOTE_ROLLOUT),
            _call(AgentToolName.ROLLBACK_ROLLOUT),
            _answer(),
        ]
    )
    runner, environment, task = _runner(model)

    result = runner.run(task)

    assert result.interactions[0].error is None
    assert result.interactions[0].result is not None
    assert result.interactions[1].result is None
    assert "mutation limit exceeded" in result.interactions[1].error
    assert model.calls[2][1] == result.interactions
    assert environment.snapshot().rollout.phase == "Promoted"
    assert environment.mutation_count == 1
    assert result.final_answer is not None


def test_runner_step_limit_raises_without_fabricating_final_answer() -> None:
    model = ScriptedModel(
        [_call(AgentToolName.GET_ROLLOUT_STATE)] * 3
    )
    runner, _, task = _runner(model, max_steps=3)

    with pytest.raises(AgentStepLimitExceeded, match="3 model steps"):
        runner.run(task)

    assert len(model.calls) == 3


def test_final_answer_at_step_limit_succeeds() -> None:
    model = ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()])
    runner, _, task = _runner(model, max_steps=2)

    result = runner.run(task)

    assert result.steps == 2
    assert len(result.interactions) == 1


def test_tool_call_at_limit_executes_then_run_raises_without_extra_model_step() -> None:
    model = ScriptedModel(
        [_call(AgentToolName.GET_ROLLOUT_STATE), _call(AgentToolName.GET_SERVICE_METRICS)]
    )
    runner, _, task = _runner(model, max_steps=1)

    with pytest.raises(AgentStepLimitExceeded):
        runner.run(task)

    assert len(model.calls) == 1


def test_model_receives_task_contract_not_scenario() -> None:
    model = ScriptedModel([_answer()])
    runner, _, task = _runner(model)

    runner.run(task)

    received_task, _ = model.calls[0]
    assert isinstance(received_task, AgentTask)
    assert not hasattr(received_task, "expected_decision")


def test_interaction_history_contains_no_evaluator_private_truth() -> None:
    model = ScriptedModel(
        [_call(AgentToolName.GET_ROLLOUT_STATE), _call(AgentToolName.GET_SERVICE_METRICS), _answer()]
    )
    runner, _, task = _runner(model)

    result = runner.run(task)

    private_names = {
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
    }
    for interaction in result.interactions:
        assert private_names.isdisjoint(interaction.model_dump())
        payload = json.loads(interaction.result)
        assert private_names.isdisjoint(payload)


def test_decoded_tool_result_cannot_mutate_environment_state() -> None:
    model = ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()])
    runner, environment, task = _runner(model)

    result = runner.run(task)
    decoded = json.loads(result.interactions[0].result)
    decoded["candidate_weight"] = 99

    assert environment.snapshot().rollout.candidate_weight == 20


def test_separate_runs_do_not_share_interaction_history() -> None:
    model = ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer(), _answer()])
    runner, _, task = _runner(model)

    first = runner.run(task)
    second = runner.run(task)

    assert len(first.interactions) == 1
    assert second.interactions == ()
    assert model.calls[2][1] == ()


def test_separate_environments_have_independent_run_histories() -> None:
    first_model = ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()])
    second_model = ScriptedModel([_answer()])
    first_runner, first_environment, first_task = _runner(first_model)
    second_runner, second_environment, second_task = _runner(second_model)

    first_result = first_runner.run(first_task)
    second_result = second_runner.run(second_task)

    assert len(first_result.interactions) == 1
    assert second_result.interactions == ()
    assert first_environment is not second_environment
    assert second_model.calls[0][1] == ()


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_runner_operates_on_all_scenarios_without_private_truth(
    scenario_id: str,
) -> None:
    model = ScriptedModel([_call(AgentToolName.GET_ROLLOUT_STATE), _answer()])
    runner, _, task = _runner(model, scenario_id)

    result = runner.run(task)

    assert result.steps == 2
    assert result.interactions[0].result is not None
    assert model.calls[0][0].task == task.task


def test_tool_interaction_requires_exactly_one_outcome() -> None:
    with pytest.raises(ValidationError):
        AgentToolInteraction(call=_call(AgentToolName.GET_ROLLOUT_STATE))
    with pytest.raises(ValidationError):
        AgentToolInteraction(
            call=_call(AgentToolName.GET_ROLLOUT_STATE), result="{}", error="failed"
        )
