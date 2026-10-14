from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentOutput,
    AgentTask,
    AgentToolCall,
    AgentToolName,
)
from rollouteval.domain.scenario import Decision
from rollouteval.scenarios import load_scenario


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_IDS = (
    "degraded-canary",
    "healthy-canary",
    "insufficient-evidence",
    "transient-anomaly",
    "conflicting-signals",
    "telemetry-unavailable",
)
TOOL_NAMES = tuple(AgentToolName)


def test_agent_task_accepts_non_empty_task() -> None:
    assert AgentTask(task="Assess the rollout.").task == "Assess the rollout."


@pytest.mark.parametrize("task", ["", "   ", "\n\t"])
def test_agent_task_rejects_empty_or_whitespace_task(task: str) -> None:
    with pytest.raises(ValidationError):
        AgentTask(task=task)


@pytest.mark.parametrize(
    "private_field",
    ["expected_decision", "expected_final_state", "required_evidence", "max_mutations"],
)
def test_agent_task_has_no_evaluator_private_fields(private_field: str) -> None:
    assert private_field not in AgentTask.model_fields
    with pytest.raises(ValidationError):
        AgentTask(task="Assess rollout.", **{private_field: "hidden"})


def test_agent_task_from_scenario_copies_only_safe_task_text() -> None:
    scenario = load_scenario(SCENARIOS / "degraded-canary.yaml")

    task = AgentTask.from_scenario(scenario)

    assert task.task == scenario.task
    assert set(AgentTask.model_fields) == {"task"}
    assert not any(value is scenario for value in task.__dict__.values())


@pytest.mark.parametrize("tool", TOOL_NAMES)
def test_agent_tool_call_accepts_each_allowed_tool(tool: AgentToolName) -> None:
    call = AgentToolCall.model_validate({"kind": "tool_call", "tool": tool.value, "arguments": {}})

    assert call.tool is tool
    assert call.arguments.model_dump() == {}


def test_agent_tool_call_rejects_unknown_tool_name() -> None:
    with pytest.raises(ValidationError):
        AgentToolCall.model_validate(
            {"kind": "tool_call", "tool": "delete_cluster", "arguments": {}}
        )


def test_agent_tool_call_rejects_non_empty_arguments() -> None:
    with pytest.raises(ValidationError):
        AgentToolCall.model_validate(
            {
                "kind": "tool_call",
                "tool": "get_service_metrics",
                "arguments": {"expected_decision": "rollback"},
            }
        )


def test_agent_tool_call_rejects_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        AgentToolCall.model_validate(
            {"kind": "tool_call", "tool": "get_rollout_state", "arguments": {}, "extra": True}
        )


@pytest.mark.parametrize("decision", list(Decision))
def test_agent_final_answer_accepts_each_decision(decision: Decision) -> None:
    answer = AgentFinalAnswer.model_validate(
        {"kind": "final", "decision": decision.value, "explanation": "Evidence supports this."}
    )

    assert answer.decision is decision


def test_agent_final_answer_rejects_invalid_decision() -> None:
    with pytest.raises(ValidationError):
        AgentFinalAnswer(kind="final", decision="wait", explanation="Wait for evidence.")


@pytest.mark.parametrize("explanation", ["", "   ", "\n\t"])
def test_agent_final_answer_rejects_empty_or_whitespace_explanation(explanation: str) -> None:
    with pytest.raises(ValidationError):
        AgentFinalAnswer(kind="final", decision=Decision.PAUSE, explanation=explanation)


def test_agent_final_answer_rejects_undeclared_fields() -> None:
    with pytest.raises(ValidationError):
        AgentFinalAnswer.model_validate(
            {
                "kind": "final",
                "decision": "pause",
                "explanation": "Insufficient evidence.",
                "reasoning_trace": "private thoughts",
            }
        )


def test_tool_call_validates_through_agent_output_union() -> None:
    output = TypeAdapter(AgentOutput).validate_python(
        {"kind": "tool_call", "tool": "get_service_metrics", "arguments": {}}
    )

    assert isinstance(output, AgentToolCall)


def test_final_answer_validates_through_agent_output_union() -> None:
    output = TypeAdapter(AgentOutput).validate_python(
        {"kind": "final", "decision": "promote", "explanation": "Signals are healthy."}
    )

    assert isinstance(output, AgentFinalAnswer)


def test_agent_output_rejects_malformed_discriminator() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(AgentOutput).validate_python(
            {"kind": "unknown", "tool": "get_rollout_state", "arguments": {}}
        )


def test_tool_call_payload_cannot_masquerade_as_final_answer() -> None:
    with pytest.raises(ValidationError):
        AgentFinalAnswer.model_validate(
            {"kind": "final", "tool": "get_rollout_state", "arguments": {}}
        )


def test_final_answer_payload_cannot_masquerade_as_tool_call() -> None:
    with pytest.raises(ValidationError):
        AgentToolCall.model_validate(
            {
                "kind": "tool_call",
                "tool": "get_rollout_state",
                "arguments": {},
                "decision": "pause",
                "explanation": "Insufficient evidence.",
            }
        )


def test_agent_contracts_are_immutable() -> None:
    task = AgentTask(task="Assess rollout.")
    call = AgentToolCall(kind="tool_call", tool=AgentToolName.GET_ROLLOUT_STATE)
    answer = AgentFinalAnswer(
        kind="final", decision=Decision.PROMOTE, explanation="Signals are healthy."
    )

    for model, field, value in (
        (task, "task", "changed"),
        (call, "tool", AgentToolName.PAUSE_ROLLOUT),
        (answer, "explanation", "changed"),
    ):
        with pytest.raises(ValidationError):
            setattr(model, field, value)


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_all_scenarios_produce_safe_agent_tasks(scenario_id: str) -> None:
    scenario = load_scenario(SCENARIOS / f"{scenario_id}.yaml")

    task = AgentTask.from_scenario(scenario)

    assert task.task == scenario.task
    assert set(AgentTask.model_fields) == {"task"}
    assert not {"expected_decision", "expected_final_state", "required_evidence", "max_mutations"} & set(
        task.model_dump()
    )
