import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentRunResult,
    AgentToolCall,
    AgentToolInteraction,
    AgentToolName,
)
from rollouteval.domain.scenario import Decision, Scenario
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import (
    EvidenceExtractionError,
    EvidenceId,
    EvidenceStatus,
    GroundingContext,
    GroundingEvidenceItem,
    build_grounding_context,
)
from rollouteval.scenarios import load_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_IDS = (
    "degraded-canary",
    "healthy-canary",
    "insufficient-evidence",
    "transient-anomaly",
    "conflicting-signals",
    "telemetry-unavailable",
)
TOOL_TO_ID = {
    AgentToolName.GET_ROLLOUT_STATE: EvidenceId.ROLLOUT_STATE,
    AgentToolName.GET_SERVICE_METRICS: EvidenceId.SERVICE_METRICS,
    AgentToolName.GET_RECENT_EVENTS: EvidenceId.RUNTIME_EVENTS,
}
ID_TO_TOOL = {value.value: key for key, value in TOOL_TO_ID.items()}


def _scenario(scenario_id: str = "degraded-canary") -> Scenario:
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _run(*interactions: AgentToolInteraction, decision: Decision = Decision.PAUSE):
    return AgentRunResult(
        final_answer=AgentFinalAnswer(
            kind="final", decision=decision, explanation="The observed data supports this action."
        ),
        interactions=tuple(interactions),
        steps=len(interactions) + 1,
    )


def _observed_interaction(scenario: Scenario, tool: AgentToolName) -> AgentToolInteraction:
    environment = RolloutEnvironment(scenario).snapshot()
    if tool is AgentToolName.GET_ROLLOUT_STATE:
        result = environment.rollout.model_dump_json()
    elif tool is AgentToolName.GET_SERVICE_METRICS:
        if environment.metrics is None:
            result = json.dumps(
                {
                    "available": False,
                    "metrics": None,
                    "message": "Service metrics are unavailable.",
                },
                separators=(",", ":"),
            )
        else:
            result = json.dumps(
                {
                    "available": True,
                    "metrics": environment.metrics.model_dump(mode="json"),
                    "message": None,
                },
                separators=(",", ":"),
            )
    elif tool is AgentToolName.GET_RECENT_EVENTS:
        result = json.dumps(
            [event.model_dump(mode="json") for event in environment.events],
            separators=(",", ":"),
        )
    else:
        result = "{}"
    return AgentToolInteraction.successful(_call(tool), result)


def _canonical_run(scenario: Scenario) -> AgentRunResult:
    interactions = tuple(
        _observed_interaction(scenario, ID_TO_TOOL[evidence_id])
        for evidence_id in scenario.required_evidence
    )
    return _run(*interactions, decision=scenario.expected_decision)


@pytest.mark.parametrize(
    ("evidence_id", "source_tool", "status"),
    [
        (EvidenceId.ROLLOUT_STATE, "get_rollout_state", EvidenceStatus.OBSERVED),
        (EvidenceId.SERVICE_METRICS, "get_service_metrics", EvidenceStatus.UNAVAILABLE),
        (EvidenceId.RUNTIME_EVENTS, "get_recent_events", EvidenceStatus.ERROR),
    ],
)
def test_grounding_evidence_item_represents_each_evidence_status(
    evidence_id: EvidenceId, source_tool: str, status: EvidenceStatus
) -> None:
    item = GroundingEvidenceItem(
        interaction_index=0,
        evidence_id=evidence_id,
        source_tool=source_tool,
        status=status,
        content="exact payload or error",
    )
    assert item.status is status


def test_grounding_evidence_item_rejects_bad_id_mutation_tool_and_bad_index() -> None:
    base = {
        "interaction_index": 0,
        "evidence_id": "rollout_state",
        "source_tool": "get_rollout_state",
        "status": "OBSERVED",
        "content": "{}",
    }
    with pytest.raises(ValidationError):
        GroundingEvidenceItem(**(base | {"evidence_id": "decision"}))
    with pytest.raises(ValidationError):
        GroundingEvidenceItem(**(base | {"source_tool": "rollback_rollout"}))
    with pytest.raises(ValidationError):
        GroundingEvidenceItem(**(base | {"interaction_index": -1}))
    with pytest.raises(ValidationError):
        GroundingEvidenceItem(**(base | {"evidence_id": "service_metrics"}))


@pytest.mark.parametrize("content", ["", "  ", "\n\t"])
def test_grounding_evidence_item_rejects_blank_content(content: str) -> None:
    with pytest.raises(ValidationError):
        GroundingEvidenceItem(
            interaction_index=0,
            evidence_id=EvidenceId.ROLLOUT_STATE,
            source_tool="get_rollout_state",
            status=EvidenceStatus.OBSERVED,
            content=content,
        )


def test_grounding_evidence_item_is_immutable() -> None:
    item = GroundingEvidenceItem(
        interaction_index=0,
        evidence_id=EvidenceId.ROLLOUT_STATE,
        source_tool="get_rollout_state",
        status=EvidenceStatus.OBSERVED,
        content="{}",
    )
    with pytest.raises(ValidationError):
        item.status = EvidenceStatus.ERROR


def test_grounding_context_has_exact_safe_fields_and_is_immutable() -> None:
    context = build_grounding_context(_scenario(), _run())
    assert set(GroundingContext.model_fields) == {"task", "decision", "explanation", "evidence"}
    assert context.evidence == ()
    with pytest.raises(ValidationError):
        context.task = "changed"


@pytest.mark.parametrize("field", ["task", "explanation"])
@pytest.mark.parametrize("value", ["", "  ", "\n\t"])
def test_grounding_context_rejects_blank_text(field: str, value: str) -> None:
    payload = {
        "task": "Task",
        "decision": Decision.PAUSE,
        "explanation": "Explanation",
        "evidence": (),
    }
    with pytest.raises(ValidationError):
        GroundingContext(**(payload | {field: value}))


def test_builder_copies_only_task_final_decision_and_explanation() -> None:
    scenario = _scenario()
    run_result = _run(decision=Decision.ROLLBACK)
    context = build_grounding_context(scenario, run_result)

    assert context.task == scenario.task
    assert context.decision is Decision.ROLLBACK
    assert context.explanation == run_result.final_answer.explanation
    assert "scenario" not in GroundingContext.model_fields
    assert "run_result" not in GroundingContext.model_fields


@pytest.mark.parametrize(
    ("tool", "evidence_id"),
    [
        (AgentToolName.GET_ROLLOUT_STATE, EvidenceId.ROLLOUT_STATE),
        (AgentToolName.GET_SERVICE_METRICS, EvidenceId.SERVICE_METRICS),
        (AgentToolName.GET_RECENT_EVENTS, EvidenceId.RUNTIME_EVENTS),
    ],
)
def test_successful_observation_is_preserved_as_grounding_evidence(
    tool: AgentToolName, evidence_id: EvidenceId
) -> None:
    scenario = _scenario()
    interaction = _observed_interaction(scenario, tool)

    context = build_grounding_context(scenario, _run(interaction))

    assert len(context.evidence) == 1
    assert context.evidence[0].evidence_id is evidence_id
    assert context.evidence[0].status is EvidenceStatus.OBSERVED
    assert context.evidence[0].content == interaction.result


def test_successful_metric_payload_is_preserved_exactly() -> None:
    scenario = _scenario()
    exact_result = ' { "available" : true, "metrics" : {"candidate_requests":8}, "message":null } '
    interaction = AgentToolInteraction.successful(_call(AgentToolName.GET_SERVICE_METRICS), exact_result)

    context = build_grounding_context(scenario, _run(interaction))

    assert context.evidence[0].content == exact_result
    assert '"candidate_requests":8' in context.evidence[0].content


def test_unavailable_service_metrics_preserve_payload_and_status() -> None:
    scenario = _scenario("telemetry-unavailable")
    interaction = _observed_interaction(scenario, AgentToolName.GET_SERVICE_METRICS)

    context = build_grounding_context(scenario, _run(interaction))

    assert context.evidence[0].status is EvidenceStatus.UNAVAILABLE
    assert context.evidence[0].content == interaction.result
    assert json.loads(context.evidence[0].content)["available"] is False


def test_observation_error_preserves_original_error_content() -> None:
    scenario = _scenario()
    error = "metrics observation failed: source unavailable "
    interaction = AgentToolInteraction.failed(_call(AgentToolName.GET_SERVICE_METRICS), error)

    context = build_grounding_context(scenario, _run(interaction))

    assert context.evidence[0].status is EvidenceStatus.ERROR
    assert context.evidence[0].content == error


@pytest.mark.parametrize(
    "tool",
    [
        AgentToolName.PROMOTE_ROLLOUT,
        AgentToolName.PAUSE_ROLLOUT,
        AgentToolName.ROLLBACK_ROLLOUT,
    ],
)
def test_mutation_interactions_are_excluded_from_grounding_evidence(tool: AgentToolName) -> None:
    scenario = _scenario()
    interaction = AgentToolInteraction.successful(_call(tool), '{"phase":"Promoted"}')
    assert build_grounding_context(scenario, _run(interaction)).evidence == ()


def test_evidence_preserves_order_and_repeated_observations() -> None:
    scenario = _scenario()
    rollout = _observed_interaction(scenario, AgentToolName.GET_ROLLOUT_STATE)
    metrics = _observed_interaction(scenario, AgentToolName.GET_SERVICE_METRICS)
    events = _observed_interaction(scenario, AgentToolName.GET_RECENT_EVENTS)

    context = build_grounding_context(scenario, _run(metrics, rollout, metrics, events))

    assert [item.source_tool for item in context.evidence] == [
        "get_service_metrics",
        "get_rollout_state",
        "get_service_metrics",
        "get_recent_events",
    ]
    assert [item.interaction_index for item in context.evidence] == [0, 1, 2, 3]


def test_unavailable_metrics_followed_by_observed_metrics_keeps_both() -> None:
    scenario = _scenario("telemetry-unavailable")
    unavailable = _observed_interaction(scenario, AgentToolName.GET_SERVICE_METRICS)
    available_payload = json.dumps(
        {"available": True, "metrics": {"candidate_requests": 1}, "message": None}
    )
    observed = AgentToolInteraction.successful(
        _call(AgentToolName.GET_SERVICE_METRICS), available_payload
    )

    context = build_grounding_context(scenario, _run(unavailable, observed))

    assert [item.status for item in context.evidence] == [
        EvidenceStatus.UNAVAILABLE,
        EvidenceStatus.OBSERVED,
    ]


def test_context_has_no_golden_truth_snapshot_trace_or_grader_data() -> None:
    context = build_grounding_context(_scenario(), _run())
    fields = set(GroundingContext.model_fields)
    assert not {
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
        "scenario_id",
        "scenario_name",
        "runtime_snapshot",
        "evaluation",
        "trace",
        "model_usage",
        "cost_estimate",
        "duration_ms",
        "model",
        "grader_results",
    } & fields
    assert not hasattr(context, "scenario")
    assert not hasattr(context, "run_result")


def test_unobserved_runtime_events_and_metrics_do_not_leak_from_scenario() -> None:
    scenario = _scenario("transient-anomaly")
    interaction = _observed_interaction(scenario, AgentToolName.GET_ROLLOUT_STATE)

    context = build_grounding_context(scenario, _run(interaction))
    serialized = context.model_dump_json()

    assert [item.evidence_id for item in context.evidence] == [EvidenceId.ROLLOUT_STATE]
    assert "candidate_error_rate" not in serialized
    assert "candidate_requests" not in serialized
    for event in scenario.initial_environment.events:
        assert event.message not in serialized


def test_malformed_successful_metrics_raise_existing_extraction_error() -> None:
    scenario = _scenario()
    malformed = AgentToolInteraction.successful(
        _call(AgentToolName.GET_SERVICE_METRICS), "not valid JSON"
    )

    with pytest.raises(EvidenceExtractionError):
        build_grounding_context(scenario, _run(malformed))


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_canonical_scenario_context_contains_only_requested_evidence(scenario_id: str) -> None:
    scenario = _scenario(scenario_id)

    context = build_grounding_context(scenario, _canonical_run(scenario))

    assert tuple(item.evidence_id.value for item in context.evidence) == scenario.required_evidence
    if scenario_id == "insufficient-evidence":
        metrics = next(item for item in context.evidence if item.evidence_id is EvidenceId.SERVICE_METRICS)
        assert json.loads(metrics.content)["metrics"]["candidate_requests"] == 8
        assert metrics.status is EvidenceStatus.OBSERVED
    if scenario_id in {"transient-anomaly", "conflicting-signals"}:
        assert any(item.evidence_id is EvidenceId.RUNTIME_EVENTS for item in context.evidence)
    if scenario_id == "telemetry-unavailable":
        metrics = next(item for item in context.evidence if item.evidence_id is EvidenceId.SERVICE_METRICS)
        assert metrics.status is EvidenceStatus.UNAVAILABLE
        assert json.loads(metrics.content)["available"] is False


def test_building_context_does_not_mutate_scenario_or_run_result() -> None:
    scenario = _scenario()
    interaction = _observed_interaction(scenario, AgentToolName.GET_ROLLOUT_STATE)
    run_result = _run(interaction, decision=Decision.ROLLBACK)
    scenario_before = scenario.model_copy(deep=True)
    run_before = run_result.model_copy(deep=True)

    build_grounding_context(scenario, run_result)

    assert scenario == scenario_before
    assert run_result == run_before
