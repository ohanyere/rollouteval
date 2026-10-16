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
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import (
    AbstentionGrader,
    DeterministicEvaluation,
    DeterministicEvaluationEngine,
    EvidenceExtractionError,
    EvidenceSufficiencyGrader,
    FinalStateGrader,
    GroundednessGrader,
    GroundednessJudge,
    GroundednessJudgment,
    GraderResult,
    GraderStatus,
    build_grounding_context,
)
from rollouteval.evaluation.safety import SafetyGrader
from rollouteval.evaluation.trajectory import TrajectoryGrader
from rollouteval.scenarios import load_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"


def _scenario(scenario_id: str = "degraded-canary"):
    return load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")


def _call(tool: AgentToolName) -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=tool)


def _answer(decision: Decision, explanation: str = "The observed rollout is progressing."):
    return AgentFinalAnswer(kind="final", decision=decision, explanation=explanation)


def _run(*interactions: AgentToolInteraction, decision: Decision = Decision.ROLLBACK, explanation: str = "The observed rollout is progressing."):
    return AgentRunResult(
        final_answer=_answer(decision, explanation),
        interactions=tuple(interactions),
        steps=len(interactions) + 1,
    )


def _observation(tool: AgentToolName, result: str | None = None):
    if result is None:
        if tool is AgentToolName.GET_ROLLOUT_STATE:
            result = '{"phase":"Progressing","candidate_weight":20}'
        elif tool is AgentToolName.GET_SERVICE_METRICS:
            result = json.dumps(
                {
                    "available": True,
                    "metrics": {"candidate_requests": 450},
                    "message": None,
                }
            )
        elif tool is AgentToolName.GET_RECENT_EVENTS:
            result = "[]"
        else:
            result = "{}"
    return AgentToolInteraction.successful(_call(tool), result)


class ScriptedJudge:
    """Test-only response double; it does not implement groundedness logic."""

    def __init__(self, judgment: GroundednessJudgment | None = None, error=None) -> None:
        self.judgment = judgment or GroundednessJudgment(
            status=GraderStatus.PASS,
            reason="The explanation is supported by the supplied observations.",
        )
        self.error = error
        self.contexts = []
        self.args = []
        self.kwargs = []

    def judge(self, context):
        self.args.append((context,))
        self.kwargs.append({})
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return self.judgment


@pytest.mark.parametrize("status", list(GraderStatus))
def test_groundedness_judgment_accepts_existing_grader_status(status: GraderStatus) -> None:
    judgment = GroundednessJudgment(status=status, reason="Judgment reason.")
    assert judgment.status is status
    assert GroundednessJudgment.model_fields["status"].annotation is GraderStatus
    assert set(GroundednessJudgment.model_fields) == {"status", "reason"}


@pytest.mark.parametrize("reason", ["", "  ", "\n\t"])
def test_groundedness_judgment_rejects_blank_reason(reason: str) -> None:
    with pytest.raises(ValidationError):
        GroundednessJudgment(status=GraderStatus.PASS, reason=reason)


def test_groundedness_judgment_is_immutable_and_forbids_extra_fields() -> None:
    judgment = GroundednessJudgment(status=GraderStatus.FAIL, reason="Unsupported metric claim.")
    with pytest.raises(ValidationError):
        judgment.status = GraderStatus.PASS
    with pytest.raises(ValidationError):
        GroundednessJudgment(
            status=GraderStatus.PASS,
            reason="Supported.",
            confidence=0.99,
        )


def test_scripted_test_judge_structurally_satisfies_protocol() -> None:
    judge = ScriptedJudge()
    assert isinstance(judge, GroundednessJudge)


@pytest.mark.parametrize("status", list(GraderStatus))
def test_grader_propagates_judgment_status_and_reason_once(status: GraderStatus) -> None:
    scenario = _scenario()
    run_result = _run()
    judgment = GroundednessJudgment(status=status, reason="Exact scripted reason.")
    judge = ScriptedJudge(judgment)

    result = GroundednessGrader(judge).grade(scenario, run_result)

    assert isinstance(result, GraderResult)
    assert result.grader == "groundedness"
    assert result.status is status
    assert result.reason == judgment.reason
    assert len(judge.contexts) == 1
    assert len(judge.args[0]) == 1
    assert judge.kwargs[0] == {}


def test_judge_receives_only_safe_grounding_context() -> None:
    scenario = _scenario()
    interaction = _observation(AgentToolName.GET_ROLLOUT_STATE)
    run_result = _run(interaction, decision=Decision.PROMOTE)
    judge = ScriptedJudge()

    GroundednessGrader(judge).grade(scenario, run_result)
    context = judge.contexts[0]

    assert context.task == scenario.task
    assert context.decision is Decision.PROMOTE
    assert context.explanation == run_result.final_answer.explanation
    assert context.evidence[0].content == interaction.result
    assert not hasattr(context, "scenario")
    assert not hasattr(context, "run_result")
    assert not {
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
        "runtime_snapshot",
        "evaluation",
        "trace",
        "model_usage",
        "cost_estimate",
    } & set(type(context).model_fields)
    assert "unexpected-canary-identifier" not in context.model_dump_json()


def test_wrong_golden_decision_can_coexist_with_groundedness_pass() -> None:
    scenario = _scenario("healthy-canary")
    run_result = _run(
        _observation(AgentToolName.GET_SERVICE_METRICS),
        decision=Decision.ROLLBACK,
        explanation="The observed candidate metrics are healthy.",
    )
    judge = ScriptedJudge(
        GroundednessJudgment(status=GraderStatus.PASS, reason="Metrics claim is supported.")
    )

    groundedness = GroundednessGrader(judge).grade(scenario, run_result)
    from rollouteval.evaluation import DecisionGrader

    decision = DecisionGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.PASS
    assert decision.status is GraderStatus.FAIL


def test_correct_golden_decision_can_coexist_with_groundedness_fail() -> None:
    scenario = _scenario("healthy-canary")
    run_result = _run(
        _observation(AgentToolName.GET_SERVICE_METRICS),
        decision=Decision.PROMOTE,
        explanation="The unseen runtime logs prove database corruption.",
    )
    judge = ScriptedJudge(
        GroundednessJudgment(status=GraderStatus.FAIL, reason="No logs were observed.")
    )

    groundedness = GroundednessGrader(judge).grade(scenario, run_result)
    from rollouteval.evaluation import DecisionGrader

    decision = DecisionGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.FAIL
    assert decision.status is GraderStatus.PASS


def test_missing_required_evidence_can_coexist_with_groundedness_pass() -> None:
    scenario = _scenario("degraded-canary")
    run_result = _run(
        _observation(AgentToolName.GET_ROLLOUT_STATE),
        explanation="The rollout is currently progressing.",
    )
    judge = ScriptedJudge()

    groundedness = GroundednessGrader(judge).grade(scenario, run_result)
    evidence = EvidenceSufficiencyGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.PASS
    assert evidence.status is GraderStatus.FAIL


def test_unsupported_claim_outcome_is_supplied_by_judge_not_local_heuristic() -> None:
    scenario = _scenario("degraded-canary")
    run_result = _run(
        _observation(AgentToolName.GET_ROLLOUT_STATE),
        explanation="Candidate error rate is 8.7%.",
    )
    judgment = GroundednessJudgment(
        status=GraderStatus.FAIL,
        reason="Service metrics were not present in observed evidence.",
    )
    judge = ScriptedJudge(judgment)

    result = GroundednessGrader(judge).grade(scenario, run_result)

    assert result.status is GraderStatus.FAIL
    assert result.reason == judgment.reason
    assert all(item.evidence_id.value != "service_metrics" for item in judge.contexts[0].evidence)


def test_complete_required_evidence_can_coexist_with_groundedness_fail() -> None:
    scenario = _scenario("degraded-canary")
    run_result = _run(
        _observation(AgentToolName.GET_ROLLOUT_STATE),
        _observation(AgentToolName.GET_SERVICE_METRICS),
        explanation="The observed logs report data loss.",
    )
    judge = ScriptedJudge(
        GroundednessJudgment(status=GraderStatus.FAIL, reason="Logs were not supplied.")
    )

    groundedness = GroundednessGrader(judge).grade(scenario, run_result)
    evidence = EvidenceSufficiencyGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.FAIL
    assert evidence.status is GraderStatus.PASS


def test_unavailable_telemetry_can_have_inconclusive_evidence_and_groundedness_pass() -> None:
    scenario = _scenario("telemetry-unavailable")
    unavailable = json.dumps(
        {"available": False, "metrics": None, "message": "Service metrics are unavailable."}
    )
    run_result = _run(
        _observation(AgentToolName.GET_ROLLOUT_STATE),
        _observation(AgentToolName.GET_SERVICE_METRICS, unavailable),
        decision=Decision.PAUSE,
        explanation="Service metrics were unavailable, so I paused.",
    )
    judge = ScriptedJudge()

    groundedness = GroundednessGrader(judge).grade(scenario, run_result)
    evidence = EvidenceSufficiencyGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.PASS
    assert evidence.status is GraderStatus.INCONCLUSIVE
    assert judge.contexts[0].evidence[1].status.value == "UNAVAILABLE"


def test_bad_trajectory_can_coexist_with_groundedness_pass() -> None:
    scenario = _scenario("degraded-canary")
    run_result = _run(explanation="The rollout is currently progressing.")
    groundedness = GroundednessGrader(ScriptedJudge()).grade(scenario, run_result)
    trajectory = TrajectoryGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.PASS
    assert trajectory.status is GraderStatus.FAIL


def test_safety_failure_can_coexist_with_groundedness_pass() -> None:
    scenario = _scenario("degraded-canary")
    run_result = _run(
        AgentToolInteraction.successful(_call(AgentToolName.ROLLBACK_ROLLOUT), '{"phase":"RolledBack"}'),
        AgentToolInteraction.failed(_call(AgentToolName.PROMOTE_ROLLOUT), "mutation limit exceeded"),
        explanation="The rollout was rolled back.",
    )
    groundedness = GroundednessGrader(ScriptedJudge()).grade(scenario, run_result)
    safety = SafetyGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.PASS
    assert safety.status is GraderStatus.FAIL


def test_wrong_final_state_can_coexist_with_groundedness_pass() -> None:
    scenario = _scenario("healthy-canary")
    run_result = _run(
        _observation(AgentToolName.GET_SERVICE_METRICS),
        decision=Decision.ROLLBACK,
        explanation="The observed candidate metrics are healthy.",
    )
    runtime_snapshot = RolloutEnvironment(scenario).snapshot()
    groundedness = GroundednessGrader(ScriptedJudge()).grade(scenario, run_result)
    final_state = FinalStateGrader().grade(scenario, runtime_snapshot)

    assert groundedness.status is GraderStatus.PASS
    assert final_state.status is GraderStatus.FAIL


def test_abstention_result_is_independent_of_groundedness_result() -> None:
    scenario = _scenario("insufficient-evidence")
    run_result = _run(
        decision=Decision.PROMOTE,
        explanation="The rollout is currently progressing.",
    )
    groundedness = GroundednessGrader(ScriptedJudge()).grade(scenario, run_result)
    abstention = AbstentionGrader().grade(scenario, run_result)

    assert groundedness.status is GraderStatus.PASS
    assert abstention.status is GraderStatus.FAIL


def test_judge_exception_propagates_unchanged() -> None:
    scenario = _scenario()
    failure = RuntimeError("scripted judge failure")
    judge = ScriptedJudge(error=failure)

    with pytest.raises(RuntimeError) as caught:
        GroundednessGrader(judge).grade(scenario, _run())

    assert caught.value is failure
    assert len(judge.contexts) == 1


def test_malformed_evidence_fails_before_judge_is_called() -> None:
    scenario = _scenario()
    run_result = _run(
        AgentToolInteraction.successful(_call(AgentToolName.GET_SERVICE_METRICS), "not JSON")
    )
    judge = ScriptedJudge()

    with pytest.raises(EvidenceExtractionError):
        GroundednessGrader(judge).grade(scenario, run_result)

    assert judge.contexts == []


def test_deterministic_evaluation_keeps_groundedness_opt_in_without_a_judge(monkeypatch) -> None:
    import rollouteval.evaluation.groundedness as groundedness_module

    assert set(DeterministicEvaluation.model_fields) == {
        "decision",
        "final_state",
        "evidence_sufficiency",
        "trajectory",
        "safety",
        "abstention",
        "groundedness",
    }
    scenario = _scenario()
    run_result = _run()

    def forbid_judge(*args, **kwargs):
        raise AssertionError("the deterministic engine must not invoke groundedness")

    monkeypatch.setattr(groundedness_module.GroundednessGrader, "grade", forbid_judge)
    evaluation = DeterministicEvaluationEngine().evaluate(
        scenario,
        run_result,
        RolloutEnvironment(scenario).snapshot(),
    )

    assert len(evaluation.results()) == 6
    assert evaluation.groundedness is None


def test_grader_does_not_mutate_scenario_run_result_or_received_context() -> None:
    scenario = _scenario()
    run_result = _run(_observation(AgentToolName.GET_ROLLOUT_STATE))
    scenario_before = scenario.model_copy(deep=True)
    run_before = run_result.model_copy(deep=True)
    judge = ScriptedJudge()

    GroundednessGrader(judge).grade(scenario, run_result)
    context = judge.contexts[0]
    context_before = context.model_copy(deep=True)

    assert scenario == scenario_before
    assert run_result == run_before
    assert context == context_before
    with pytest.raises(ValidationError):
        context.task = "changed"
