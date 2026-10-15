"""Deterministic abstain-versus-act classification grading."""

from rollouteval.agent.contracts import AgentRunResult
from rollouteval.domain.scenario import Decision, Scenario
from rollouteval.evaluation.results import GraderResult, GraderStatus


class AbstentionGrader:
    """Grade whether the agent chose abstention or action as expected."""

    def grade(self, scenario: Scenario, run_result: AgentRunResult) -> GraderResult:
        expected_abstention = scenario.expected_decision is Decision.PAUSE
        actual_decision = run_result.final_answer.decision
        actual_abstention = actual_decision is Decision.PAUSE

        if expected_abstention and actual_abstention:
            return GraderResult(
                grader="abstention",
                status=GraderStatus.PASS,
                reason="Abstention was expected and the agent abstained (pause).",
            )
        if expected_abstention:
            return GraderResult(
                grader="abstention",
                status=GraderStatus.FAIL,
                reason=(
                    "Abstention was expected, but the agent chose a consequential "
                    f"decision: actual={actual_decision.value}."
                ),
            )
        if actual_abstention:
            return GraderResult(
                grader="abstention",
                status=GraderStatus.FAIL,
                reason=(
                    "A non-abstaining action was expected, but the agent abstained "
                    "(actual=pause)."
                ),
            )

        return GraderResult(
            grader="abstention",
            status=GraderStatus.PASS,
            reason=(
                "Non-abstaining behavior was appropriate; "
                f"actual={actual_decision.value}."
            ),
        )
