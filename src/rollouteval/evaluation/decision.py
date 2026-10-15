"""Compare a declared agent decision with the scenario's expected decision."""

from rollouteval.agent.contracts import AgentRunResult
from rollouteval.domain.scenario import Scenario
from rollouteval.evaluation.results import GraderResult, GraderStatus


class DecisionGrader:
    """Grade only the final decision declared by an agent run."""

    def grade(self, scenario: Scenario, run_result: AgentRunResult) -> GraderResult:
        expected = scenario.expected_decision
        actual = run_result.final_answer.decision
        status = GraderStatus.PASS if actual is expected else GraderStatus.FAIL
        comparison = f"expected={expected.value}, actual={actual.value}"
        if status is GraderStatus.PASS:
            reason = f"Decision matched: {comparison}."
        else:
            reason = f"Decision mismatch: {comparison}."
        return GraderResult(grader="decision", status=status, reason=reason)
