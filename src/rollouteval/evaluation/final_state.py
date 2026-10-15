"""Compare an independently observed runtime snapshot with golden final state."""

from rollouteval.domain.scenario import InitialEnvironment, Scenario
from rollouteval.evaluation.results import GraderResult, GraderStatus


class FinalStateGrader:
    """Grade rollout phase and candidate weight from an environment snapshot."""

    def grade(
        self,
        scenario: Scenario,
        runtime_snapshot: InitialEnvironment,
    ) -> GraderResult:
        """Compare only independent runtime state with the golden final state."""
        expected = scenario.expected_final_state
        actual = runtime_snapshot.rollout
        mismatches: list[str] = []
        if actual.phase != expected.phase:
            mismatches.append(f"phase expected={expected.phase}, actual={actual.phase}")
        if actual.candidate_weight != expected.candidate_weight:
            mismatches.append(
                "candidate_weight "
                f"expected={expected.candidate_weight}, actual={actual.candidate_weight}"
            )

        if mismatches:
            return GraderResult(
                grader="final_state",
                status=GraderStatus.FAIL,
                reason="Final state mismatch: " + "; ".join(mismatches) + ".",
            )

        return GraderResult(
            grader="final_state",
            status=GraderStatus.PASS,
            reason=(
                "Final state matched: "
                f"phase={expected.phase}, candidate_weight={expected.candidate_weight}."
            ),
        )
