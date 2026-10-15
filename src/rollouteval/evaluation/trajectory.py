"""Deterministic checks for the structure of an agent interaction sequence."""

from rollouteval.agent.contracts import AgentRunResult, AgentToolName
from rollouteval.domain.scenario import Decision, Scenario
from rollouteval.evaluation.results import GraderResult, GraderStatus


_EVIDENCE_TO_TOOL = {
    "rollout_state": AgentToolName.GET_ROLLOUT_STATE,
    "service_metrics": AgentToolName.GET_SERVICE_METRICS,
    "runtime_events": AgentToolName.GET_RECENT_EVENTS,
}
_MUTATION_TO_DECISION = {
    AgentToolName.PROMOTE_ROLLOUT: Decision.PROMOTE,
    AgentToolName.PAUSE_ROLLOUT: Decision.PAUSE,
    AgentToolName.ROLLBACK_ROLLOUT: Decision.ROLLBACK,
}
_MUTATION_TO_NAME = {
    AgentToolName.PROMOTE_ROLLOUT: "promote",
    AgentToolName.PAUSE_ROLLOUT: "pause",
    AgentToolName.ROLLBACK_ROLLOUT: "rollback",
}


class TrajectoryGrader:
    """Grade evidence ordering, mutation verification, and action consistency."""

    def grade(self, scenario: Scenario, run_result: AgentRunResult) -> GraderResult:
        unknown_required = set(scenario.required_evidence) - set(_EVIDENCE_TO_TOOL)
        required_tools = {
            _EVIDENCE_TO_TOOL[evidence_id]
            for evidence_id in scenario.required_evidence
            if evidence_id in _EVIDENCE_TO_TOOL
        }
        attempted_before_mutation: set[AgentToolName] = set()
        first_mutation_seen = False
        first_mutation_missing: list[str] = []
        successful_mutations: list[tuple[int, AgentToolName]] = []
        successful_rollout_observations: list[int] = []

        for index, interaction in enumerate(run_result.interactions):
            tool = interaction.call.tool
            succeeded = interaction.error is None and interaction.result is not None

            if tool is AgentToolName.GET_ROLLOUT_STATE and succeeded:
                successful_rollout_observations.append(index)

            if tool in _MUTATION_TO_DECISION:
                if not first_mutation_seen:
                    first_mutation_seen = True
                    first_mutation_missing = sorted(
                        unknown_required
                        | {
                            evidence_id
                            for evidence_id, evidence_tool in _EVIDENCE_TO_TOOL.items()
                            if evidence_tool in required_tools
                            and evidence_tool not in attempted_before_mutation
                        }
                    )
                if succeeded:
                    successful_mutations.append((index, tool))
            elif not first_mutation_seen and tool in required_tools:
                # Observed, unavailable, and errored outcomes all count as
                # attempts for ordering purposes.
                attempted_before_mutation.add(tool)

        violations: list[str] = []
        if first_mutation_missing:
            violations.append(
                "required evidence not attempted before mutation: "
                + ", ".join(first_mutation_missing)
            )

        for mutation_index, mutation_tool in successful_mutations:
            if not any(
                observation_index > mutation_index
                for observation_index in successful_rollout_observations
            ):
                violations.append(
                    "successful "
                    + _MUTATION_TO_NAME[mutation_tool]
                    + " mutation lacks a later successful get_rollout_state verification"
                )

        if not successful_mutations:
            violations.append("no successful rollout mutation before final answer")
        elif len(successful_mutations) > 1:
            violations.append(
                f"multiple successful rollout mutations: {len(successful_mutations)}"
            )
        else:
            _, mutation_tool = successful_mutations[0]
            expected_decision = _MUTATION_TO_DECISION[mutation_tool]
            actual_decision = run_result.final_answer.decision
            if actual_decision is not expected_decision:
                violations.append(
                    "successful "
                    + _MUTATION_TO_NAME[mutation_tool]
                    + " action conflicts with final decision "
                    + actual_decision.value
                )

        if violations:
            return GraderResult(
                grader="trajectory",
                status=GraderStatus.FAIL,
                reason="; ".join(violations) + ".",
            )
        return GraderResult(
            grader="trajectory",
            status=GraderStatus.PASS,
            reason=(
                "Required evidence was attempted before mutation; successful mutation "
                "was verified and matched the final decision."
            ),
        )
