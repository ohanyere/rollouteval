"""Provider-neutral groundedness judging boundary."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from rollouteval.agent.contracts import AgentRunResult
from rollouteval.domain.scenario import Scenario
from rollouteval.evaluation.grounding import GroundingContext, build_grounding_context
from rollouteval.evaluation.results import GraderResult, GraderStatus


class GroundednessJudgment(BaseModel):
    """Structured semantic judgment about explanation support in observed evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    status: GraderStatus
    reason: str = Field(min_length=1)


@runtime_checkable
class GroundednessJudge(Protocol):
    """Synchronous interface for judging only a bounded grounding context."""

    def judge(self, context: GroundingContext) -> GroundednessJudgment:
        """Return a structured judgment for the supplied agent-visible context."""
        ...


class GroundednessGrader:
    """Build safe grounding context and delegate exactly one semantic judgment."""

    def __init__(self, judge: GroundednessJudge) -> None:
        self._judge = judge

    def grade(self, scenario: Scenario, run_result: AgentRunResult) -> GraderResult:
        """Judge explanation grounding independently of objective criteria."""
        context = build_grounding_context(scenario, run_result)
        judgment = self._judge.judge(context)
        return GraderResult(
            grader="groundedness",
            status=judgment.status,
            reason=judgment.reason,
        )
