"""Compose deterministic criteria and optional groundedness without scoring."""

from pydantic import BaseModel, ConfigDict

from rollouteval.agent.contracts import AgentRunResult
from rollouteval.domain.scenario import InitialEnvironment, Scenario
from rollouteval.evaluation.abstention import AbstentionGrader
from rollouteval.evaluation.decision import DecisionGrader
from rollouteval.evaluation.evidence import EvidenceSufficiencyGrader
from rollouteval.evaluation.final_state import FinalStateGrader
from rollouteval.evaluation.groundedness import (
    GROUNDEDNESS_EVALUATOR_VERSION,
    GroundednessGrader,
    GroundednessJudge,
)
from rollouteval.evaluation.grounding import build_grounding_context
from rollouteval.evaluation.results import GraderResult
from rollouteval.evaluation.safety import SafetyGrader
from rollouteval.evaluation.trajectory import TrajectoryGrader


class DeterministicEvaluation(BaseModel):
    """Immutable, individually inspectable results for all six criteria."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: GraderResult
    final_state: GraderResult
    evidence_sufficiency: GraderResult
    trajectory: GraderResult
    safety: GraderResult
    abstention: GraderResult
    groundedness: GraderResult | None = None

    def results(self) -> tuple[GraderResult, ...]:
        """Return criterion results in the engine's stable execution order."""
        deterministic = (
            self.decision,
            self.final_state,
            self.evidence_sufficiency,
            self.trajectory,
            self.safety,
            self.abstention,
        )
        optional = (self.groundedness,) if self.groundedness is not None else ()
        return deterministic + optional


class DeterministicEvaluationEngine:
    """Run existing deterministic graders in a fixed order, without scoring."""

    def __init__(self) -> None:
        self._decision = DecisionGrader()
        self._final_state = FinalStateGrader()
        self._evidence_sufficiency = EvidenceSufficiencyGrader()
        self._trajectory = TrajectoryGrader()
        self._safety = SafetyGrader()
        self._abstention = AbstentionGrader()

    def evaluate(
        self,
        scenario: Scenario,
        run_result: AgentRunResult,
        runtime_snapshot: InitialEnvironment,
        *,
        groundedness_judge: GroundednessJudge | None = None,
    ) -> DeterministicEvaluation:
        """Evaluate criteria sequentially using the caller-supplied snapshot."""
        decision = self._decision.grade(scenario, run_result)
        final_state = self._final_state.grade(scenario, runtime_snapshot)
        evidence_sufficiency = self._evidence_sufficiency.grade(scenario, run_result)
        trajectory = self._trajectory.grade(scenario, run_result)
        safety = self._safety.grade(scenario, run_result)
        abstention = self._abstention.grade(scenario, run_result)
        groundedness = None
        if groundedness_judge is not None:
            try:
                groundedness = GroundednessGrader(groundedness_judge).grade(
                    scenario, run_result
                )
            except Exception as exc:
                try:
                    context = build_grounding_context(scenario, run_result)
                    evidence_references = tuple(
                        f"{item.evidence_id.value}@{item.interaction_index}"
                        for item in context.evidence
                    )
                except Exception:
                    evidence_references = ()
                groundedness = GraderResult(
                    grader="groundedness",
                    status="INCONCLUSIVE",
                    reason=f"Groundedness evaluation failed: {type(exc).__name__}.",
                    evaluator_version=GROUNDEDNESS_EVALUATOR_VERSION,
                    criterion_id="groundedness",
                    evidence_references=evidence_references,
                    judge_provider=getattr(groundedness_judge, "provider_name", None),
                    judge_model=getattr(groundedness_judge, "model", None),
                    prompt_sha256=getattr(groundedness_judge, "prompt_sha256", None),
                    judge_error=type(exc).__name__,
                )
        return DeterministicEvaluation(
            decision=decision,
            final_state=final_state,
            evidence_sufficiency=evidence_sufficiency,
            trajectory=trajectory,
            safety=safety,
            abstention=abstention,
            groundedness=groundedness,
        )
