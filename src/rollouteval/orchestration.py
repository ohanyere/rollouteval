"""Compose one agent episode and its separate observable artifacts."""

from pydantic import BaseModel, ConfigDict

from rollouteval.agent.contracts import AgentRunResult, AgentTask
from rollouteval.agent.runner import AgentRunner
from rollouteval.domain.scenario import InitialEnvironment, Scenario
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation.engine import (
    DeterministicEvaluation,
    DeterministicEvaluationEngine,
)
from rollouteval.evaluation.groundedness import GroundednessJudge
from rollouteval.observability.cost import (
    CostEstimator,
    PricingSnapshot,
    RunCostEstimate,
)
from rollouteval.observability.trace import RunTrace, TraceRecorder


class OrchestrationConfigurationError(ValueError):
    """Raised for invalid caller-supplied orchestration configuration."""


class ObservableRun(BaseModel):
    """Immutable, separate artifacts from one completed rollout episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_run: AgentRunResult
    runtime_snapshot: InitialEnvironment
    trace: RunTrace
    evaluation: DeterministicEvaluation
    cost_estimate: RunCostEstimate | None = None


class ObservableRunOrchestrator:
    """Run and evaluate one episode using caller-owned runtime and trace state.

    The caller must wire the supplied runner's rollout tools to the same
    environment supplied here. The orchestrator deliberately does not inspect
    runner internals to verify that precondition.
    """

    def execute(
        self,
        *,
        scenario: Scenario,
        runner: AgentRunner,
        environment: RolloutEnvironment,
        trace_recorder: TraceRecorder,
        pricing: PricingSnapshot | None = None,
        groundedness_judge: GroundednessJudge | None = None,
    ) -> ObservableRun:
        """Execute one run, capture independent artifacts, then evaluate them."""
        trace_identity = trace_recorder.snapshot()
        if trace_identity.scenario_id != scenario.id:
            raise OrchestrationConfigurationError(
                "trace recorder scenario_id does not match scenario.id"
            )

        task = AgentTask.from_scenario(scenario)
        agent_run = runner.run(task, trace_recorder=trace_recorder)
        runtime_snapshot = environment.snapshot()
        trace = trace_recorder.snapshot()
        evaluation = DeterministicEvaluationEngine().evaluate(
            scenario,
            agent_run,
            runtime_snapshot,
            groundedness_judge=groundedness_judge,
        )
        cost_estimate = CostEstimator().estimate(trace, pricing) if pricing is not None else None
        return ObservableRun(
            agent_run=agent_run,
            runtime_snapshot=runtime_snapshot,
            trace=trace,
            evaluation=evaluation,
            cost_estimate=cost_estimate,
        )
