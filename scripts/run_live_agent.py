"""Run one manually selected live agent episode against a golden scenario."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rollouteval.agent import AgentRunner, AgentRunResult, AgentTask
from rollouteval.agent.providers import GroqResponsesModel, OpenAIResponsesModel
from rollouteval.domain.scenario import Scenario
from rollouteval.environment import RolloutEnvironment
from rollouteval.scenarios import load_scenario
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools
from rollouteval.provider_client import configured_api_provider


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
SUPPORTED_SCENARIO = "degraded-canary"


def build_live_run(
    scenario_id: str,
    model_id: str,
    *,
    client: Any | None = None,
    provider: str,
) -> tuple[Scenario, RolloutEnvironment, AgentTask, AgentRunner, RolloutObservationTools]:
    """Construct the existing environment and runner for degraded-canary only."""
    if scenario_id != SUPPORTED_SCENARIO:
        raise ValueError(f"only {SUPPORTED_SCENARIO!r} is supported for this live run")

    scenario = load_scenario(SCENARIO_DIR / f"{scenario_id}.yaml")
    environment = RolloutEnvironment(scenario)
    observation_tools = RolloutObservationTools(environment)
    mutation_tools = RolloutMutationTools(environment)
    task = AgentTask.from_scenario(scenario)
    model_type = GroqResponsesModel if provider == "groq" else OpenAIResponsesModel
    model = model_type(model=model_id, client=client)
    runner = AgentRunner(model, observation_tools, mutation_tools)
    return scenario, environment, task, runner, observation_tools


def safe_run_summary(
    scenario_id: str,
    model_id: str,
    task: AgentTask,
    result: AgentRunResult,
    observation_tools: RolloutObservationTools,
    environment: RolloutEnvironment,
) -> dict[str, object]:
    """Build a printable result containing runtime-visible data only."""
    interactions = []
    for interaction in result.interactions:
        interactions.append(
            {
                "tool": interaction.call.tool.value,
                "status": "error" if interaction.error is not None else "success",
                "result": (
                    json.loads(interaction.result) if interaction.result is not None else None
                ),
                "error": interaction.error,
            }
        )

    rollout_state = observation_tools.get_rollout_state()
    return {
        "scenario_id": scenario_id,
        "model_id": model_id,
        "task": task.task,
        "steps": result.steps,
        "tool_interactions": interactions,
        "final_answer": {
            "decision": result.final_answer.decision.value,
            "explanation": result.final_answer.explanation,
        },
        "final_rollout_state": {
            "phase": rollout_state.phase,
            "candidate_weight": rollout_state.candidate_weight,
        },
        "mutation_count": environment.mutation_count,
    }


def execute_live_run(
    scenario_id: str,
    model_id: str,
    *,
    client: Any | None = None,
    provider: str,
) -> dict[str, object]:
    """Execute one runner episode and return its safe inspectable summary."""
    _, environment, task, runner, observation_tools = build_live_run(
        scenario_id, model_id, client=client, provider=provider
    )
    result = runner.run(task)
    return safe_run_summary(
        scenario_id, model_id, task, result, observation_tools, environment
    )


def _safe_error_message(error: Exception) -> str:
    """Avoid printing recognizable API key strings from provider errors."""
    return re.sub(r"\b(?:sk-|gsk_)[A-Za-z0-9_-]{12,}\b", "[REDACTED]", str(error))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=(SUPPORTED_SCENARIO,), default=SUPPORTED_SCENARIO
    )
    parser.add_argument("--provider", choices=("groq", "openai"), required=True)
    parser.add_argument("--model", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if configured_api_provider(args.provider) is None:
        provider_name = f"{args.provider.upper()}_API_KEY"
        print(
            f"LIVE RUN BLOCKED: {provider_name} is not configured.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = execute_live_run(args.scenario, args.model, provider=args.provider)
    except Exception as exc:
        print(
            f"{type(exc).__name__}: {_safe_error_message(exc)}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
