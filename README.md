# RolloutEval

RolloutEval evaluates an autonomous canary-rollout agent. The current milestone
is only a golden-case contract: load and validate
`scenarios/degraded-canary.yaml` as a typed Pydantic `Scenario`.

Not yet implemented: evaluating complete agent trajectories against
independent evidence rather than relying on the agent's final answer.

Run the tests from the repository root with:

```sh
python3 -m pytest
```

This milestone does not execute an agent or infer a decision from the evidence.
