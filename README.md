# RolloutEval

RolloutEval evaluates an autonomous canary-rollout agent in a simulated
environment and scores its run with deterministic graders.

Run the tests from the repository root with:

```sh
python3 -m pytest
```

Live agent and groundedness calibration scripts support explicit OpenAI or
Groq provider selection. Supply both `--provider openai|groq` and
`--model MODEL_ID`; Groq credentials are read from `GROQ_API_KEY`.

## Run artifacts

Completed `ObservableRun` instances can be persisted explicitly as versioned
JSON artifacts and read through the generic reporting service:

```python
from rollouteval.run_artifacts import RunArtifactStore, RunReportService

store = RunArtifactStore("./run-artifacts")
store.save(observable_run)
report = RunReportService(store).get(observable_run.trace.run_id)
```

The artifact root is caller-selected. Saving is atomic and idempotent for the
same run; reusing a run ID for different content raises a conflict. Reporting
shows criteria independently and does not derive an overall success verdict.
Discovery reads recognized `run-<identity-hash>.json` artifacts, ignores
unrelated filenames, and rejects malformed recognized artifacts.

## Benchmark runs

Benchmarks contain an explicitly ordered tuple of validated scenarios. The
digest covers the complete benchmark definition, including task order. Agent
versions require a stable ID, version, and implementation SHA-256; an executor
is supplied separately from that persisted identity.

Assuming `agent_model` is an existing provider-neutral `AgentModel` instance:

```python
from rollouteval.benchmark import (
    AgentDefinition,
    AgentIdentity,
    BenchmarkArtifactStore,
    BenchmarkReportService,
    BenchmarkRunner,
    BenchmarkSpec,
    ModelAgentExecutor,
)
from rollouteval.run_artifacts import RunArtifactStore
from rollouteval.scenarios import load_scenario

benchmark = BenchmarkSpec(
    benchmark_id="canary-core",
    version="1",
    tasks=(load_scenario("scenarios/healthy-canary.yaml"),),
)
agent = AgentDefinition(
    identity=AgentIdentity(
        agent_id="rollout-agent",
        version="2026.10",
        implementation_sha256="a" * 64,
    ),
    executor=ModelAgentExecutor(agent_model),
)
runner = BenchmarkRunner(
    RunArtifactStore("./run-artifacts"),
    BenchmarkArtifactStore("./benchmark-artifacts"),
)
result = runner.run(agent, benchmark, trials_per_task=1)
report = BenchmarkReportService(
    BenchmarkArtifactStore("./benchmark-artifacts"),
    RunArtifactStore("./run-artifacts"),
).get(result.benchmark_run_id)
```

Trials execute sequentially in benchmark order and independent trials continue
after an execution failure. Every completed trial has its own normal run
artifact; the benchmark artifact stores identities, references, and criterion
counts, not copied run payloads. Reports retain each criterion independently
and do not invent an aggregate pass/fail verdict.
