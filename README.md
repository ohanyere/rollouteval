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

## External local-command agents

An external agent can implement the version-1 JSON protocol without importing
RolloutEval. RolloutEval starts its configured argv directly (never through a
shell), sends one JSON request on stdin for each agent turn, and expects one
JSON response on stdout. The response wraps the existing agent output shape:

```json
{"protocol_version":1,"response":{"output":{"kind":"final","decision":"promote","explanation":"Observed metrics remain within the rollout limits."}}}
```

The request includes `protocol_version`, `request_id`, `scenario_id`, `run_id`,
the agent-visible task text, and prior tool interactions. It excludes expected
answers and evaluator reference truth. Tool calls use the same `kind`, `tool`,
and `arguments` fields as the in-process agent contract. Each turn starts a
fresh command process; RolloutEval's existing runner handles tool dispatch and
subsequent turns.

```python
import sys

from rollouteval.agent.external import ExternalAgentSpec, build_external_agent
from rollouteval.benchmark import (
    BenchmarkArtifactStore,
    BenchmarkReportService,
    BenchmarkRunner,
    BenchmarkSpec,
)
from rollouteval.run_artifacts import RunArtifactStore
from rollouteval.scenarios import load_scenario

agent = build_external_agent(ExternalAgentSpec(
    agent_id="my-rollout-agent",
    version="1.0.0",
    implementation_sha256="0123456789abcdef" * 4,
    command=(sys.executable, "./my_agent.py"),
    working_directory=".",
    timeout_seconds=20,
))
benchmark = BenchmarkSpec(
    benchmark_id="canary-core",
    version="1",
    tasks=(load_scenario("scenarios/healthy-canary.yaml"),),
)
run_store = RunArtifactStore("./run-artifacts")
benchmark_store = BenchmarkArtifactStore("./benchmark-artifacts")
result = BenchmarkRunner(run_store, benchmark_store).run(agent, benchmark)
report = BenchmarkReportService(benchmark_store, run_store).get(
    result.benchmark_run_id
)
```

Only variables explicitly listed in `ExternalAgentSpec.environment` are passed
to the child; the parent environment is not inherited. Configure a digest for
the complete agent implementation (not merely the Python launcher). Connection
commands and environment values are not part of the persisted agent identity.
Local command agents are **not a security sandbox**: they can access resources
permitted to the operating-system user, including filesystem and network access.
Use only trusted local code until a separate sandboxed execution boundary is
available.

## Remote HTTP agents

A remote service can expose the same version-1 request/response protocol. The
HTTP adapter uses Python's standard library; it POSTs JSON, requires a JSON
content type, applies a timeout and response limit, and does not follow
redirects. Endpoint URLs must use HTTP or HTTPS and cannot contain embedded
credentials. Bearer authentication is provided separately at runtime:

```python
import os

from rollouteval.agent.remote import RemoteAgentSpec, build_remote_agent
from rollouteval.benchmark import (
    BenchmarkArtifactStore,
    BenchmarkReportService,
    BenchmarkRunner,
    BenchmarkSpec,
)
from rollouteval.run_artifacts import RunArtifactStore
from rollouteval.scenarios import load_scenario

agent = build_remote_agent(
    RemoteAgentSpec(
        agent_id="hosted-rollout-agent",
        version="2.1",
        implementation_sha256="fedcba9876543210" * 4,
        endpoint_url="https://agent.example.test/rollouteval/v1/respond",
    ),
    bearer_token=os.environ["REMOTE_AGENT_TOKEN"],
)
benchmark = BenchmarkSpec(
    benchmark_id="canary-core",
    version="1",
    tasks=(load_scenario("scenarios/healthy-canary.yaml"),),
)
run_store = RunArtifactStore("./run-artifacts")
benchmark_store = BenchmarkArtifactStore("./benchmark-artifacts")
result = BenchmarkRunner(run_store, benchmark_store).run(agent, benchmark)
report = BenchmarkReportService(benchmark_store, run_store).get(
    result.benchmark_run_id
)
```

The bearer token remains runtime-only and is not part of agent identity or
artifacts. Endpoint URLs are also omitted from persisted provenance to avoid
retaining sensitive query parameters. Redirects are rejected rather than
followed, preventing credential forwarding to a redirect target. Remote HTTP
connectivity is **not a security sandbox or trust boundary**; response data is
validated as untrusted structured input, but Stage 7C does not provide network
isolation or comprehensive SSRF protection.
