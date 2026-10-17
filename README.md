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
