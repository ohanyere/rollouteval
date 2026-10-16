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
