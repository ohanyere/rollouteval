# Stage 6F Live Groundedness Acceptance

**Disposition: STAGE 6F LIVE GROUNDEDNESS ACCEPTANCE: PASSED**

This record closes the live Stage 6F.2 calibration. The live run used provider
Groq and model `openai/gpt-oss-120b`. The run reported 15 cases, 14 exact
matches, one mismatch, and exact agreement of 0.933333 (93.33%). No evaluator,
prompt, threshold, provider/model, evidence, or groundedness behavior was tuned
as part of this closure.

## Outcome and metrics

| Human reference | Judge PASS | Judge FAIL | Judge INCONCLUSIVE | Total |
| --- | ---: | ---: | ---: | ---: |
| PASS | 7 | 0 | 0 | 7 |
| FAIL | 0 | 7 | 0 | 7 |
| INCONCLUSIVE | 1 | 0 | 0 | 1 |
| **Total** | **8** | **7** | **0** | **15** |

- Exact agreements: 14/15 (93.33%); disagreement: 1/15.
- PASS: precision 7/8 (87.5%); recall 7/7 (100%).
- FAIL: precision 7/7 (100%); recall 7/7 (100%).
- INCONCLUSIVE: recall 0/1; precision is not meaningful because the judge
  produced no INCONCLUSIVE prediction.

## Case 13 closure

Case 13 (`evidence-evolved-to-healthy`) retains the human reference PASS and
the live judge returned PASS. The revised evidence-specific wording is now
correctly calibrated. No further Case 13 cleanup is warranted.

## Case 15 known disagreement

Case 15 (`ambiguous-meaningful-instability`) retains the human reference
INCONCLUSIVE; the live judge returned PASS. It is the sole disagreement and is
classified as **genuine semantic ambiguity**. The judge treated the observed
intermittent readiness changes as sufficient evidence of rollout instability.
The human reviewer retained INCONCLUSIVE because that observation is not
decisive enough for an affirmative grounded conclusion. This is not evidence
that the human reference should change, and the evaluator is not tuned to
eliminate this disagreement.

## Provenance and integrity

The live run's outcome and provider/model are recorded from the supplied
Stage 6F.2 run result. The supplied result did not include an execution log or
run timestamp, so neither is inferred here.

At closure, the working tree contains the intentional Stage 6F.1 Case 13
wording/review update. Dataset comparison confirms the Case 13 explanation is
the sole changed case payload; all other cases, including Case 15, match their
pre-cleanup payloads. The Case 15 semantic payload matches the frozen hash
asserted by `tests/test_groundedness_review.py`. This verifies unchanged
calibration inputs across the live run as far as the available artifacts allow;
the live execution log itself was not supplied for independent audit.

- Evaluator/source commit: `13c9192d06fa1ec3592a3c5c3e3632b04d14a71a`
  (`feat: add Groq provider support`); working tree evaluator source is
  unchanged from this commit.
- Prompt SHA-256 (`GROUNDEDNESS_INSTRUCTIONS` UTF-8 bytes):
  `7cfc46f94bdc5bd44454d71c2720f2c03eaf5baa1fb18bb8cdc75a0774428673`.
- Current dataset file SHA-256 (includes the intentional Case 13 wording
  cleanup):
  `256762a6fb7f4c00101108c583faa8884614b8edcba5cac18aa3a737134c5007`.
- Approved review YAML SHA-256 (includes the Case 13 reapproval):
  `d68045889bd2950beed256bd63c01735c50591d22f310d4dcbe250b95406123f`.
- Case 15 raw YAML block SHA-256:
  `d17ec93da165fbc0137c780e3533f5fe77e3417861cef7533a9d2c80d3387db3`.
- Case 15 canonical semantic payload SHA-256:
  `07daf171da791a670fe6cbfdd1bf348ddd60df0c18f3d7b14610854cdc396338`.

## Remaining limitation

The live judge may resolve genuinely ambiguous evidence toward PASS rather than
INCONCLUSIVE, as demonstrated by Case 15. Retain this as calibration evidence
and a documented limitation; do not silently correct it or initiate another
calibration/tuning cycle as part of this acceptance closure.
