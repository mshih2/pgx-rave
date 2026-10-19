# PGx-RAVE atomic-claim verifier

Code for the Verify step of PGx-RAVE. It scores an answer claim by claim against long-form ground
truth (GT) and returns precision (P), recall (R) and F1.5, together with the label and credit of
every claim. The labels show where an answer loses facts: GT claims it misses, claims it gets
wrong, and correct claims outside the GT.

The verifier is a research tool, not a clinical one; its judges are LLMs.

## How an answer is scored

| Step | Model | What it does | Prompt files |
|---|---|---|---|
| 1. Decomposition | Gemma-4-31B-it | Splits the answer and the GT into single-fact claims in two passes over the same text (recommendation facts, context facts); one deduplication call then keeps every distinct fact once, copied verbatim | `decompose_recommendation.txt`, `decompose_context.txt`, `dedup.txt` |
| 2. Labelling judge | Grok 4.5 | One call per answer labels each answer claim (forward) and each GT claim (backward), then corrects its own inconsistencies | `label_judge.txt` |
| 3. Grading judge | Mistral-Small-24B-Instruct-2501 | Re-reads each positively labelled claim and turns its label into graded credit | `grade_forward.txt`, `grade_backward.txt` |
| 4. Score | (code) | P, R and F1.5 from the credits | (none) |

**1. Decomposition.** Answer and GT are split under the same definition: a claim is a single,
self-contained factual statement that can be checked against the reference on its own, such as one
recommendation for one phenotype or one dosing caveat. A reasoning block (`<think>…</think>`) at
the start of an answer is removed first.

**2. Labelling judge.** The judge sees all answer and GT claims. It labels each answer claim
`addressed` (matches a GT claim), `partially_addressed`, `wrong` or `extra` (judged correct but
outside the GT), with the GT claims it matches, and each GT claim `covered`, `partial` or
`missing`. It then lists the claims it labelled inconsistently across the two directions and
corrects them. Gemma-4-31B-it, tried as the labelling judge, was less accurate.

**3. Grading judge.** Each claim labelled `addressed`, `partially_addressed`, `covered` or
`partial` is re-read, one call per claim, on a three-grade scale. The credit is read from the
log-probabilities of the first grade token (top 15), not sampled: the probabilities of the full,
partial and wrong grades are renormalised over these three, and

c = 1 · p(full) + 0.5 · p(partial) + 0 · p(wrong).

Claims labelled `extra`, `wrong` or `missing` are not sent and get no credit. A claim whose
log-probabilities cannot be read keeps the labelling judge's weight (1.0 for a full match, 0.5 for
a partial one).

**4. Score.**

- P = forward credit / (n + 0.1 · n_extra), where n counts the answer claims not labelled
  `extra`. Each extra claim thus counts as 0.1 of a false positive (`EXTRA_WEIGHT`), a small
  weight that discourages padding without treating correct extra content as an error.
- R = backward credit / number of GT claims.
- F1.5 = 3.25 · P · R / (2.25 · P + R), which weights recall 1.5 times as much as precision:
  almost all unmatched answer claims are extra rather than wrong, so precision mostly penalises
  verbosity, whereas each missed GT claim is missing information.

`n_extra` is counted from the labelling judge's forward labels as it returned them, before its
consistency corrections. An answer with no countable claims gets P = 1 (vacuous); with no GT
claims, R = 0.

## Models

| Model | Used for | Setting |
|---|---|---|
| Gemma-4-31B-it (`gemma-4-31b-it`) on an OpenAI-compatible vLLM server | decomposition and deduplication, temperature 0 | `DECOMPOSER_URL` (default `http://127.0.0.1:8006/v1`) |
| Grok 4.5 (`grok-4.5`) through the xAI API | labelling judge, temperature 0, no output cap | `XAI_API_KEY` (required), `LABEL_MODEL` |
| Mistral-Small-24B-Instruct-2501 (`mistral-small-24b`) on an OpenAI-compatible vLLM server | grading judge, log-probabilities (top 15) | `GRADER_URL` (default `http://127.0.0.1:8012/v1`) |

Server launch settings (`vllm serve <model> <flags>`):

| Model | vLLM | Flags |
|---|---|---|
| Gemma-4-31B-it | 0.19.1 | `--served-model-name gemma-4-31b-it --port 8006 --dtype bfloat16 --max-model-len 32768` |
| Mistral-Small-24B-Instruct-2501 | 0.25.1 | `--served-model-name mistral-small-24b --port 8012 --max-model-len 32768 --max-logprobs 20` |

`--max-logprobs` must be at least 15, the number of top log-probabilities the grading judge reads.
One Gemma server can serve both the decomposer and PGxRoute's critique step (both default to port
8006).

The decomposer server should expose vLLM's `/tokenize` route, which is used to fit each request
into the context; without it, requests run unclamped. Other defaults are in
`pgxverifier/config.py`.

## Running

```bash
pip install -r requirements.txt
export XAI_API_KEY=...
python run.py --input items.jsonl --out results/
```

`items.jsonl` holds one item per line:

```json
{"question_id": "q1", "question": "...", "answer": "...", "gt_claims": ["...", "..."]}
```

To score against the bank, take `question` and `gt_claims` from
`../Question-bank/ai-assisted/questions.jsonl`. Alternatively, give `"ground_truth": "<text>"`
instead of `gt_claims` to have the GT decomposed with the same two passes first. The bank's GT
claims were decomposed once this way, then curated, and are passed as `gt_claims`.

From Python, with `Verifier/` on the path:

```python
from pgxverifier.verifier import verify
result = verify(question, answer, gt_claims=gt_claims)
result["scores"]["F1.5"]
```

Each result (`results/<question_id>.json`) records:

- `scores`: precision, recall, F1.5 and claim counts;
- `answer_claims` and `gt_claims`;
- `labels`: the labelling judge's raw output and its corrected labels;
- `credits`: the graded credit of each positively labelled claim;
- the decomposition record: both passes, the merges, warnings and prompt hashes.

`results/scores.csv` collects the scores of every result. Results are written per item and
existing ones are skipped, so an interrupted run resumes.

## Files

```
run.py                     command-line runner
requirements.txt
pgxverifier/               Python package
  config.py                endpoints, models and settings
  verifier.py              Verifier.verify(question, answer, gt_claims=... | ground_truth=...)
  jsonparse.py             reading JSON out of model replies
  steps/
    decompose.py           two passes + deduplication
    label.py               labelling judge (xAI Responses API client and the joint call)
    grade.py               grading judge (log-probability credit)
    score.py               P, R, F1.5
prompts/                   the six prompts
```

## Licence

Code and prompts: Apache License 2.0 (`LICENSE`). The models keep their own licences and terms.
