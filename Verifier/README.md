# PGx-RAVE atomic-claim verifier

Code for the Verify step of PGx-RAVE: it scores an answer claim by claim
against long-form ground truth (GT) and returns precision (P), recall (R), F1.5 and the label and
credit of every claim.

| Step | Model | What it does | Prompt files |
|---|---|---|---|
| Decomposition | Gemma-4-31B-it | Splits a text into single-fact claims in two passes (recommendation facts, context facts) over the same text, then one deduplication call keeps every distinct fact once, copied verbatim | `decompose_recommendation.txt`, `decompose_context.txt`, `dedup.txt` |
| Labelling judge | Grok 4.5 | One call per answer labels each answer claim (forward) addressed / partially_addressed / wrong / extra, and each GT claim (backward) covered / partial / missing, then corrects its own inconsistencies | `label_judge.txt` |
| Grading judge | Mistral-Small-24B | Re-reads each positively labelled claim, one call per claim, and turns the label into expected credit from the grade-token log-probabilities: c = p(full) + 0.5 · p(partial) | `grade_forward.txt`, `grade_backward.txt` |
| Score | — | P = forward credit / (n − 0.9 · n_extra), so an extra claim counts 0.1 of a false positive; R = backward credit / GT claims; F1.5 = 3.25PR / (2.25P + R) | — |

The verifier is a research tool, not a clinical one; its judges are LLMs.

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

## Models

| Model | Used for | Setting |
|---|---|---|
| Gemma-4-31B-it (`gemma-4-31b-it`) on an OpenAI-compatible vLLM server | decomposition and deduplication, temperature 0 | `DECOMPOSER_URL` (default `http://127.0.0.1:8006/v1`) |
| Grok 4.5 (`grok-4.5`) through the xAI API | labelling judge, temperature 0, no output cap | `XAI_API_KEY` (required), `LABEL_MODEL` |
| Mistral-Small-24B (`mistral-small-24b`) on an OpenAI-compatible vLLM server | grading judge, log-probabilities (top 15) | `GRADER_URL` (default `http://127.0.0.1:8012/v1`) |

The decomposer server should expose vLLM's `/tokenize` route, which is used to fit each request
into the context; without it requests run unclamped. Other defaults are in
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

Use `"ground_truth": "<text>"` instead of `gt_claims` to have the GT decomposed with the same two
passes first; GT claims can also be decomposed once, curated, and then passed as `gt_claims`. A
reasoning block (`<think>…</think>`) at the
start of an answer is removed before decomposition.

From Python, with the repository root on the path:

```python
from pgxverifier.verifier import verify
result = verify(question, answer, gt_claims=gt_claims)
result["scores"]["F1.5"]
```

Each result (`results/<question_id>.json`) records `scores` (precision, recall, F1.5 and claim
counts), `answer_claims`, `gt_claims`, `labels` (the labelling judge's raw output and its
corrected labels), `credits` (graded credit per positively labelled claim) and the
decomposition record (both passes, the merges, warnings and prompt hashes).
`results/scores.csv` collects the scores of every result. Results are written per item and
existing ones are skipped, so an interrupted run resumes.

## Licence

Code and prompts: Apache License 2.0 (`LICENSE`). The models keep their own licences and terms.
