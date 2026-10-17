# AI-assisted questions

Question-answer pairs from the PGx question bank that were drafted with AI assistance and then
curated: 153 English questions. Each one comes with a long-form ground-truth (GT) answer and the GT
claims the Verifier scores answers against.

| Tier | Questions | Evaluation | Tuning | Evidence an answer needs |
|---|---|---|---|---|
| hard | 43 | 35 | 8 | CPIC guideline or supplement text beyond the CPIC API's core recommendation |
| intermediate | 55 | 45 | 10 | one CPIC API recommendation lookup |
| common | 55 | 45 | 10 | none from CPIC: everyday medication questions with no PGx content |

## How they were made

Three licensed pharmacists experienced in interpreting CPIC guidelines defined with us the
principles for a question-answer pair: the elements a question must state, the facets an answer
must cover and the answer format. They also wrote hard sample pairs from the CPIC guidelines and
drug information. Those pharmacist-written pairs are not in this folder.

The questions here were generated with AI assistance:

- **Hard and intermediate questions** were modelled on the pharmacist-written pairs in what an
  answer must cover, its format and its easy-to-digest style. Their GT is built only from the
  tier's CPIC source: guideline and supplement text for hard questions, the CPIC API lookup for
  intermediate ones.
- **Common questions** were also generated with AI assistance.

LLMs cross-reviewed the hard and intermediate questions and their GT against their sources. The
non-pharmacist authors then checked every item word by word. Each GT was decomposed into
single-fact claims, and the claims were curated against the GT and its sources.

The evaluation questions are the ones the pipelines, routers and frontier baselines are compared
on. The tuning questions were used to tune the pipeline prompts and to test basic routing.

## Format

`questions.jsonl` holds one question per line:

| Field | Content |
|---|---|
| `question_id` | the question's number in the full bank |
| `tier` | `hard`, `intermediate` or `common` |
| `split` | `evaluation` or `tuning` |
| `question` | the question text |
| `ground_truth` | the GT answer |
| `gt_claims` | the curated GT claims |

To score an answer, pass `question` and `gt_claims` to the Verifier (`../../Verifier`).
`../../Router-training/scripts/build_data.py` reads this file as a bank. Rows with no `fold`
field go to `all.jsonl`.

## Use

The questions describe invented patients; no patient data were used. The pairs are for research,
not clinical decisions.

## Licence

The question-answer pairs are under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, `LICENSE`). Text quoted from the CPIC API is CC0 1.0. Passages quoted from CPIC
guideline articles and supplements remain under their publishers' terms.
