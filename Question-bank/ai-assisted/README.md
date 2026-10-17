# AI-assisted questions

This folder holds the 153 questions of the PGx question bank that were generated with AI
assistance and then curated: every question except the 12 pharmacist-written pairs (question IDs
0–11, in `../pharmacist-written`). Each question comes with its long-form ground-truth (GT) answer
and the GT claims the Verifier scores answers against.

## The question bank

The bank has 165 English questions, 55 per tier. Tiers are defined by where the evidence lives.
Thirty questions (10 per tier) were used to tune the pipeline prompts and test basic routing
methods. The remaining 135 (45 per tier) are the evaluation set; after decomposition, they hold
2,040 GT claims.

| Tier | Evidence an answer needs | Questions | Evaluation | Tuning |
|---|---|---|---|---|
| hard | CPIC guideline or supplement text beyond the CPIC API's core recommendation | 55 | 45 | 10 |
| intermediate | one CPIC API lookup | 55 | 45 | 10 |
| common | none from CPIC: everyday medication questions with no PGx content | 55 | 45 | 10 |

This folder holds all of them except the 12 pharmacist-written hard pairs (10 evaluation, 2
tuning):

| Tier | Questions here | Evaluation | Tuning |
|---|---|---|---|
| hard | 43 | 35 | 8 |
| intermediate | 55 | 45 | 10 |
| common | 55 | 45 | 10 |
| total | 153 | 125 | 28 |

## How they were made

We built the bank with three licensed pharmacists experienced in interpreting CPIC guidelines.
They defined with us the principles for a question-answer pair (the elements a question must
state, the facets an answer must cover and the answer format) and wrote 12 hard sample pairs from
the CPIC guidelines and drug information.

- **Hard and intermediate questions** were generated with AI assistance, modelled on these 12 in
  what an answer must cover, its format and its easy-to-digest style. Their GT is built only from
  the tier's CPIC source and verified against it; where CPIC is silent, the GT says so. Hard GT
  was written by Claude Fable 5.1 from minimal guideline and supplement passages selected by two
  LLMs (Claude Opus 4.8 and Fable 5.1). Intermediate GT is curated with AI from related CPIC API
  lookups and trimmed to the information relevant to the question.
- **Common questions** were also generated with AI assistance; their GT was reached through a
  discussion between Opus 4.8 and Fable 5.1.

LLMs cross-reviewed each new question, and the non-pharmacist authors then checked every item word
by word. Each GT was decomposed into single-fact claims, and the claims were curated against the GT
and its sources. No question or GT in this folder has yet been reviewed by a pharmacist.
`../../Question-generation` is the code for this process.

## Format

`questions.jsonl` holds one question per line:

| Field | Content |
|---|---|
| `question_id` | the question's number in the full bank (12 to 164) |
| `tier` | `hard`, `intermediate` or `common` |
| `split` | `evaluation` or `tuning` |
| `question` | the question text |
| `ground_truth` | the GT answer |
| `gt_claims` | the curated GT claims |

The routers' two-fold cross-validation split of the 135 evaluation questions is in
`../folds.txt` (one question ID and its fold, 0 or 1, per line).

To score an answer, pass `question` and `gt_claims` to the Verifier (`../../Verifier`).
`../../Router-training/scripts/build_data.py --folds ../folds.txt` reads this file as a bank and
writes the training rows of each fold to their own file; tuning questions, which have no fold, go
to `all.jsonl`.

## Use

The questions describe invented patients; no patient data were used. The pairs are for research,
not clinical decisions.

## Licence

The question-answer pairs are under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, `LICENSE`). Text quoted from the CPIC API is CC0 1.0. Passages quoted from CPIC
guideline articles and supplements remain under their publishers' terms.
