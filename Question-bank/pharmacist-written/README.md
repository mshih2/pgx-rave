# Pharmacist-written pairs

The 12 hard question-answer pairs of the PGx question bank written by pharmacists (question IDs
0–11). Ten are in the evaluation set and two in the tuning set.

Three licensed pharmacists experienced in interpreting CPIC guidelines took part as research
collaborators. They defined with us the principles for a question-answer pair (the elements a
question must state, the facets an answer must cover and the answer format) and wrote both the
questions and the answers of these 12 pairs themselves, from the CPIC guidelines and drug
information. They were informed of the planned data release and consented.

The pharmacists' answers are the ground truth (GT) of these questions. For scoring, each answer
was decomposed into claims and the claims curated like those of the other questions
(`../../Question-generation`).

These pairs are also the model for the AI-assisted questions in `../ai-assisted`, which were
generated to match them in what an answer must cover, its format and its easy-to-digest style.

## Format

`questions.jsonl` holds one pair per line:

| Field | Content |
|---|---|
| `question_id` | 0 to 11 (the AI-assisted questions are numbered from 12) |
| `tier` | `hard` |
| `question` | the question text |
| `answer` | the pharmacists' answer |

Each answer follows the same format: step-by-step reasoning, the key data points and their
sources, and a final answer.

The file can be passed directly to `python run.py import-examples` in `../../Question-generation`,
which uses the pairs as the example pairs for generating questions.

## Use

The questions describe invented patients; no patient data were used. The pairs are for research,
not clinical decisions.

## Licence

The question-answer pairs are under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, `LICENSE`). Text quoted from other sources remains under its owners' terms.
