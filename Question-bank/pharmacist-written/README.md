# Pharmacist-written pairs

The 12 hard sample question-answer pairs written by three licensed pharmacists experienced in
interpreting CPIC guidelines. They wrote them from the CPIC guidelines and drug information,
following the principles for a question-answer pair they defined with us: the elements a question
must state, the facets an answer must cover and the answer format.

These pairs are the model for the AI-assisted questions in `../ai-assisted`, which were generated
to match them in what an answer must cover, its format and its easy-to-digest style.

## Format

`questions.jsonl` holds one pair per line:

| Field | Content |
|---|---|
| `pair_id` | 1 to 12 |
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
