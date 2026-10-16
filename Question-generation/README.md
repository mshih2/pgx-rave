# Question and ground-truth generation

Code for building the PGx question bank: hard, intermediate and common questions, each with a
long-form ground-truth (GT) answer and the curated GT claims the Verifier scores against. Two
LLMs, Claude Opus 4.8 and Claude Fable 5.1, do the writing, selecting and reviewing; every step
that judges content runs them as independent readers whose findings are settled by rule; the
authors then check every item by hand.

## Tiers and their sources

| Tier | Answerable from | GT built from |
|---|---|---|
| hard | CPIC guideline or supplement text beyond the CPIC API's core recommendation | the question's CPIC lookup and minimal guideline/supplement passages |
| intermediate | one CPIC API recommendation lookup | the question's CPIC lookup, trimmed to what the question asks |
| common | general drug information (no PGx content) | a discussion between the two LLMs |

A question's CPIC lookup is what PGxRoute's API-only pipeline receives for it (`../PGxRoute`:
structured input, then the CPIC recommendation lookup), and its passages come from the same
retrieval corpus, document tags and supersession rule as the Full pipeline. So every fact in a
PGx GT is one the pipelines can be given.

## Method

**Inputs.** Twelve pharmacist-written example pairs (question and answer) are the model for what
a question states and asks, what an answer covers, its format and its easy-to-digest style. The
construction principles (`prompts/principles.md`) say what a question must state (gene,
genotype, drug, population, clinical context) and how an answer is grounded. The six clinical
question facets (`prompts/facets.json`) guide what a question asks.

**Hard questions.**
1. *Categorize information.* CPIC guideline and supplement text is sorted into the information
   categories of `prompts/categories.json` (Opus 4.8, one document at a time).
2. *Find sparse categories.* The bank's hard questions are tagged with the same categories; the
   categories with the fewest hard questions come first (among equals, the one with more text).
3. *Generate a question.* For a sparse category, a CPIC recommendation row is drawn whose gene and
   drug have current text in that category and whose (gene, drug) pair no hard question has taken.
   The question writer (Opus 4.8) sees the principles, the facets, the example pairs, the row and
   the category's passages, and must make the question ask for one point only the passages print
   (quoted verbatim and checked). A question whose signature (categories, facets, answer
   direction, population) repeats an existing hard question is rejected.
4. *Select GT passages, blind.* Opus 4.8 and Fable 5.1 each name the minimum passage set from all
   current passages of the documents covering the gene and drug, with byte-identical prompts and
   without seeing each other. Fable 5.1 converges the two by reading the passages, not counting
   votes, and writes notes for the GT writer that quote the passages they summarise.
5. *Write the GT.* Fable 5.1 writes the GT from the CPIC lookup and the final passages only,
   modelled on the example pairs.

**Intermediate questions.** Questions are written (Opus 4.8) from one CPIC recommendation row at a
time, spread across gene families, modelled on the example pairs and guided by the facets; a
question is kept only if its own lookup returns that row. The GT is curated by Fable 5.1 from the
question's CPIC lookup (all rows it returns), in CPIC's own words, and trimmed to what the
question asks; sentences that are not verbatim in the lookup are listed for the review.

**Common questions.** Everyday medication questions are written (Opus 4.8) per category
(`prompts/common_categories.json`); a question naming a gene is dropped. The GT is reached through
a discussion: Opus 4.8 drafts, Fable 5.1 agrees or asks for changes, Opus 4.8 accepts or rebuts
each and revises, until Fable 5.1 agrees (unresolved points go to the human check).

**Cross-review of each new question, then human curation.**
1. *Review the question and GT.* Opus 4.8 and Fable 5.1 review the question and GT against the
   material the GT was built from, independently. A settler (Opus 4.8) acts on a finding only if
   both readers made it, may still overrule it on reading the material, and may not order a
   repair the material cannot support. Fable 5.1 applies the settled changes and nothing else.
2. *Decompose into claims* with the Verifier's own two-pass decomposition and deduplication
   (`../Verifier`), so GT claims are made exactly as answer claims are.
3. *Review claim coverage.* Opus 4.8 and Fable 5.1 check, independently, that the claims cover the
   GT exactly; the settler keeps the findings both made.
4. *Curate the claims.* Fable 5.1 curates the claims against the GT and its sources under the
   written plan in `prompts/claim_curation.txt`, starting from the settled findings, for up to six
   rounds or until a round changes nothing.
5. *Human curation.* The authors check every question, GT and claim list word by word from a
   generated sheet.

Pharmacist pairs keep their GT; they go through claim decomposition, coverage review and curation.

**Tuning split.** Ten questions per tier are set aside for tuning: hard ones spread over
gene-variant categories (gene family and phenotype), intermediate ones one per gene family, common
ones two from each of five everyday-medication categories. The remaining questions are the
evaluation set, split into two folds per tier.

## Files

```
run.py                      command-line runner, one step at a time
requirements.txt
qgen/                       Python package
  config.py                 models, paths and limits
  llm.py                    Claude calls (streamed; JSON parsing; refusal handling)
  files.py                  work-directory layout, prompt rendering
  sources/
    cpic.py                 CPIC table snapshot, candidate rows, a question's CPIC lookup
    corpus.py               guideline and supplement passages (PGxRoute's corpus)
    categories.py           information categories, coverage, sparse categories, collisions
  hard/
    generate.py             questions for sparse categories
    select.py               bundle, two blind selections, converge
    write_gt.py             GT writer
  intermediate/
    generate.py             questions from one CPIC row
    gt.py                   GT curated from the CPIC lookup
  common/
    generate.py             everyday medication questions
    gt.py                   GT by discussion
  review.py                 two readers, settler, revision
  claims.py                 decomposition, coverage review, curation
  bank.py                   example import, check sheets, split, bank
prompts/                    every prompt, the principles, facets and categories
```

## Setup

```bash
pip install -r requirements.txt
pip install -r ../PGxRoute/requirements.txt ../Verifier/requirements.txt
```

| Needed for | Setting |
|---|---|
| Opus 4.8 and Fable 5.1 (Anthropic API) | `ANTHROPIC_API_KEY` (or an `ant auth login` profile) |
| a question's CPIC lookup | the CPIC API (`CPIC_BASE`, default `https://api.cpicpgx.org/v1`), via PGxRoute |
| hard-tier passages and categories | a retrieval corpus at `CORPUS_DIR` (see `../PGxRoute/README.md`) |
| claim decomposition | the Verifier's decomposer server, `DECOMPOSER_URL` (see `../Verifier/README.md`) |

Other settings (`qgen/config.py`): `QGEN_WORK_DIR` (default `work/`), `QGEN_OPUS_MODEL` (default
`claude-opus-4-8`), `QGEN_FABLE_MODEL` (default `claude-fable-5-1`), `QGEN_QUESTION_MODEL`
(default: the Opus model), `QGEN_EFFORT` (default `high`), `QGEN_WORKERS`, `QGEN_SEED`,
`QGEN_CURATION_ROUNDS` (default 6), `QGEN_REVIEW_ROUNDS` (default 2), `QGEN_COMMON_GT_ROUNDS`
(default 4).

Requests to Claude Fable 5.1 opt into the API's server-side refusal fallback, so a declined
request is answered by another model instead of failing; every output records the model that
served it (`served_by`), and `QGEN_FALLBACKS=0` turns the fallback off.

## Running

```bash
# sources, once
python run.py snapshot
python run.py candidates
python run.py categorize
python run.py import-examples examples.jsonl     # the pharmacist pairs: {"question", "answer"} per line
python run.py tag
python run.py coverage

# questions and GT, then the cross-review and claims
python run.py generate --tier hard --n 10
python run.py generate --tier intermediate --n 10
python run.py generate --tier common --n 2       # per category
python run.py pipeline --tier hard               # select, write GT, review, claims
python run.py pipeline --tier intermediate
python run.py pipeline --tier common

# human curation and the bank
python run.py sheet --tier hard                  # -> work/sheets/hard.md
python run.py mark-checked h001 --by <initials>
python run.py split
python run.py bank                               # -> work/bank.jsonl
```

Each step can also be run on its own (`select`, `write-gt`, `review`, `claims`) for given
question IDs or a whole tier, and `python run.py --workers 2 ...` sets the parallelism. Every
question's work is kept in `work/questions/<qid>/`: its question, CPIC lookup, passage bundle,
both selections, the final set, every GT version, both reviews and the settlement of each round,
the claims at each stage, and a log of which model ran each step.

`work/bank.jsonl` has one `{"question_id", "tier", "question", "ground_truth", "gt_claims",
"category", "source", "split", "fold"}` per line, the format `../Router-training/scripts/build_data.py`
and the Verifier read. Only questions marked as checked by hand are included unless
`--include-unchecked` is given.

## Licence

Apache License 2.0 (`LICENSE`).
