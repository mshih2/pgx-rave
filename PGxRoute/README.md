# PGxRoute: PGx-RAVE routers and answer pipelines

Code for the routers and the three answer pipelines of PGxRoute, the route-and-answer part of
PGx-RAVE. Each question is answered by one pipeline, chosen by a router.

PGx-RAVE is a research prototype, not a clinical tool. Its answers can be wrong; any clinical use
needs verification by a licensed pharmacist or clinician.

| Pipeline | Tier | Steps | Prompt files |
|---|---|---|---|
| `reasoning_only` | common | reasoner only (general-medicine prompt; no CPIC, no retrieval) | `reasoning_only_system.txt` |
| `api_only` | intermediate | structured input (SI) → CPIC API → reasoner | `api_only_system.txt` |
| `full` | hard | SI → CPIC API → retrieval (BGE-M3 pool of 60, listwise reranker, top 8) → reasoner → critique → CPIC citation | `full_reasoner_system.txt`, `rerank.txt`, `full_critique.txt` |

## Structured input (SI)

`pgxroute.steps.si_builder.build(question)` reads the question against `data/cpic_lookup.json` and returns the
genes, genotypes and drugs it names, the SI text used as the retrieval query, and the
`{gene: {allele: count}}` argument for the CPIC recommendation lookup. A gene, allele or drug is
recognised only where CPIC prints that name, and questions are assumed to name genotypes
correctly. The module docstring states the reading rules.

The table is built from the CPIC API, so its aliases are CPIC's own: a parenthesised star allele
(`c.1905+1G>A (*2A)`), a synonym list, an HLA serotype without its colon, a variant position name
or dbSNP id, and an unspecified copy number. It also records each gene's reference allele,
chromosome and phenotype names, the gene groups CPIC keys together, the genes each drug's
recommendation is keyed on, and each drug's guideline.

All pipelines send `answer_format_user.txt` as the user turn of the reasoner call.

## Routers

| Router | Folder | Reads | Picks by |
|---|---|---|---|
| `rule` | `pgxroute/routers/rule.py` | the SI | three rules: no gene and no variant → Reasoning-only; one gene, ≥1 variant, one drug, no condition → API-only; else Full |
| `logreg`, `xgboost` | `pgxroute/routers/ml/` | question text | TF-IDF (word 1–2-grams) + logistic regression or XGBoost |
| `fg`, `sf` | `pgxroute/routers/rl/` | question text | Qwen3-8B with the facet-guided (`fg`) or self-faceted (`sf`) prompt, untrained or GRPO-trained |

```python
from pgxroute.router import get_router
pipeline = get_router("fg").route(question)    # "reasoning_only", "api_only" or "full"
```

```bash
# from the repository root
python -m pgxroute.router --router rule --question "A 52-year-old man ..."
python run.py --router xgboost --questions questions.jsonl --out answers/   # route, then answer
```

- **Rule.** The SI is built by lookup and has no conditions, so `route` applies the rules
  without conditions; pass `conditions` to `routers.rule.pick` if you have them.
- **Text classifiers.** No trained model is included. Train and save one with
  `python -m pgxroute.routers.ml.text_classifier --kind xgboost --data bank.jsonl --out data/routers/xgboost.pkl`
  (`bank.jsonl`: `question` plus `pipeline` or `tier` per line); the router loads
  `data/routers/<name>.pkl` (`CLASSIFIER_DIR`). `two_fold_predict` gives out-of-fold picks.
- **LLM routers.** Serve Qwen3-8B (untrained) or a trained checkpoint with an OpenAI-compatible
  server and set `ROUTER_BASE_URL` / `ROUTER_MODEL`. Thinking on, temperature 0.6, top-p 0.95,
  2,048 tokens; an unparsable reply is resampled twice, then the question goes to
  Reasoning-only. The trained checkpoints and the RL training code are not included.

## Files

```
run.py                     command-line runner
requirements.txt
pgxroute/                  Python package
  config.py                endpoints, models, decoding settings, paths
  pipelines.py             answer(question, pipeline)
  router.py                get_router(name), route(question, name)
  steps/                   the steps the pipelines chain
    si_builder.py  cpic_api.py  retrieval.py  reasoner.py  critique.py  citation.py
  routers/
    rule.py                rule baseline
    ml/text_classifier.py  TF-IDF + logistic regression / XGBoost
    rl/llm_router.py       Qwen3-8B router, FG and SF prompts
prompts/                   the six pipeline prompts; routers/ holds the router prompts
data/                      CPIC lookup table; corpus, embedding cache and trained classifiers go here
tools/                     offline bank-preparation tools (never run at answer time)
```

| File | Role |
|---|---|
| `pgxroute/pipelines.py` | `answer(question, pipeline)`: runs one pipeline and returns the answer with its intermediate outputs, tokens and timings |
| `pgxroute/config.py` | Endpoints, models, decoding settings and paths (all overridable by environment variables; paths default to folders under the repository root) |
| `pgxroute/steps/si_builder.py` | Builds the SI of a question by lookup against the CPIC table: its genes, genotypes and drugs, and the CPIC lookup argument |
| `pgxroute/steps/cpic_api.py` | Converts the diplotypes to CPIC lookup keys, calls `/rpc/recommendation_lookup`, formats the rows for the SI's drugs |
| `pgxroute/steps/retrieval.py` | Corpus check, gene/drug scope, BGE-M3 hybrid ranking, listwise reranking |
| `pgxroute/steps/reasoner.py` | Reasoner request and answer cleaning (the reasoning block is removed before delivery) |
| `pgxroute/steps/critique.py` | Critique of the Full pipeline's draft |
| `pgxroute/router.py` | `get_router(name)` and `route(question, name)` for the five routers; also a command-line entry |
| `pgxroute/routers/rule.py` | Rule baseline over the SI's genes, variants and drugs |
| `pgxroute/routers/ml/text_classifier.py` | TF-IDF text classifiers: train, save, load, route, out-of-fold picks |
| `pgxroute/routers/rl/llm_router.py` | LLM router: builds the FG or SF prompt, samples, parses the JSON after `</think>`, resamples |
| `pgxroute/steps/citation.py` | Strips model-written citation identifiers and appends the CPIC guideline link of the question's first drug, read from the table |
| `run.py` | Command-line runner |
| `prompts/` | The six pipeline prompts, and `prompts/routers/` with the FG and SF router templates and `tool_lines.json`, the tool descriptions both templates fill in at `{tool_list}` |
| `data/cpic_lookup.json` | The CPIC lookup table: every gene, allele, drug and phenotype name of the CPIC API, with the aliases CPIC prints for them, each gene's reference allele and chromosome, the gene groups CPIC keys together, and each drug's guideline |
| `tools/audit_question_names.py` | Offline check of a question bank against the table (reads ground truth; never used at answer time) |

## Models

Two OpenAI-compatible vLLM servers are expected:

| Model | Served name | Used for | Setting |
|---|---|---|---|
| Qwen3.8-27B | `qwen3.8-27b` | reasoner (all pipelines), reranker (Full) | `REASONER_BASE_URL` (default `http://127.0.0.1:8019/v1`) |
| Gemma-4-31B-it | `gemma-4-31b-it` | critique (Full) | `CRITIC_URL` (default `http://127.0.0.1:8006/v1/chat/completions`) |

BGE-M3 (`BAAI/bge-m3`) runs in process through FlagEmbedding on `EMBED_DEVICE`. If FlagEmbedding
needs a different `transformers` version from the main environment, install it in a separate
environment and set `EMBED_PYTHON` to that interpreter; ranking then runs as a subprocess
(`python -m pgxroute.steps.retrieval --rank`, started from the repository root). That
environment needs only FlagEmbedding and its dependencies.

Decoding settings (defaults in `config.py`):

- **Reasoner:** temperature 0.05, top-p 0.95, thinking on with reasoning effort medium, output cap 24,000 tokens. A truncated completion raises and is not recorded; an empty completion is retried up to 3 times.
- **Reranker:** temperature 0, thinking off, output cap 4,000 tokens (raised to 8,000 and 16,000 only if the ranking is truncated). Passages are shortened to fit a 40,960-token window.
- **Critique:** temperature 0.05, output cap 2,048 tokens, 3 attempts. If all fail, the draft is kept.
- **Inputs passed on:** CPIC output is cut at 8,000 characters and retrieved context at 32,000. An empty section is replaced by an explicit "EMPTY" notice.

## Data

```
data/cpic_lookup.json            the CPIC lookup table (included)
data/corpus/                     retrieval corpus (not included; see below)
data/emb_cache/                  BGE-M3 unit embeddings (created on first use)
```

**Corpus.** The retrieval corpus is built from CPIC guideline articles and supplements, whose
text is mostly under publisher copyright, so it is not included. Each document is one JSON file
with `doc_name`, `source`, `gene`, `drug`, `prose_chunks`, and `blocks` (tables and figures), each
unit holding a `self_contained_text`. `doc_manifest.json` lists the documents and marks
superseded ones with `superseded_by`; those are excluded at retrieval. The corpus is checked at
load against `_provenance.json` and `EXPECTED_CORPUS_VERSION`; set `VERIFY_CORPUS=0` to use a
corpus you built yourself, and `python -m pgxroute.steps.retrieval --verify` (from the repository
root) to check a copy.

## Running

```bash
pip install -r requirements.txt
python run.py --pipeline full --question "A 52-year-old man ..."
python run.py --pipeline api_only --questions questions.jsonl --out answers/
```

`run.py` can be started from any directory. From Python, with the repository root on the path:

```python
from pgxroute.pipelines import answer
result = answer("A 52-year-old man ...", pipeline="api_only")
```

Each output records:
- `answer`: the final text.
- The intermediate outputs: `structured_input` (the SI text), `genes`, `variants`, `drugs`, `cpic_result`, `retrieval`, `draft`, `critiqued` and `citation`.
- `raw_completion`: the reasoner's full output, including its reasoning block.
- `usage_by_step` and `usage`: input and output tokens of the reasoner, reranker and critique calls. Building the SI is a table lookup and costs no tokens.
- `timing`: seconds per step.

## Licence

Code and prompts: Apache License 2.0 (`LICENSE`). The CPIC API content the pipelines retrieve is
CC0 1.0. The models keep their own licences.
