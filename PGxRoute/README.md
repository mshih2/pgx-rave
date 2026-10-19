# PGxRoute: routers and answer pipelines of PGx-RAVE

PGxRoute is the route-and-answer part of PGx-RAVE. A router reads each question and sends it to
one of three answer pipelines; each pipeline is matched to one question tier and gathers only the
evidence that tier needs.

PGx-RAVE is a research prototype, not a clinical tool. Its answers can be wrong; any clinical use
needs verification by a licensed pharmacist or clinician.

## Pipelines

| Pipeline | Tier | Evidence | Steps | Prompt files |
|---|---|---|---|---|
| `reasoning_only` | common | none (the model's general medication knowledge) | reasoner only | `reasoning_only_system.txt` |
| `api_only` | intermediate | the CPIC API's core recommendation | structured input (SI) → CPIC API → reasoner | `api_only_system.txt` |
| `full` | hard | the CPIC recommendation plus guideline and supplement passages | SI → CPIC API and retrieval of 8 corpus units → reasoner → critique → CPIC citation | `full_reasoner_system.txt`, `rerank.txt`, `full_critique.txt` |

Every pipeline answers with Qwen3.8-27B (the reasoner) and sends `answer_format_user.txt` as the
user turn of the reasoner call. Each step a pipeline adds raises its token cost, which a router
must trade against answer quality.

- **Reasoning-only.** The prompt is written for everyday medication questions and forbids
  genotype reasoning; a question that hinges on a genotype gets a conservative general answer
  that defers that part to the prescriber.
- **API-only.** Builds the SI, uses it to call the CPIC API and answers from the returned core
  recommendation. If the SI names no gene and no drug, no lookup is possible and the pipeline
  returns a fixed "No CPIC guideline information available" answer without calling the model.
- **Full.** Adds, alongside the API call, retrieval of 8 corpus units (below). A Gemma-4-31B-it
  critique then checks the draft against the CPIC API output and the units: it keeps what they
  support, corrects or removes what they do not, and adds source content the draft missed. Code
  finally strips model-written citation identifiers (PMIDs, DOIs, URLs, author-et-al references)
  and appends the link of the CPIC guideline for the question's first drug, read from the lookup
  table.

### Structured input (SI)

API-only and the Full pipeline start from the SI, which is built by lookup, not generation.
`pgxroute.steps.si_builder.build(question)` matches the question against `data/cpic_lookup.json`
and returns:

- the genes, genotypes and drugs the question names;
- the SI text: the question followed by these key elements, used as the retrieval query;
- the `{gene: {allele: count}}` argument of the CPIC recommendation lookup.

A gene, allele or drug is recognised only where the question names it as CPIC does (up to case
and the aliases in the table), and questions are assumed to name genotypes correctly. The module
docstring states the reading rules: each gene is read in its own clause, a single allele on an
autosomal gene is completed with the reference allele, and a gene that CPIC keys together with a
named one (TPMT with NUDT15, say) is supplied with CPIC's no-variant value.

The table is built from the CPIC API, so its aliases are CPIC's own: a parenthesised star allele
(`c.1905+1G>A (*2A)`), a synonym list, an HLA serotype without its colon, a variant position name
or dbSNP id, and an unspecified copy number. It also records each gene's reference allele,
chromosome and phenotype names, the gene groups CPIC keys together, the genes each drug's
recommendation is keyed on, and each drug's guideline. Its `meta` field gives the CPIC API release
and download date.

`pgxroute/steps/cpic_api.py` converts the SI's diplotypes to CPIC lookup keys, calls
`/rpc/recommendation_lookup`, keeps the rows for the SI's drugs (all rows if it names none) and
formats them as text for the reasoner.

### Retrieval (Full pipeline)

Retrieval runs in three steps (`pgxroute/steps/retrieval.py`):

1. **Scope.** Documents tagged with at least one of the SI's genes, narrowed to those that also
   name one of its drugs unless that leaves fewer than 8 units. Superseded documents
   (`superseded_by` in `doc_manifest.json`) are then removed, so a gene-drug pair covered by an
   older and a newer guideline is answered from the newest.
2. **Search.** BGE-M3 ranks the scoped units by hybrid similarity (dense + 0.3 × sparse) to the
   SI text; the top 60 form the pool.
3. **Rerank.** The reasoner model, with thinking off, reads the original question (not the SI)
   and the pool, and returns the 8 most relevant units under one inclusion rule and three
   exclusion rules (`prompts/rerank.txt`).

## Routers

| Router | Code | Reads | Picks by |
|---|---|---|---|
| `rule` | `pgxroute/routers/rule.py` | the SI | three rules: no gene and no variant → Reasoning-only; exactly one gene, at least one variant, exactly one drug and no condition → API-only; otherwise → Full |
| `logreg`, `xgboost` | `pgxroute/routers/ml/` | the question text | TF-IDF features with logistic regression or XGBoost |
| `fg`, `sf` | `pgxroute/routers/rl/` | the question text | Qwen3-8B with the facet-guided (`fg`) or self-faceted (`sf`) prompt, untrained or GRPO-trained |

```python
from pgxroute.router import get_router
pipeline = get_router("fg").route(question)    # "reasoning_only", "api_only" or "full"
```

```bash
# from PGxRoute/
python -m pgxroute.router --router rule --question "A 52-year-old man ..."
python run.py --router xgboost --questions questions.jsonl --out answers/   # route, then answer
```

- **Rule baseline.** Every question, common ones included, first goes through the SI step. The
  lookup-built SI has no condition vocabulary, so `route` applies the rules without conditions;
  pass `conditions` to `routers.rule.pick` if you have them.
- **Text classifiers.** TF-IDF over word 1–2-grams (minimum document frequency 2, sublinear term
  frequency), with logistic regression (scikit-learn defaults, `max_iter` 4000) or XGBoost (300
  trees, depth 4, learning rate 0.1, row/column subsampling 0.9/0.7). In the paper, they learn to
  predict the pipeline with the highest verifier F1.5 for each question (the one using fewer tokens
  on a tie), from questions rewritten to remove authoring-style cues, and are evaluated out of
  fold on the original questions; these labels and rewritten questions are not included. No
  trained model is included either. Train and save one from `PGxRoute/` with
  `mkdir -p data/routers && python -m pgxroute.routers.ml.text_classifier --kind xgboost --data bank.jsonl --out data/routers/xgboost.pkl`,
  where `bank.jsonl` holds `question` plus `pipeline` per line (or `tier`, mapped to the tier's
  pipeline). The router loads `data/routers/<name>.pkl` (`CLASSIFIER_DIR`); `two_fold_predict`
  gives out-of-fold picks.
- **LLM routers.** The prompt describes each pipeline as a tool (`prompts/routers/tool_lines.json`).
  FG adds a fixed list of six clinical question facets named by the pharmacists (pharmacogenetic
  risk analysis; dosing recommendation; safety and toxicity risk; alternative therapy and clinical
  context; monitoring strategy; testing indication and implications) and asks for the smallest
  sufficient pipeline; SF lets the model list in its own words what the question asks for. Serve
  Qwen3-8B (untrained) or a trained checkpoint with an OpenAI-compatible server and set
  `ROUTER_BASE_URL` / `ROUTER_MODEL`. Sampling: thinking on, temperature 0.6, top-p 0.95, 2,048
  new tokens. Only a JSON tool name after the last `</think>` counts; an unparsable reply is
  resampled up to twice, then the question goes to Reasoning-only. No trained checkpoint is
  included; the training code is in `../Router-training`.

## Models and decoding

Two OpenAI-compatible vLLM servers are expected:

| Model | Served name | Used for | Setting |
|---|---|---|---|
| Qwen3.8-27B | `qwen3.8-27b` | reasoner (all pipelines), reranker (Full) | `REASONER_BASE_URL` (default `http://127.0.0.1:8019/v1`) |
| Gemma-4-31B-it | `gemma-4-31b-it` | critique (Full) | `CRITIC_URL` (default `http://127.0.0.1:8006/v1/chat/completions`) |

Server launch settings (`vllm serve <model> <flags>`):

| Model | vLLM | Flags |
|---|---|---|
| Qwen3.8-27B | 0.19.1 | `--served-model-name qwen3.8-27b --port 8019 --dtype bfloat16 --max-model-len 49152` |
| Gemma-4-31B-it | 0.19.1 | `--served-model-name gemma-4-31b-it --port 8006 --dtype bfloat16 --max-model-len 32768` |

The Qwen3.8-27B context must hold the reranker's 40,960-token window. Sampling parameters that a
request leaves unset come from the model's `generation_config.json` (vLLM's default
`--generation-config auto`): top-k 20 for the reasoner, and top-p 0.95 and top-k 64 for the
critique.

BGE-M3 (`BAAI/bge-m3`) runs in process through FlagEmbedding on `EMBED_DEVICE`. If FlagEmbedding
needs a different `transformers` version from the main environment, install it in a separate
environment and set `EMBED_PYTHON` to that interpreter; ranking then runs as a subprocess
(`python -m pgxroute.steps.retrieval --rank`, started from `PGxRoute/`). That environment needs
only FlagEmbedding and its dependencies.

Decoding settings (defaults in `pgxroute/config.py`):

- **Reasoner:** temperature 0.05, top-p 0.95, thinking on with reasoning effort medium, output cap
  24,000 tokens. The reasoning block is removed before delivery, and a truncated completion raises
  and is not recorded.
- **Reranker:** temperature 0, thinking off, output cap 4,000 tokens (raised to 8,000 and 16,000
  only if the ranking is truncated). Passages are shortened to fit a 40,960-token window.
- **Critique:** temperature 0.05, output cap 2,048 tokens. If the critique fails, the draft is
  kept.
- **Inputs passed on:** the CPIC API result is cut at 8,000 characters and the retrieved units at
  32,000; API-only, which retrieves nothing, gets an explicit "nothing retrieved" marker in their
  place.

## Running

```bash
pip install -r requirements.txt
python run.py --pipeline full --question "A 52-year-old man ..."
python run.py --pipeline api_only --questions questions.jsonl --out answers/
```

`run.py` can be started from any directory; `--router <name>` lets a router pick the pipeline
instead of `--pipeline`. From Python, with `PGxRoute/` on the path:

```python
from pgxroute.pipelines import answer
result = answer("A 52-year-old man ...", pipeline="api_only")
```

Each output records:

- `answer`: the final text.
- The intermediate outputs: `structured_input` (the SI text), `genes`, `variants`, `drugs`,
  `cpic_result`, `retrieval`, `draft`, `critiqued` and `citation`.
- `raw_completion`: the reasoner's full output, including its reasoning block.
- `usage_by_step` and `usage`: input and output tokens of the reasoner, reranker and critique
  calls. Building the SI is a table lookup and costs no tokens.
- `timing`: seconds per step.

## Retrieval corpus

The corpus is built from CPIC guideline articles and supplements, whose text is mostly under
publisher copyright, so it is not included; neither is the code that built it. The paper's corpus
has 77 documents (29 guidelines and 48 supplements); 19 superseded documents are excluded at
retrieval, leaving 58 documents and 1,725 retrievable units (prose passages, table-row groups and
figure descriptions).

To use your own corpus, write each document as one JSON file:

- `doc_name`, `source`, and the document's genes and drugs as lists in `gene` and `drug`;
- `prose_chunks`: a list of prose units, each with a `self_contained_text`;
- `blocks`: tables and figures. A table's units are its `table_chunks`, each with a
  `self_contained_text`; a figure is a block with `"type": "figure"` and its own
  `self_contained_text`.

List the documents in `doc_manifest.json`, one entry per document with a `doc_id` (its
`doc_name`), and mark superseded ones with `superseded_by`. The code checks the corpus at load
against `_provenance.json` and `EXPECTED_CORPUS_VERSION`; set `VERIFY_CORPUS=0` to use a corpus
you built yourself, and run `python -m pgxroute.steps.retrieval --verify` (from `PGxRoute/`) to
check a copy of the original.

```
data/cpic_lookup.json            the CPIC lookup table (included)
data/corpus/                     retrieval corpus (not included)
data/emb_cache/                  BGE-M3 unit embeddings (created on first use)
data/routers/                    trained text classifiers (not included)
```

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
| `pgxroute/router.py` | `get_router(name)` and `route(question, name)` for the five routers; also a command-line entry |
| `pgxroute/config.py` | Endpoints, models, decoding settings, limits and paths (paths default to folders under `PGxRoute/`) |
| `pgxroute/steps/si_builder.py` | Builds the SI by lookup against the CPIC table |
| `pgxroute/steps/cpic_api.py` | CPIC lookup keys, the `/rpc/recommendation_lookup` call, rows for the SI's drugs |
| `pgxroute/steps/retrieval.py` | Corpus check, gene/drug scope, BGE-M3 hybrid ranking, listwise reranking |
| `pgxroute/steps/reasoner.py` | Reasoner request and answer cleaning |
| `pgxroute/steps/critique.py` | Critique of the Full pipeline's draft |
| `pgxroute/steps/citation.py` | Strips model-written citation identifiers and appends the CPIC guideline link |
| `pgxroute/routers/rule.py` | Rule baseline over the SI's genes, variants and drugs |
| `pgxroute/routers/ml/text_classifier.py` | TF-IDF text classifiers: train, save, load, route, out-of-fold picks |
| `pgxroute/routers/rl/llm_router.py` | LLM router: builds the FG or SF prompt, samples, parses the JSON after `</think>`, resamples |
| `prompts/` | The six pipeline prompts, and `prompts/routers/` with the FG and SF router templates and `tool_lines.json`, the tool descriptions both templates fill in at `{tool_list}` |
| `data/cpic_lookup.json` | The CPIC lookup table: every gene, allele, drug and phenotype name of the CPIC API, with the aliases CPIC prints for them, each gene's reference allele and chromosome, the gene groups CPIC keys together, and each drug's guideline |
| `tools/audit_question_names.py` | Offline check of a question bank against the table (reads ground truth; never used at answer time) |

## Licence

Code and prompts: Apache License 2.0 (`LICENSE`). The CPIC API content the pipelines retrieve is
CC0 1.0. The models keep their own licences.
