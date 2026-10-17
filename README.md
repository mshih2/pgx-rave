# PGx-RAVE

Code, prompts and question bank for PGx-RAVE (Route, Answer, Verify, Evolve), an agentic framework
that answers open-ended pharmacogenomic (PGx) questions from the guidance of the Clinical
Pharmacogenetics Implementation Consortium (CPIC).

PGx-RAVE is a research prototype, not a clinical tool. Its answers can be wrong; any clinical use
needs verification by a licensed pharmacist or clinician.

## The four steps

| Step | What it does | Folder |
|---|---|---|
| Route | A router reads the question and sends it to one of three answer pipelines. | [`PGxRoute/`](PGxRoute/) |
| Answer | The pipeline gathers only the evidence the question's tier needs (none, one CPIC API lookup, or the lookup plus CPIC guideline and supplement passages) and answers with a mid-size open model (27B–31B). | [`PGxRoute/`](PGxRoute/) |
| Verify | An atomic-claim verifier splits the answer and the ground truth (GT) into single-fact claims, labels and grades each claim, and returns precision, recall and F1.5. | [`Verifier/`](Verifier/) |
| Evolve | The verifier's results feed back into the system, in two loops. The verifier's score trains the router (the code is in this repository). Its claim labels guided one round of prompt fixes, steered by the authors and largely carried out by LLMs, with no code here; an evolving agent that automates this loop is future work. | [`Router-training/`](Router-training/) |

Route and Answer together are called PGxRoute.

## Repository layout

| Folder | What it holds |
|---|---|
| [`PGxRoute/`](PGxRoute/) | The three answer pipelines (Reasoning-only, API-only, Full) and five routers: a rule baseline, two TF-IDF text classifiers (logistic regression, XGBoost) and Qwen3-8B LLM routers with the facet-guided (FG) or self-faceted (SF) prompt |
| [`Verifier/`](Verifier/) | The atomic-claim verifier: claim decomposition, the labelling and grading judges, and the scores |
| [`Router-training/`](Router-training/) | GRPO training of the LLM routers, with rewards computed live from the PGxRoute pipelines and the Verifier (includes the modified verl it runs on) |
| [`Question-generation/`](Question-generation/) | Code that builds a question bank: questions, GT and curated GT claims for the three tiers, cross-reviewed by two LLMs before a human check |
| [`Question-bank/pharmacist-written/`](Question-bank/pharmacist-written/) | The 12 hard question-answer pairs written by pharmacists (question IDs 0–11) |
| [`Question-bank/ai-assisted/`](Question-bank/ai-assisted/) | The other 153 questions (IDs 12–164), generated with AI assistance, each with its GT answer and GT claims |
| [`Question-bank/folds.txt`](Question-bank/folds.txt) | The routers' two-fold cross-validation split of the 135 evaluation questions (question ID, fold) |
| [`Frontier-baselines/`](Frontier-baselines/) | The system prompt and runner used to ask frontier models with web search the same questions |

Each code folder has its own README, requirements and licence. PGxRoute and the Verifier run on
their own; the other folders build on them:

- Router-training answers each training question with all three PGxRoute pipelines and scores
  the answers with the Verifier, at every step.
- Question-generation gets each question's CPIC lookup and guideline passages through PGxRoute,
  and decomposes the GT into claims with the Verifier.
- Answers from PGxRoute or Frontier-baselines are scored with the Verifier against the bank's
  GT claims.

## Question bank

The bank has 165 English questions, 55 per tier. Tiers are defined by where the evidence lives:

- **hard:** needs CPIC guideline or supplement text beyond the CPIC API's core recommendation;
- **intermediate:** answerable from one CPIC API lookup;
- **common:** everyday medication questions with no PGx content.

Thirty questions (10 per tier) were used to tune the pipeline prompts and test basic routing
methods. The remaining 135 (45 per tier) are the evaluation set; after decomposition, they hold
2,040 GT claims.

We built the bank with three licensed pharmacists experienced in interpreting CPIC guidelines.
They defined with us the principles for a question-answer pair (the elements a question must
state, the facets an answer must cover and the answer format) and wrote 12 hard sample pairs from
the CPIC guidelines and drug information.

- The other hard and intermediate questions were generated with AI assistance, modelled on these
  12 in what an answer must cover, its format and its easy-to-digest style. Their GT is built
  only from the tier's CPIC source and verified against it; where CPIC is silent, the GT says so.
- The common questions were also generated with AI assistance; their GT was reached through a
  discussion between two LLMs.

LLMs cross-reviewed each new question, and the non-pharmacist authors then checked every item
word by word. Apart from the 12 pharmacist-written pairs, no question or GT has yet been reviewed
by a pharmacist. [`Question-generation/`](Question-generation/) is the code for building questions
and GT this way.

## Models and services

| Used by | Model or service | How it is reached |
|---|---|---|
| PGxRoute pipelines | Qwen3.8-27B (reasoner, reranker), Gemma-4-31B-it (critique), BGE-M3 (retrieval) | your own vLLM servers; BGE-M3 in process or in a separate environment |
| PGxRoute LLM routers | Qwen3-8B, untrained or GRPO-trained | your own vLLM server |
| PGxRoute, Question-generation | CPIC API | `https://api.cpicpgx.org/v1` |
| Verifier | Gemma-4-31B-it (decomposition), Mistral-Small-24B-Instruct-2501 (grading judge) | your own vLLM servers |
| Verifier | Grok 4.5 (labelling judge) | xAI API, `XAI_API_KEY` |
| Question-generation | Claude Opus 4.8, Claude Fable 5.1 | Anthropic API, `ANTHROPIC_API_KEY` |
| Frontier-baselines | GPT-5.4, GPT-5.6-sol, Gemini-3.8-flash, Gemini-3.1-pro-preview (with web search) | `OPENAI_API_KEY`, `GEMINI_API_KEY` |

Server addresses and model names are set through environment variables, described in each
folder's README. API keys are read from the environment (Question-generation also accepts an
`ant auth login` profile) and are never stored in the repository.

## Quick start

Answer a question with the API-only pipeline, then score the answer against GT claims:

```bash
pip install -r PGxRoute/requirements.txt -r Verifier/requirements.txt

# PGxRoute: a Qwen3.8-27B server at REASONER_BASE_URL
python PGxRoute/run.py --pipeline api_only --question "A 52-year-old man ..."

# Verifier: the decomposer and grader servers, and XAI_API_KEY
python Verifier/run.py --input items.jsonl --out results/
```

`items.jsonl` holds `{"question_id", "question", "answer", "gt_claims"}` per line (see
`Verifier/README.md`); the bank's questions and GT claims are in `Question-bank/ai-assisted/`.

## Not included

Apart from the question-answer pairs in `Question-bank/`, this repository holds code and prompts
only: no model outputs or scores, and no trained routers. `PGxRoute/` and `Router-training/`
describe how to train your own.

**Retrieval corpus.** The Full pipeline retrieves from a text corpus built once, offline, from
CPIC guideline articles and their supplements. The corpus is not included, for two reasons:

- Most of these documents are under publisher copyright, and their licences differ from one
  document to the next.
- A corpus file also carries derived content (repaired tables and model-written figure
  descriptions), which some licences do not allow to be redistributed.

To run the Full pipeline, build a corpus from the guideline PDFs in the format described in
`PGxRoute/README.md` (one JSON file per document, plus `doc_manifest.json`), place it in
`PGxRoute/data/corpus/` (or set `CORPUS_DIR`), and set `VERIFY_CORPUS=0`.

The corpus is needed only by the Full pipeline and the steps that use it: hard questions in
Question-generation, and the live reward in Router-training. These parts work without it:

- the API-only and Reasoning-only pipelines;
- the routers;
- the Verifier;
- the intermediate and common tiers of Question-generation;
- Frontier-baselines.

## Licence

The code and prompts in every folder are under the Apache License 2.0 (each folder's `LICENSE`).
The question-answer pairs in `Question-bank/` are under CC BY 4.0 (the `LICENSE` in each of its
folders). The verl code in `Router-training/verl/` keeps its Apache 2.0 licence and `NOTICE`. CPIC
API content is CC0 1.0. The models and APIs keep their own licences and terms. The code, prompts
and question-answer pairs we release are for research use, not clinical decisions.
