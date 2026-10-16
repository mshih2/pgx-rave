# PGx-RAVE

Code and prompts for PGx-RAVE (Route, Answer, Verify, Evolve), an agentic framework that answers
open-ended pharmacogenomic (PGx) questions from CPIC guidance:

- **Route.** A router reads the question and picks one of three answer pipelines.
- **Answer.** The pipelines run on mid-size open models. Each is matched to where a question's
  evidence lives: general drug knowledge, one CPIC API lookup, or CPIC guideline and supplement text.
- **Verify.** An atomic-claim verifier splits the answer and the ground truth (GT) into
  single-fact claims, labels and grades each claim, and returns precision, recall and F1.5.
- **Evolve.** A self-improvement loop driven by the verifier's claim labels. It is in development
  and not part of this repository.

PGx-RAVE is a research prototype, not a clinical tool. Its answers can be wrong; any clinical use
needs verification by a licensed pharmacist or clinician.

## Repository layout

| Folder | What it holds |
|---|---|
| [`PGxRoute/`](PGxRoute/) | Route and Answer: the three answer pipelines (Reasoning-only, API-only, Full) and the routers (rule, TF-IDF + logistic regression / XGBoost, and Qwen3-8B LLM routers with the facet-guided or self-faceted prompt) |
| [`Verifier/`](Verifier/) | Verify: claim decomposition, the labelling and grading judges, and the scores |
| [`Question-generation/`](Question-generation/) | Builds a question bank: hard, intermediate and common questions with GT and curated GT claims, cross-reviewed by two LLMs before a human check |
| [`Router-training/`](Router-training/) | GRPO training of the LLM router, with rewards computed live from the PGxRoute pipelines and the Verifier (includes the modified verl it runs on) |
| [`Frontier-baselines/`](Frontier-baselines/) | The system prompt and runner for asking frontier models with web search the same questions |

Each folder has its own README, requirements and licence. The folders work together:

- Question-generation gets each question's CPIC lookup and passages through PGxRoute, and
  decomposes GT claims with the Verifier.
- Router-training answers each training question with every PGxRoute pipeline and scores the
  answers with the Verifier, every step.
- Answers from PGxRoute or Frontier-baselines are scored with the Verifier against the bank's
  GT claims.

## Models and services

| Used by | Model or service | How it is reached |
|---|---|---|
| PGxRoute | Qwen3.8-27B (reasoner, reranker), Gemma-4-31B-it (critique), BGE-M3 (retrieval) | your own vLLM servers; BGE-M3 in process or in a separate environment |
| PGxRoute LLM routers | Qwen3-8B, untrained or trained | your own vLLM server |
| PGxRoute, Question-generation | CPIC API | `https://api.cpicpgx.org/v1` |
| Verifier | Gemma-4-31B-it (decomposition), Mistral-Small-24B (grading) | your own vLLM servers |
| Verifier | Grok 4.5 (labelling judge) | xAI API, `XAI_API_KEY` |
| Question-generation | Claude Opus 4.8, Claude Fable 5.1 | Anthropic API, `ANTHROPIC_API_KEY` |
| Frontier-baselines | GPT, Gemini (with web search) | `OPENAI_API_KEY`, `GEMINI_API_KEY` |

Server addresses and model names are set through environment variables, described in each
folder's README. API keys are read from the environment only.

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
`Verifier/README.md`).

## Not included

This repository holds code and prompts only. It includes no question bank, model outputs or
scores, and no trained router models; `PGxRoute/` and `Router-training/` describe how to train
your own.

**Retrieval corpus.** The Full pipeline retrieves from a corpus built from CPIC guideline
articles and their supplements. The corpus is not included, for two reasons:

- Most of these documents are under publisher copyright, and their licences differ from one
  document to the next.
- A corpus file also carries derived content (repaired tables and model-written figure
  descriptions), which some licences do not allow to be redistributed.

To run the Full pipeline, build a corpus from the guideline PDFs in the format described in
`PGxRoute/README.md`: one JSON file per document, plus `doc_manifest.json`. Place it in
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
The verl code in `Router-training/verl/` keeps its Apache 2.0 licence and `NOTICE`. CPIC API
content is CC0 1.0. The models and APIs keep their own licences and terms.
