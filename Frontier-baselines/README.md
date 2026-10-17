# Frontier-model baselines

The system prompt and runner used to ask frontier models with web search the same questions the
PGxRoute pipelines answer. The comparison asks whether a clinician would do as well with a
general-purpose frontier model and web search, which must find and correctly convey the CPIC
content that PGxRoute reads directly.

## Prompt

`prompts/system_prompt.txt` is the whole system prompt (398 words), sent unchanged to every
model. The user turn is the question text alone, with nothing added.

Grok 4.5 condensed the prompt from three pipeline prompts of the time (API-only, an earlier
Full-pipeline reasoner and Reasoning-only), keeping their answer rules and dropping those that
presuppose retrieved text pasted into the prompt; we then edited it twice by hand. The rules kept
are: strength labels, the primary-before-conditional order, the evidence-grade vs. classification
rule, the ban on fabricated citation identifiers, the scope guard, the FINAL ANSWER line, and one
worked example on an invented gene and drug. A gene-drug answer may use CPIC sources only: the
CPIC guideline, its supplement and the CPIC API.

## Models and settings

| Model | API | Settings |
|---|---|---|
| GPT-5.4 (`gpt-5.4`), GPT-5.6-sol (`gpt-5.6-sol`) | OpenAI Responses API | `instructions` = the prompt, `input` = the question; reasoning effort `high`; tools `[web_search]`, `tool_choice` `auto`; no temperature or output cap set |
| Gemini-3.8-flash (`gemini-3.8-flash`), Gemini-3.1-pro-preview (`gemini-3.1-pro-preview`) | Gemini `generateContent` | `system_instruction` = the prompt, one user turn = the question; tools `[google_search]` (Google Search grounding); temperature 0.0 |

`--max-tool-calls N` caps the number of web-search calls of a GPT model; some GPT calls in our
runs set it to 10. In the paper, each model answered every question 10 times. Claude and Grok are
not among the baselines: we used them heavily in this work, so their answers could be favoured.

## Running

```bash
export OPENAI_API_KEY=...    # for gpt-*
export GEMINI_API_KEY=...    # for gemini-*
python run.py --model gpt-5.4 --questions questions.jsonl --out answers/gpt-5.4
python run.py --model gemini-3.8-flash --questions questions.jsonl --out answers/gemini-3.8-flash
```

`questions.jsonl` holds one `{"question_id": ..., "question": ...}` per line. Each answer is
written to `<out>/<question_id>.json` with the answer text, the number of search calls and the
vendor's token usage. Existing files are skipped, so a run can be resumed. `run.py` uses the
Python standard library only.

Answers can be scored with `../Verifier` (the `answer` field). The paper counts GPT tokens as API
input (with search results) plus output (with reasoning), and Gemini tokens as prompt, candidate
and thought tokens; vendor token counts are not like-for-like.

## Licence

Code and prompt: Apache License 2.0 (`LICENSE`).
