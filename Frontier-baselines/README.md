# Frontier-model baselines

The prompt and settings used to ask frontier models with web search the same questions the
PGxRoute pipelines answer.

## Prompt

`prompts/system_prompt.txt` is the whole system prompt, sent unchanged to every model. The user
turn is the question text alone, with nothing added.

The prompt was condensed from the API-only, Full-pipeline and Reasoning-only pipeline prompts
into one prompt for a model without retrieval. Rules that presuppose pasted reference sections
were removed, while these were kept: strength labels, the primary-before-conditional order, the
evidence-grade vs. classification rule, the ban on fabricated citation identifiers, the scope
guard, the FINAL ANSWER line, and one worked example on an invented gene and drug. A gene-drug
answer may use CPIC sources only: the CPIC guideline, its supplement and the CPIC API.

## Models and settings

| Model | API | Settings |
|---|---|---|
| GPT-5.4 (`gpt-5.4`), GPT-5.6-sol (`gpt-5.6-sol`) | OpenAI Responses API | `instructions` = the prompt, `input` = the question; reasoning effort `high`; tools `[web_search]`, `tool_choice` `auto`; no temperature or output cap set |
| Gemini-3.8-flash (`gemini-3.8-flash`), Gemini-3.1-pro-preview (`gemini-3.1-pro-preview`) | Gemini `generateContent` | `system_instruction` = the prompt, one user turn = the question; tools `[google_search]`; temperature 0.0 |

Some GPT calls also set `max_tool_calls` to 10 (`--max-tool-calls 10`).

## Running

```bash
export OPENAI_API_KEY=...    # for gpt-*
export GEMINI_API_KEY=...    # for gemini-*
python run.py --model gpt-5.4 --questions questions.jsonl --out answers/gpt-5.4
python run.py --model gemini-3.8-flash --questions questions.jsonl --out answers/gemini-3.8-flash
```

`questions.jsonl` holds one `{"question_id": ..., "question": ...}` per line. Each answer is
written to `<out>/<question_id>.json` with the answer text, the number of search calls and the
token usage. Existing files are skipped, so a run can be resumed. `run.py` uses the Python
standard library only. Answers can be scored with `Verifier/` (the `answer` field).

## Licence

Code and prompt: Apache License 2.0 (`LICENSE`).
