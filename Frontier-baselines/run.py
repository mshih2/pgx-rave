"""Ask a frontier model with web search, using prompts/system_prompt.txt.

  python run.py --model gpt-5.4 --questions questions.jsonl --out answers/gpt-5.4
  python run.py --model gemini-3.8-flash --questions questions.jsonl --out answers/gemini-3.8-flash

questions.jsonl: one {"question_id": ..., "question": ...} per line. The system prompt is sent as
the system instruction and the question, unchanged, as the only user turn.

  gpt-*     OpenAI Responses API: reasoning effort high, tools [web_search], tool_choice auto;
            no temperature and no output cap are set. Key: OPENAI_API_KEY.
  gemini-*  Gemini generateContent: tools [google_search], temperature 0.0. Key: GEMINI_API_KEY.

Each answer is written to <out>/<question_id>.json; existing files are skipped, so a run can be
resumed. Standard library only.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = (HERE / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").rstrip()
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
GEMINI_BASE = os.getenv("GEMINI_BASE_URL",
                        "https://generativelanguage.googleapis.com/v1beta").rstrip("/")


def _post(url, body, headers, timeout, attempts, retry_codes):
    """-> (response json, None) or (None, error text)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    for attempt in range(attempts):
        try:
            return json.load(urllib.request.urlopen(req, timeout=timeout)), None
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in retry_codes and attempt < attempts - 1:
                continue
            return None, f"HTTP {e.code} {msg[:150]}"
        except Exception as e:
            if attempt < attempts - 1:
                continue
            return None, f"ERR {e}"
    return None, "give up"


def ask_openai(model, question, max_tool_calls=None):
    body = {"model": model, "reasoning": {"effort": "high"},
            "tools": [{"type": "web_search"}], "tool_choice": "auto",
            "instructions": SYSTEM, "input": question}
    if max_tool_calls:
        body["max_tool_calls"] = max_tool_calls
    r, err = _post(f"{OPENAI_BASE}/responses", body,
                   {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                   timeout=1800, attempts=3, retry_codes=(429, 500, 502, 503, 529))
    if err:
        return None, err
    text = "".join(c.get("text", "") for i in r.get("output", []) if i.get("type") == "message"
                   for c in i.get("content", []))
    types = [i.get("type") for i in r.get("output", [])]
    return {"answer": text, "output_types": types,
            "n_search_calls": sum(1 for t in types if "search" in str(t)),
            "usage": r.get("usage", {})}, None


def ask_gemini(model, question, max_tool_calls=None):
    body = {"system_instruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.0}}
    r, err = _post(f"{GEMINI_BASE}/models/{model}:generateContent", body,
                   {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                   timeout=1200, attempts=4, retry_codes=(429, 500, 502, 503))
    if err:
        return None, err
    if not r.get("candidates"):       # safety block, recitation or quota
        return None, "NO_CANDIDATES: " + json.dumps(r.get("promptFeedback") or r.get("error") or r)[:200]
    c = r["candidates"][0]
    text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if "text" in p)
    gm = c.get("groundingMetadata", {})
    return {"answer": text, "n_search_calls": len(gm.get("webSearchQueries") or []),
            "n_grounding_chunks": len(gm.get("groundingChunks") or []),
            "search_queries": gm.get("webSearchQueries") or [],
            "finish_reason": c.get("finishReason"), "usage": r.get("usageMetadata", {})}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gpt-... or gemini-...")
    ap.add_argument("--questions", required=True, help="JSONL with question_id and question")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tool-calls", type=int, help="gpt-* only: cap on web-search calls")
    args = ap.parse_args()
    if args.model.startswith("gpt"):
        ask = ask_openai
    elif args.model.startswith("gemini"):
        ask = ask_gemini
    else:
        ap.error("--model must start with gpt or gemini")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]

    def one(item):
        dst = out / f"{item['question_id']}.json"
        if dst.exists():
            return item["question_id"], "cached"
        res, err = ask(args.model, item["question"], args.max_tool_calls)
        if err:
            return item["question_id"], err
        dst.write_text(json.dumps({"question_id": item["question_id"], "model": args.model,
                                   "question": item["question"], **res}, indent=1),
                       encoding="utf-8")
        return item["question_id"], f"{len(res['answer'].split())} words, " \
                                    f"{res['n_search_calls']} searches"

    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for qid, status in ex.map(one, items):
            if status.startswith(("HTTP", "ERR", "give", "NO_CANDIDATES")):
                failed += 1
            print(f"{qid}: {status}", flush=True)
    print(f"{args.model}: {len(items) - failed} ok, {failed} failed of {len(items)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
