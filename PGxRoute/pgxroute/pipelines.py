"""The three PGx-RAVE answer pipelines.

  api_only        SI -> CPIC API -> reasoner
  reasoning_only  reasoner only (general-medicine prompt; no CPIC, no retrieval)
  full            SI -> CPIC API -> retrieval (BGE-M3 + reranker, top 8) -> reasoner
                  -> critique -> CPIC citation

The SI is built from the question by lookup against the CPIC table (si_builder).

Every pipeline answers with Qwen3.8-27B. A router (pgxroute.router) picks one pipeline per
question.
"""
import logging
import time

from . import config
from .steps import cpic_api, reasoner, si_builder

log = logging.getLogger(__name__)

PIPELINES = ("api_only", "reasoning_only", "full")
# Returned by API-only when the SI names no gene and no drug (no CPIC lookup is possible).
NO_CPIC_ANSWER = ('{"title": "No CPIC guideline information available", '
                  '"details": "No CPIC guideline information available"}')

_prompts = {}


def _prompt(name):
    if name not in _prompts:
        _prompts[name] = (config.PROMPT_DIR / name).read_text(encoding="utf-8")
    return _prompts[name]


def _add(total, usage):
    total["input_tokens"] += int(usage.get("input_tokens") or 0)
    total["output_tokens"] += int(usage.get("output_tokens") or 0)


def _run_reasoner(result, template, question, cpic_text="", context=""):
    messages = reasoner.build_messages(_prompt(template), question, cpic_text, context,
                                       _prompt("answer_format_user.txt"))
    t = time.time()
    raw, usage = reasoner.generate(messages)
    result["timing"]["reasoner"] = time.time() - t
    result["usage_by_step"]["reasoner"] = {k: usage[k] for k in ("input_tokens", "output_tokens")}
    result["raw_completion"] = raw
    return reasoner.clean_answer(raw)


def answer(question, pipeline="full"):
    """Answer one question with one pipeline. Returns a dict whose "answer" is the final text."""
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline must be one of {PIPELINES}, got {pipeline!r}")
    question = (question or "").strip()
    if not question:
        raise ValueError("empty question")
    t0 = time.time()
    result = {"question": question, "pipeline": pipeline, "answer": "",
              "timing": {}, "usage_by_step": {}}

    if pipeline == "reasoning_only":
        result["answer"] = _run_reasoner(result, "reasoning_only_system.txt", question)
    else:
        t = time.time()
        si = si_builder.build(question, cpic_api.client)
        result["timing"]["structured_input"] = time.time() - t
        # Gene x drug rows scope the CPIC lookup and the retrieval corpus; a question that
        # names only genes or only drugs scopes by whichever it names.
        rows = ([{"Gene Name": g, "Drug Name": d} for g in si["genes"] for d in si["drugs"]]
                or [{"Gene Name": g} for g in si["genes"]]
                or [{"Drug Name": d} for d in si["drugs"]])
        result.update(structured_input=si["si_text"], genes=si["genes"],
                      variants=si["variants"], drugs=si["drugs"], scope_rows=rows,
                      alleles=si["lookup"])

        t = time.time()
        # The lookup carries partner genes CPIC keys together with the question's own; only the
        # genes the question reports are named as tested.
        cpic_text = (cpic_api.client.search(rows, si["lookup"], tested_genes=si["genes"],
                                            supplied_genes=si["supplied_genes"],
                                            inferred_genes=si["phenotypes"]) if rows else "")
        result["timing"]["cpic_api"] = time.time() - t
        result["cpic_result"] = cpic_text

        if pipeline == "api_only":
            if rows:
                result["answer"] = _run_reasoner(result, "api_only_system.txt", question, cpic_text)
            else:
                result["answer"] = NO_CPIC_ANSWER
        else:
            _full(result, question, si["si_text"], rows, cpic_text)

    total = {"input_tokens": 0, "output_tokens": 0}
    for u in result["usage_by_step"].values():
        _add(total, u)
    result["usage"] = total
    result["timing"]["total"] = time.time() - t0
    return result


def _full(result, question, si, rows, cpic_text):
    from .steps import citation, critique, retrieval

    t = time.time()
    r = retrieval.retrieve(si, rows, question)
    result["timing"]["retrieval"] = time.time() - t
    result["usage_by_step"]["reranker"] = r.pop("usage")
    result["retrieval"] = r

    draft = _run_reasoner(result, "full_reasoner_system.txt", question, cpic_text, r["context"])
    if not draft:
        raise RuntimeError("reasoner produced no draft")
    result["draft"] = draft

    t = time.time()
    revised, usage = critique.critique(question, cpic_text, r["context"], draft)
    result["timing"]["critique"] = time.time() - t
    result["usage_by_step"]["critique"] = usage
    if revised is None:
        log.error("critique failed on every attempt; keeping the draft")
        revised = draft
    result["critiqued"] = revised

    genes, drugs = [], []
    for row in rows:
        g, d = row.get("Gene Name"), row.get("Drug Name")
        if g and g not in genes:
            genes.append(g)
        if d and d not in drugs:
            drugs.append(d)
    # One citation per answer: the guideline of the first SI drug.
    final, removed, guideline = citation.attach_citation(revised, genes, drugs[0] if drugs else "")
    result["answer"] = final
    result["citation"] = {"guideline": guideline, "removed": removed}
