"""Select the minimal guideline passages a hard question's GT is written from.

  bundle    the question, its CPIC section and every current passage of the documents covering
            its gene and drug, numbered P1..Pn. Built once, by code, so both selectors read the
            same material.
  select    the two LLMs (config.READERS) each name the smallest passage set that answers the
            question completely. They work blind: the prompt is byte-identical for both, carries
            nothing written for this one question, and neither sees the other's answer.
  converge  config.CONVERGER reads the bundle and both selections and decides the final set by
            reading the passages, not by counting votes. It also writes the notes the GT writer
            needs, quoting the passages beside any summary of them.
"""
import hashlib
import json
import logging

from .. import config, files, llm
from ..sources import corpus, cpic

log = logging.getLogger(__name__)


def bundle(qid, rebuild=False):
    path = files.qdir(qid) / "bundle.json"
    if path.exists() and not rebuild:
        return files.read_json(path)
    q = files.read_json(files.qdir(qid) / "question.json")
    section = cpic.cpic_section(q["question"])
    files.write_json(files.qdir(qid) / "section.json", section)
    units = corpus.scope(section["genes"] or q["genes"], section["drugs"] or q["drugs"])
    if not units:
        raise RuntimeError(f"{qid}: no current guideline or supplement text covers "
                           f"{q['genes']} and {q['drugs']}; it cannot be a hard question")
    b = {"qid": qid, "question": q["question"], "cpic_text": section["cpic_text"],
         "passages": [{"id": f"P{i}", "uid": u["uid"], "doc": u["doc"], "kind": u["kind"],
                       "text": u["text"]} for i, u in enumerate(units, 1)]}
    files.write_json(path, b)
    return b


def bundle_text(b, ids=None):
    keep = [p for p in b["passages"] if ids is None or p["id"] in ids]
    out = [f"## THE QUESTION\n\n{b['question'].strip()}",
           f"## THE CPIC SECTION\n\nThe CPIC recommendation lookup's output for this patient, as "
           f"the answer receives it whole.\n\n```text\n{b['cpic_text'].strip() or '(empty)'}\n```",
           f"## PASSAGES ({len(keep)})"]
    for p in keep:
        out.append(f"### {p['id']}  ({p['kind']}, {p['doc']})\n```text\n{p['text'].strip()}\n```")
    return "\n\n".join(out)


def _check_ids(obj_ids, b, who):
    valid = {p["id"] for p in b["passages"]}
    bad = [i for i in obj_ids if i not in valid]
    if bad:
        raise RuntimeError(f"{who} named passages that are not in the bundle: {bad}")


def select(qid):
    b = bundle(qid)
    prompt = files.prompt("minimum_set.txt", principles=files.principles(), bundle=bundle_text(b))
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    for model, (obj, reply) in llm.ask_json_each(config.READERS, prompt).items():
        _check_ids(obj.get("minimum_set") or [], b, model)
        files.write_json(files.qdir(qid) / f"selection_{files.tag(model)}.json",
                         {**obj, "prompt_sha256": digest, **reply.record()})
        files.log_event(qid, "select", **reply.record())


def converge(qid):
    b = bundle(qid)
    meta = ("prompt_sha256", "requested_model", "served_by", "usage")
    runs = []
    for i, model in enumerate(config.READERS, 1):
        sel = files.read_json(files.qdir(qid) / f"selection_{files.tag(model)}.json")
        sel = {k: v for k, v in sel.items() if k not in meta}
        runs.append(f"### Selection {i}\n```json\n{json.dumps(sel, ensure_ascii=False, indent=1)}\n```")
    prompt = files.prompt("converge.txt", principles=files.principles(), bundle=bundle_text(b),
                          selections="\n\n".join(runs))
    obj, reply = llm.ask_json(config.CONVERGER, prompt)
    ids = [m["id"] for m in obj.get("final") or []]
    if not ids:
        raise RuntimeError(f"{qid}: converge returned an empty passage set")
    _check_ids(ids, b, config.CONVERGER)
    uid = {p["id"]: p["uid"] for p in b["passages"]}
    for m in obj["final"]:
        m["uid"] = uid[m["id"]]
    files.write_json(files.qdir(qid) / "final.json", {**obj, **reply.record()})
    files.log_event(qid, "converge", **reply.record())
    q = files.read_json(files.qdir(qid) / "question.json")
    q["stage"] = "passages_selected"
    files.write_json(files.qdir(qid) / "question.json", q)
    return obj


def run(qid):
    select(qid)
    return converge(qid)
