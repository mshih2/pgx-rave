"""GT claims: decomposition, coverage review and curation.

  decompose  the Verifier's own decomposition (two passes and deduplication), so GT claims are
             made exactly as answer claims are -> claims_decomposed.json
  coverage   the two LLM readers each check, independently, that the claims cover the GT exactly:
             a fact the GT states that no claim carries, a claim the GT does not support, two
             facts in one claim or one fact split across claims. The settler keeps the findings
             both readers made (it may overrule one) -> coverage_*.json, claims_settled.json
  curate     config.CURATOR curates the claims against the GT and its sources under the written
             plan in prompts/claim_curation.txt, starting from the settled findings, for up to
             config.CURATION_ROUNDS rounds or until a round changes nothing -> claims.json
"""
import json
import logging

from . import config, files, llm
from .review import material

log = logging.getLogger(__name__)

OPS = {"drop", "rewrite", "split", "merge", "add"}


def numbered(claims):
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims))


def decompose(qid):
    from pgxverifier.verifier import Verifier
    q = files.read_json(files.qdir(qid) / "question.json")
    record = Verifier().claims(q["question"], files.read_text(files.qdir(qid) / "gt.md"))
    files.write_json(files.qdir(qid) / "claims_decomposed.json", record)
    files.log_event(qid, "decompose", n_claims=len(record["claims"]), warnings=record["warnings"])
    return record["claims"]


def coverage(qid):
    claims = files.read_json(files.qdir(qid) / "claims_decomposed.json")["claims"]
    answer = files.read_text(files.qdir(qid) / "gt.md")
    prompt = files.prompt("claim_coverage.txt", answer=answer, n=len(claims),
                          claims=numbered(claims))
    reports = []
    for model, (obj, reply) in llm.ask_json_each(config.READERS, prompt).items():
        files.write_json(files.qdir(qid) / f"coverage_{files.tag(model)}.json", {**obj, **reply.record()})
        files.log_event(qid, "claim_coverage", **reply.record())
        reports.append(obj)
    readers = "\n\n".join(f"### Reader {i}\n```json\n{json.dumps(r, ensure_ascii=False, indent=1)}\n```"
                          for i, r in enumerate(reports, 1))
    settled, reply = llm.ask_json(config.SETTLER, files.prompt(
        "settle_claims.txt", answer=answer, claims=numbered(claims), readers=readers))
    files.write_json(files.qdir(qid) / "claims_settled.json", {**settled, **reply.record()})
    files.log_event(qid, "settle_claims", **reply.record())
    return settled


def apply_edits(claims, edits):
    """Apply one round's edits (claim numbers refer to the list the round was given).
    -> (new claims, rejected edits). An edit touching a claim another edit already touched is
    rejected rather than guessed at."""
    replace, inserts, touched, rejected = {}, {}, set(), []
    for e in edits:
        op, ns, new = e.get("op"), e.get("claims") or [], [str(x) for x in e.get("new") or []]
        if op not in OPS or any(not isinstance(n, int) or not 0 <= n < len(claims) for n in ns):
            rejected.append({**e, "why_rejected": "unknown op or claim number"})
            continue
        if op == "add":
            after = ns[0] if ns else -1
            inserts.setdefault(after, []).extend(new)
            continue
        if not ns or set(ns) & touched or (op != "drop" and not new):
            rejected.append({**e, "why_rejected": "no claims, overlapping edit or empty text"})
            continue
        touched |= set(ns)
        if op == "drop":
            for n in ns:
                replace[n] = []
        elif op == "merge":
            replace[ns[0]] = new
            for n in ns[1:]:
                replace[n] = []
        else:                                   # rewrite, split
            replace[ns[0]] = new
    out = list(inserts.get(-1, []))
    for i, c in enumerate(claims):
        out += replace.get(i, [c]) + inserts.get(i, [])
    return out, rejected


def curate(qid):
    q = files.read_json(files.qdir(qid) / "question.json")
    claims = files.read_json(files.qdir(qid) / "claims_decomposed.json")["claims"]
    settled = files.read_json(files.qdir(qid) / "claims_settled.json", default={})
    findings = json.dumps(settled.get("actions") or [], ensure_ascii=False, indent=1)
    answer = files.read_text(files.qdir(qid) / "gt.md")
    rounds, stable = [], False
    for n in range(1, config.CURATION_ROUNDS + 1):
        prompt = files.prompt("claim_curation.txt", material=material(q), answer=answer,
                              claims=numbered(claims), n=len(claims), round=n,
                              findings=findings if n == 1 else "(none; this is a later round)")
        obj, reply = llm.ask_json(config.CURATOR, prompt)
        edits = obj.get("edits") or []
        new, rejected = apply_edits(claims, edits)
        rounds.append({"round": n, "edits": edits, "rejected": rejected,
                       "notes_for_human_check": obj.get("notes_for_human_check") or [],
                       "n_before": len(claims), "n_after": len(new), **reply.record()})
        files.log_event(qid, f"curate_r{n}", n_edits=len(edits), **reply.record())
        if not edits:
            stable = True
            break
        claims = new
    files.write_json(files.qdir(qid) / "claims.json",
                     {"claims": claims, "stable": stable, "rounds": rounds})
    q["stage"] = "claims_curated"
    files.write_json(files.qdir(qid) / "question.json", q)
    return claims


def run(qid):
    decompose(qid)
    coverage(qid)
    return curate(qid)
