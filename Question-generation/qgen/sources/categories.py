"""Information categories: sort CPIC guideline and supplement text into the categories of
prompts/categories.json, tag the bank's questions with the same categories, and rank categories
by how sparsely the bank covers them.

  categorize()   corpus units -> WORK_DIR/unit_categories.jsonl (resumable)
  tag()          bank questions without tags -> question.json "tags"
  coverage()     -> WORK_DIR/coverage.json: per category, current units and hard questions
  sparse()       categories ordered from least to most covered
  collides()     the anti-collision rule for a new candidate
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from .. import config, files, llm
from . import corpus

log = logging.getLogger(__name__)

BATCH_UNITS = 30


def _taxonomy_text():
    return "\n".join(f"- {c['name']}: {c['definition']}" for c in files.taxonomy())


def _names():
    return {c["name"] for c in files.taxonomy()}


def _label_batch(batch):
    body = "\n\n".join(f"### {u['uid']}\n```text\n{u['text'].strip()}\n```" for u in batch)
    prompt = files.prompt("categorize.txt", categories=_taxonomy_text(), units=body)
    obj, reply = llm.ask_json(config.CATEGORIZER, prompt)
    names, rows = _names(), []
    for u in batch:
        got = obj.get(u["uid"]) or {}
        cats = [c for c in got.get("categories") or [] if c in names]
        unknown = [c for c in got.get("categories") or [] if c not in names]
        if unknown:
            log.warning("%s: categories outside the list dropped: %s", u["uid"], unknown)
        rows.append({"uid": u["uid"], "doc": u["doc"], "categories": cats,
                     "printed_negative": bool(got.get("printed_negative")),
                     **reply.record()})
    return rows


def categorize(workers=None):
    """Label every current corpus unit with its information categories. Units are sent one
    document at a time (in batches of BATCH_UNITS), so a model sees each unit in context."""
    path = files.work("unit_categories.jsonl")
    done = {r["uid"]: r for r in files.read_jsonl(path)}
    todo = [u for u in corpus.units() if u["uid"] not in done]
    by_doc = {}
    for u in todo:
        by_doc.setdefault(u["doc"], []).append(u)
    batches = [v[i:i + BATCH_UNITS] for v in by_doc.values() for i in range(0, len(v), BATCH_UNITS)]
    log.info("%d units to label in %d batches (%d already labelled)", len(todo), len(batches), len(done))
    with ThreadPoolExecutor(workers or config.WORKERS) as ex:
        for rows in ex.map(_label_batch, batches):
            for r in rows:
                done[r["uid"]] = r
            files.write_jsonl(path, list(done.values()))
    return list(done.values())


def unit_categories():
    rows = files.read_jsonl(files.work("unit_categories.jsonl"))
    if not rows:
        raise FileNotFoundError("no unit categories; run `python run.py categorize` first")
    return {r["uid"]: r for r in rows}


def _tag_one(q):
    gt = files.qdir(q["qid"]) / "gt.md"
    prompt = files.prompt("tag_question.txt", categories=_taxonomy_text(),
                          facets=files.facets_text(), question=q["question"],
                          answer=files.read_text(gt) if gt.exists() else "(not written yet)")
    obj, reply = llm.ask_json(config.CATEGORIZER, prompt)
    names = _names()
    q["tags"] = {"categories": [c for c in obj.get("categories") or [] if c in names],
                 "facets": obj.get("facets") or [],
                 "answer_direction": obj.get("answer_direction"),
                 "population": obj.get("population"),
                 "tagged_by": reply.model}
    files.write_json(files.qdir(q["qid"]) / "question.json", q)
    return q


def tag(workers=None):
    """Tag every PGx question that has no tags yet (generated questions are tagged when made)."""
    todo = [q for q in files.questions() if q["tier"] != "common" and not q.get("tags")]
    with ThreadPoolExecutor(workers or config.WORKERS) as ex:
        list(ex.map(_tag_one, todo))
    log.info("%d questions tagged", len(todo))


def coverage():
    units = unit_categories()
    current = {u["uid"] for u in corpus.units()}
    hard = [q for q in files.questions("hard")]
    untagged = [q["qid"] for q in hard if not q.get("tags")]
    if untagged:
        raise RuntimeError(f"untagged hard questions {untagged}; run `python run.py tag` first")
    out = {}
    for c in files.taxonomy():
        name = c["name"]
        out[name] = {
            "units": sum(1 for uid, r in units.items() if uid in current and name in r["categories"]),
            "documents": len({r["doc"] for uid, r in units.items()
                              if uid in current and name in r["categories"]}),
            "hard_questions": sum(1 for q in hard if name in q["tags"]["categories"]),
        }
    files.write_json(files.work("coverage.json"), out)
    return out


def sparse(cov=None):
    """Categories with corpus text, least covered by hard questions first."""
    cov = cov or files.read_json(files.work("coverage.json"))
    rows = [(n, v) for n, v in cov.items() if v["units"] > 0]
    rows.sort(key=lambda nv: (nv[1]["hard_questions"], -nv[1]["units"]))
    return [n for n, _ in rows]


def _signature(tags):
    return (tuple(sorted(tags.get("categories") or [])), tuple(sorted(tags.get("facets") or [])),
            tags.get("answer_direction"), tags.get("population"))


def collides(tier, genes, drug, cand_id=None, tags=None):
    """-> the reason a new question would repeat one already in the bank, or None.

    hard: the (gene, drug) pair is taken by a hard question, or the signature (categories,
          facets, answer direction, population) equals a hard question's.
    intermediate: the same CPIC recommendation row is already used.
    """
    drug = (drug or "").lower()
    for q in files.questions(tier):
        if tier == "hard":
            if drug in [d.lower() for d in q.get("drugs") or []] and set(genes) & set(q.get("genes") or []):
                return f"(gene, drug) pair taken by {q['qid']}"
            if tags and q.get("tags") and _signature(tags) == _signature(q["tags"]):
                return f"same signature as {q['qid']}"
        elif cand_id and q.get("cand_id") == cand_id:
            return f"CPIC row {cand_id} used by {q['qid']}"
    return None
