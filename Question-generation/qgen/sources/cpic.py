"""CPIC API information: a snapshot of the tables the generator reads, the candidate rows,
and a question's CPIC section.

A candidate is one row of the `recommendation` table: a drug and the exact genotype combination
CPIC keys its advice on. Rows that key on several genes stay one candidate, so no gene-phenotype
combination CPIC does not publish is ever proposed.

The CPIC section of a question is what PGxRoute's API-only pipeline receives for it: the
structured input's lookup keys sent to the recommendation lookup, rows kept for its drugs. GT is
built from that text, so every API fact in a GT is one the pipelines can be given.
"""
import json
import logging
import time
from collections import defaultdict

import requests

from .. import config, files
from . import corpus

log = logging.getLogger(__name__)

# Recommendation fields, in the order a candidate row is shown to a model.
ROW_FIELDS = [
    ("Recommendation", "drugrecommendation"),
    ("Classification", "classification"),
    ("Implication", "implications"),
    ("Phenotype", "phenotypes"),
    ("Activity score", "activityscore"),
    ("Population", "population"),
    ("Dosing", "dosinginformation"),
    ("Alternative drug available", "alternatedrugavailable"),
    ("Other guidance", "otherprescribingguidance"),
    ("Comment", "comments"),
]


def _get(table, params, headers=None, timeout=300):
    r = requests.get(f"{config.CPIC_BASE}/{table}", params=params, timeout=timeout,
                     headers={"Accept": "application/json", **(headers or {})})
    r.raise_for_status()
    return r


def snapshot():
    """Fetch the tables in config.CPIC_TABLES into WORK_DIR/cpic_snapshot/, with row counts
    checked against the API's exact count."""
    out = files.work("cpic_snapshot")
    meta = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base_url": config.CPIC_BASE,
            "tables": {}}
    for table, select in config.CPIC_TABLES.items():
        expected = int(_get(table, {"select": "*"}, {"Prefer": "count=exact", "Range": "0-0"},
                            timeout=60).headers["Content-Range"].split("/")[-1])
        rows = _get(table, {"select": select, "limit": 500_000}).json()
        files.write_json(out / f"{table}.json", rows)
        meta["tables"][table] = {"rows": len(rows), "expected_rows": expected, "select": select,
                                 "complete": len(rows) == expected}
        log.info("%-16s %7d rows%s", table, len(rows),
                 "" if len(rows) == expected else f"  (expected {expected})")
    files.write_json(out / "snapshot_meta.json", meta)
    bad = [t for t, m in meta["tables"].items() if not m["complete"]]
    if bad:
        raise RuntimeError(f"incomplete CPIC tables: {bad}")
    return meta


def _table(name):
    p = files.work("cpic_snapshot", f"{name}.json")
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run `python run.py snapshot` first")
    return files.read_json(p)


def build_candidates():
    """recommendation rows -> WORK_DIR/candidates.jsonl."""
    recs, dips, drugs = _table("recommendation"), _table("diplotype"), _table("drug")
    pairs, guidelines = _table("pair"), _table("guideline")
    drug_name = {d["drugid"]: d["name"] for d in drugs}
    guideline = {g["id"]: g for g in guidelines}
    level = {(p["genesymbol"], p["drugid"]): p.get("cpiclevel")
             for p in pairs if not p.get("removed")}
    examples = defaultdict(list)
    for d in dips:
        for gene, val in (d.get("lookupkey") or {}).items():
            examples[(gene, str(val))].append(d["diplotype"])
    have_corpus = corpus.available()
    if not have_corpus:
        log.warning("no corpus at %s: corpus coverage of the candidates is left unknown",
                    corpus.corpus_dir())

    out = []
    for r in recs:
        key = r.get("lookupkey") or {}
        genes = sorted(key)
        drug = drug_name.get(r["drugid"], r["drugid"])
        docs = (sorted({doc for g in genes for doc in corpus.docs_covering(g, drug)})
                if have_corpus else None)
        out.append({
            "cand_id": f"rec{r['id']}",
            "drug": drug,
            "genes": genes,
            "lookupkey": key,
            "population": r.get("population"),
            "classification": r.get("classification"),
            "guideline": (guideline.get(r.get("guidelineid")) or {}).get("name"),
            "cpiclevel_by_gene": {g: level.get((g, r["drugid"])) for g in genes},
            "example_diplotypes": {g: examples.get((g, str(key[g])), [])
                                   [:config.MAX_EXAMPLE_DIPLOTYPES] for g in genes},
            "corpus_docs": docs,
            "row": {col: r.get(col) for _, col in ROW_FIELDS},
        })
    files.write_jsonl(files.work("candidates.jsonl"), out)
    log.info("%d candidates written", len(out))
    return out


def candidates():
    rows = files.read_jsonl(files.work("candidates.jsonl"))
    if not rows:
        raise FileNotFoundError("no candidates; run `python run.py candidates` first")
    return rows


def _value(v):
    if v in (None, "", {}, []):
        return ""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, dict):
        return "; ".join(f"{k}: {t}" for k, t in v.items() if t not in (None, ""))
    return str(v)


def render_candidate(c):
    """A candidate row as plain text, as CPIC prints its fields."""
    lines = [f"Drug: {c['drug']}",
             f"Genes and lookup keys: {json.dumps(c['lookupkey'], ensure_ascii=False)}",
             "Example diplotypes: " + "; ".join(f"{g}: {', '.join(v) or '(none listed)'}"
                                                for g, v in c["example_diplotypes"].items())]
    for label, col in ROW_FIELDS:
        v = _value(c["row"].get(col))
        if v:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def cpic_section(question):
    """The CPIC text PGxRoute's API-only pipeline receives for this question, with the structured
    input it was looked up from."""
    from pgxroute.steps import cpic_api, si_builder
    si = si_builder.build(question, cpic_api.client)
    rows = ([{"Gene Name": g, "Drug Name": d} for g in si["genes"] for d in si["drugs"]]
            or [{"Gene Name": g} for g in si["genes"]]
            or [{"Drug Name": d} for d in si["drugs"]])
    text = (cpic_api.client.search(rows, si["lookup"], tested_genes=si["genes"],
                                   supplied_genes=si["supplied_genes"],
                                   inferred_genes=si["phenotypes"]) if rows else "")
    return {"si_text": si["si_text"], "genes": si["genes"], "variants": si["variants"],
            "drugs": si["drugs"], "scope_rows": rows, "cpic_text": text}
