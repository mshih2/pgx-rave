"""Guideline and supplement text, read through PGxRoute's retrieval corpus (CORPUS_DIR).

The passages a hard GT is written from come from the same units, document tags and supersession
rule as the Full pipeline's retrieval scope, so a GT never rests on text the pipeline cannot reach
or on a guideline version a newer one replaces.

A unit is identified as "<document file stem>#<kind><n>", n counting that kind within the
document in corpus order; the corpus is frozen, so the identifiers are stable.
"""
import functools
from pathlib import Path


def corpus_dir():
    from pgxroute import config as route_config
    return Path(route_config.CORPUS_DIR)


def available():
    d = corpus_dir()
    return d.is_dir() and any(d.glob("*.json"))


@functools.lru_cache(maxsize=1)
def _loaded():
    from pgxroute.steps import retrieval
    texts, meta, docs, kinds = retrieval.load_units()
    doc_of, superseded = retrieval._superseded_docs()
    units, counts = [], {}
    for i, (text, m, doc, kind) in enumerate(zip(texts, meta, docs, kinds)):
        stem = Path(doc).stem
        n = counts[(stem, kind)] = counts.get((stem, kind), 0) + 1
        units.append({"uid": f"{stem}#{kind}{n}", "index": i, "doc": stem, "kind": kind,
                      "genes": m["genes"], "drugs": m["drugs"], "text": text,
                      "superseded": doc_of.get(doc) in superseded})
    return units


def units(include_superseded=False):
    return [u for u in _loaded() if include_superseded or not u["superseded"]]


def by_uid():
    return {u["uid"]: u for u in _loaded()}


def _covers(u, gene, drug):
    return gene.upper() in u["genes"] and any(drug.lower() in d for d in u["drugs"])


def docs_covering(gene, drug):
    """Current documents tagged with both the gene and the drug."""
    return {u["doc"] for u in units() if _covers(u, gene, drug)}


def scope(genes, drugs):
    """Current units of documents tagged with one of the genes and one of the drugs (the Full
    pipeline's scope, restricted to drug-matching documents)."""
    from pgxroute.steps import retrieval
    rows = [{"Gene Name": g, "Drug Name": d} for g in genes for d in drugs]
    allowed = set(retrieval.scope_indices(rows))
    return [u for u in units()
            if u["index"] in allowed and any(_covers(u, g, d) for g in genes for d in drugs)]
