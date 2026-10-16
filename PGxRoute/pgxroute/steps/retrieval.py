"""Retrieval step of the Full pipeline.

  scope   documents whose genes overlap the SI genes; among them, those whose drugs contain an
          SI drug, used only if that keeps >= 8 units (else the gene set); superseded
          documents (doc_manifest.json `superseded_by`) are then removed
  search  BGE-M3 hybrid score (dense + 0.3 x sparse) of the SI text against the scoped units;
          top 60 form the pool
  rerank  listwise reranker (Qwen3.8-27B, thinking off) reads the original question and the
          pool, and returns the 8 most useful units

The corpus is frozen: at load time its stamp and every file hash are checked against
_provenance.json and config.EXPECTED_CORPUS_VERSION, and any mismatch raises (VERIFY_CORPUS=0
turns the check off for a corpus built elsewhere).

Run `python -m pgxroute.steps.retrieval --verify` from the repository root to check a corpus copy.
"""
import glob
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

from .. import config

log = logging.getLogger(__name__)

CORPUS = str(config.CORPUS_DIR)
PROMPT = (config.PROMPT_DIR / "rerank.txt").read_text(encoding="utf-8")

_verified = None
_units = None
_tokenizer = None
_superseded = None


def _is_content_file(basename):
    return not basename.startswith(("doc_manifest", "_"))


def verify_corpus():
    """Recompute the corpus stamp and check every recorded file. Returns the stamp.

    With VERIFY_CORPUS=0 nothing is checked; the stamp is then a hash of the content files,
    which keys the embedding cache."""
    global _verified
    if _verified is not None:
        return _verified
    if not config.VERIFY_CORPUS:
        h = hashlib.sha256()
        for f in sorted(glob.glob(CORPUS + "/*.json")):
            h.update(os.path.basename(f).encode() + open(f, "rb").read())
        _verified = h.hexdigest()
        return _verified
    prov = os.path.join(CORPUS, "_provenance.json")
    if not os.path.exists(prov):
        raise RuntimeError(f"corpus check: {prov} not found")
    files = json.load(open(prov))["files"]
    stamp = hashlib.sha256("\n".join(f"{n}  {files[n]['sha256']}"
                                     for n in sorted(files)).encode()).hexdigest()
    if stamp != config.EXPECTED_CORPUS_VERSION:
        raise RuntimeError(f"corpus check: stamp {stamp[:8]} != expected "
                           f"{config.EXPECTED_CORPUS_VERSION[:8]} ({len(files)} files)")
    disk = {os.path.basename(f) for f in glob.glob(CORPUS + "/*.json")}
    missing = sorted({b for b in disk if _is_content_file(b)} - set(files))
    orphans = sorted(set(files) - disk)
    bad = [n for n in sorted(files)
           if hashlib.sha256(open(os.path.join(CORPUS, n), "rb").read()).hexdigest()
           != files[n]["sha256"]]
    if missing or orphans or bad:
        raise RuntimeError(f"corpus check: {len(missing)} unrecorded, {len(orphans)} missing, "
                           f"{len(bad)} changed files (e.g. {(missing + orphans + bad)[:3]})")
    _verified = stamp
    return stamp


def load_units():
    """-> (texts, metadata, doc file, kind) for every prose, table and figure unit."""
    global _units
    if _units is not None:
        return _units
    verify_corpus()
    U, M, DOC, KIND = [], [], [], []
    for f in sorted(glob.glob(CORPUS + "/*.json")):
        b = os.path.basename(f)
        if not _is_content_file(b):
            continue
        d = json.load(open(f))
        m = {"genes": [g.upper() for g in d.get("gene", [])],
             "drugs": [x.lower() for x in d.get("drug", [])]}
        for pc in d.get("prose_chunks", []):
            if pc.get("self_contained_text"):
                U.append(pc["self_contained_text"]); M.append(m); DOC.append(b); KIND.append("prose")
        for blk in d.get("blocks", []):
            for tc in (blk.get("table_chunks") or []):
                if tc.get("self_contained_text"):
                    U.append(tc["self_contained_text"]); M.append(m); DOC.append(b); KIND.append("table")
            if blk.get("type") == "figure" and blk.get("self_contained_text"):
                U.append(blk["self_contained_text"]); M.append(m); DOC.append(b); KIND.append("fig")
    _units = (U, M, DOC, KIND)
    return _units


def _superseded_docs():
    """-> (file -> doc_name, {superseded doc_name: successor}).

    A manifest entry is matched to a corpus document by exact doc_id, otherwise by the one
    document with the same source, gene set and drug set whose name contains the manifest year.
    Unmatched or ambiguous entries are kept (never silently removed)."""
    global _superseded
    if _superseded is not None:
        return _superseded
    doc_of, meta_of = {}, {}
    for f in sorted(glob.glob(CORPUS + "/*.json")):
        b = os.path.basename(f)
        if not _is_content_file(b):
            continue
        d = json.load(open(f))
        doc_of[b] = d["doc_name"]
        meta_of[d["doc_name"]] = (d.get("source"),
                                  frozenset(g.upper() for g in d.get("gene", [])),
                                  frozenset(x.lower() for x in d.get("drug", [])))
    manifest = json.load(open(os.path.join(CORPUS, "doc_manifest.json")))
    ids = {e["doc_id"] for e in manifest}

    def corpus_doc(e):
        if e.get("not_ingested"):
            return None
        if e["doc_id"] in meta_of:
            return e["doc_id"]
        key = (e.get("source"), frozenset(g.upper() for g in e.get("genes", [])),
               frozenset(x.lower() for x in e.get("drugs", [])))
        year = str(e.get("year"))
        c = [n for n, k in meta_of.items() if k == key and year in n.split("_") and n not in ids]
        return c[0] if len(c) == 1 else None

    excluded, unmatched = {}, []
    for e in manifest:
        if not e.get("superseded_by"):
            continue
        n = corpus_doc(e)
        if n is None:
            unmatched.append(e["doc_id"])
        else:
            excluded[n] = e["superseded_by"]
    log.info("%d of %d corpus documents excluded as superseded", len(excluded), len(meta_of))
    if unmatched:
        log.info("superseded in the manifest but not matched (kept): %s", unmatched)
    _superseded = (doc_of, excluded)
    return _superseded


def scope_indices(scope_rows):
    U, M, DOC, _ = load_units()
    rows = scope_rows or []
    genes = {str(x.get("Gene Name", "")).upper() for x in rows if x.get("Gene Name")}
    drugs = {str(x.get("Drug Name", "")).lower() for x in rows if x.get("Drug Name")}
    gi = [i for i, m in enumerate(M) if set(m["genes"]) & genes]
    di = [i for i in gi if any(dr in dj for dr in drugs for dj in M[i]["drugs"])]
    chosen = di if len(di) >= config.MIN_DRUG_SCOPE_UNITS else gi
    doc_of, excluded = _superseded_docs()
    kept = [i for i in chosen if doc_of[DOC[i]] not in excluded]
    if chosen and not kept:
        log.info("supersession filter would empty the scope; unfiltered scope used")
        return chosen
    return kept


# ------------------------------------------------------------------- BGE-M3 hybrid scoring
def _emb_cache_path(stamp):
    os.makedirs(config.EMB_CACHE_DIR, exist_ok=True)
    return os.path.join(config.EMB_CACHE_DIR, f"bgem3_{stamp[:16]}.pkl")


def _rank(query_text, scope):
    """Scope indices ordered by hybrid score, best first. Unit embeddings are cached on disk
    per corpus stamp."""
    import pickle
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel
    stamp = verify_corpus()
    U = load_units()[0]
    cache, enc = _emb_cache_path(stamp), None
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            c = pickle.load(fh)
        if c.get("stamp") == stamp and c.get("n") == len(U):
            enc = c
    model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=True, devices=config.EMBED_DEVICE)
    if enc is None:
        e = model.encode(U, batch_size=8, max_length=8192,
                         return_dense=True, return_sparse=True, return_colbert_vecs=False)
        enc = {"stamp": stamp, "n": len(U), "dense": np.asarray(e["dense_vecs"], np.float32),
               "sparse": list(e["lexical_weights"])}
        with open(cache, "wb") as fh:
            pickle.dump(enc, fh)
    q = model.encode([query_text], batch_size=1, max_length=8192,
                     return_dense=True, return_sparse=True, return_colbert_vecs=False)
    qd, qs = np.asarray(q["dense_vecs"][0], np.float32), q["lexical_weights"][0]
    score = enc["dense"][scope] @ qd + config.SPARSE_WEIGHT * np.array(
        [model.compute_lexical_matching_score(qs, enc["sparse"][i]) for i in scope], np.float32)
    return [scope[k] for k in np.argsort(-score)]


def rank(query_text, scope):
    if not config.EMBED_PYTHON:
        return _rank(query_text, scope)
    with tempfile.TemporaryDirectory(prefix="pgx_rank_") as td:
        job, out = os.path.join(td, "job.json"), os.path.join(td, "out.json")
        json.dump({"query_text": query_text, "scope": scope}, open(job, "w"))
        # Run as a module from the repository root so the package imports resolve.
        p = subprocess.run([config.EMBED_PYTHON, "-m", __spec__.name, "--rank", job, out],
                           cwd=config.REPO_ROOT, capture_output=True, text=True,
                           timeout=config.EMBED_TIMEOUT)
        if p.returncode != 0:
            raise RuntimeError(f"ranking subprocess failed (rc={p.returncode}): "
                               f"{(p.stderr or p.stdout)[-800:]}")
        return json.load(open(out))["order"]


# ------------------------------------------------------------------- listwise rerank
def _strip_header(t):
    """Drop a unit's first line when it is a bracketed header."""
    return t.split("\n", 1)[1].strip() if t.startswith("[") and "\n" in t else t.strip()


def _build_prompt(question, bodies, max_tokens):
    """Fill the prompt, shortening every passage by 12% per step until it fits the window."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_TOKENIZER)
    limit = config.RERANKER_CONTEXT_WINDOW - max_tokens - config.RERANKER_MARGIN
    cap = None
    for _ in range(30):
        cut = bodies if cap is None else [b if len(b) <= cap else b[:cap] for b in bodies]
        p = PROMPT.format(q=question, passages="\n".join(f"[{j+1}] {b}" for j, b in enumerate(cut)))
        n = len(_tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=False))
        if n <= limit:
            return p
        cap = int((cap or max(len(b) for b in bodies)) * 0.88)
    raise RuntimeError(f"rerank prompt does not fit at max_tokens={max_tokens}")


def _call_reranker(prompt, max_tokens):
    """-> (ranking text, finish_reason, usage). Temperature 0, think block stripped."""
    err = "?"
    for _ in range(3):
        try:
            body = json.dumps({"model": config.RERANKER_MODEL, "temperature": 0.0,
                               "max_tokens": max_tokens,
                               "chat_template_kwargs": {"enable_thinking": config.RERANKER_THINKING},
                               "messages": [{"role": "user", "content": prompt}]}).encode()
            req = urllib.request.Request(config.RERANKER_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=1800))
            ch = d["choices"][0]
            return (re.sub(r"^.*?</think>\s*", "", ch["message"]["content"], flags=re.S).strip(),
                    ch["finish_reason"], d.get("usage") or {})
        except Exception as e:
            err = str(e)[:90]
    return "", f"ERROR:{err}", {}


def retrieve(query_text, scope_rows, question):
    """-> {"context", "picks", "pool_size", "corpus_version", "usage"}.

    query_text: what BGE-M3 embeds (the SI text). question: what the reranker reads."""
    stamp = verify_corpus()
    U, _, DOC, KIND = load_units()
    out = {"context": "", "picks": [], "pool_size": 0, "corpus_version": stamp,
           "usage": {"input_tokens": 0, "output_tokens": 0}}
    scope = scope_indices(scope_rows)
    if not scope:
        log.info("empty retrieval scope (no SI gene matches the corpus)")
        return out
    pool = rank(query_text, scope)[:config.POOL_SIZE]
    bodies = [_strip_header(U[i]) for i in pool]

    first, *retries = config.RERANKER_MAX_TOKENS
    text, finish, usage = _call_reranker(_build_prompt(question, bodies, first), first)
    calls = [usage]
    for mt in retries:
        if finish != "length" or mt >= config.RERANKER_CONTEXT_WINDOW - config.RERANKER_MARGIN:
            break
        log.info("ranking truncated; retrying with max_tokens=%d", mt)
        text, finish, usage = _call_reranker(_build_prompt(question, bodies, mt), mt)
        calls.append(usage)
    for u in calls:
        out["usage"]["input_tokens"] += int(u.get("prompt_tokens") or 0)
        out["usage"]["output_tokens"] += int(u.get("completion_tokens") or 0)
    if finish != "stop":
        raise RuntimeError(f"reranker finish_reason={finish}; a truncated ranking is not used")

    picks = []
    for x in re.findall(r"\d+", text):
        v = int(x)
        if 1 <= v <= len(pool) and v not in picks:
            picks.append(v)
        if len(picks) >= config.TOP_N:
            break
    if len(picks) < config.TOP_N:      # kept as returned; never backfilled
        log.info("reranker returned %d of %d picks", len(picks), config.TOP_N)
    sel = [pool[v - 1] for v in picks]
    out.update(context="\n\n".join(f"[Evidence {j+1}] {U[i]}" for j, i in enumerate(sel)),
               picks=[{"doc": DOC[i], "kind": KIND[i]} for i in sel], pool_size=len(pool))
    return out


if __name__ == "__main__":
    if "--rank" in sys.argv:
        i = sys.argv.index("--rank")
        job = json.load(open(sys.argv[i + 1]))
        json.dump({"order": _rank(job["query_text"], job["scope"])}, open(sys.argv[i + 2], "w"))
    elif "--verify" in sys.argv:
        from collections import Counter
        s = verify_corpus()
        print(f"corpus_version {s}\nunits {len(load_units()[0])} {dict(Counter(load_units()[3]))}")
    else:
        print(__doc__)
