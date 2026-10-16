"""Citation step of the Full pipeline (deterministic code, no model).

Strips citation identifiers the model wrote (PMIDs, DOIs, URLs, author-et-al, journal
references) and appends the link of the CPIC guideline for the question's first drug. CPIC
names each guideline after the genes and drug class it covers and links every drug to it, so
the guideline is read from the table.
"""
import re

from . import si_builder

_TOKENS = [
    re.compile(r"(?i)\bPMID:?\s*\d{6,9}\b"),
    re.compile(r"(?i)\bPMC(?:ID)?:?\s*PMC?\d{6,8}\b"),
    re.compile(r"(?i)\bDOI:?\s*10\.\d{4,9}/[^\s,);\]]+"),
    re.compile(r"https?://\S+"),
    re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s*(?:,|and|&)\s*[A-Z][A-Za-z'\-]+)*\s+et\s+al\.?(?:,?\s*\d{4})?"),
    re.compile(r"\b\d{1,4}\(\d{1,3}\):\s?\d+[\-–]\d+\b"),         # vol(issue):pages
]
# A parenthetical only when it contains a citation marker (not "(CPIC strength: Strong)").
_PAREN_CITE = re.compile(
    r"\([^()]*?(?:\bet al\.?|\bPMID\b|\bPMC\d|\bDOI\b|https?://|Clin(?:ical)?\s+Pharmacol|"
    r"Proc\.?\s*Natl|N\s*Engl\s*J|Lancet|J\s*Clin\s*Oncol)[^()]*\)", re.I)


def guideline_for(genes, drug):
    """(name, url) of the CPIC guideline for a drug, or, when the question names no drug CPIC
    covers, the one guideline whose genes are exactly the question's. None if nothing fits."""
    table = si_builder.table()
    d = (drug or "").strip().lower()
    hit = table["drug_guideline"].get(d) or table["drug_guideline"].get(
        table["drugs"].get(d, "").lower())
    if hit:
        return hit["name"], hit["url"]
    genes_u = {str(g).upper() for g in (genes or [])}
    if genes_u:
        named = {g["name"]: g for g in table["drug_guideline"].values()}
        cands = [g for name, g in named.items()
                 if genes_u and genes_u <= {w.strip(" ,") for w in name.replace(" and ", " ").split()}]
        if len(cands) == 1:
            return cands[0]["name"], cands[0]["url"]
    return None


def strip_citations(text):
    """-> (text without citation identifiers, removed spans)."""
    removed = [m.group(0) for m in _PAREN_CITE.finditer(text)]
    t = _PAREN_CITE.sub("", text)
    for pat in _TOKENS:
        removed += [m.group(0) for m in pat.finditer(t)]
        t = pat.sub("", t)
    t = re.sub(r"\(\s*[;,]?\s*\)", "", t)
    t = re.sub(r"\[\s*[;,]?\s*\]", "", t)
    t = re.sub(r"\s+([.,;:)\]])", r"\1", t)
    t = re.sub(r"\(\s*,\s*", "(", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip(), removed


def attach_citation(text, genes, drug):
    """-> (final text, removed spans, (guideline name, url) or None)."""
    clean, removed = strip_citations(text)
    gl = guideline_for(genes, drug)
    if gl:
        clean = f"{clean} [Provenance: follows the CPIC {gl[0]} guideline ({gl[1]}).]"
    return clean, removed, gl
