"""CPIC API step (API-only and Full pipelines).

Turns the SI's diplotypes into CPIC lookup keys, calls /rpc/recommendation_lookup, keeps the
rows for the SI's drugs, and formats them as plain-text blocks for the reasoner.
"""
import json
import logging

import requests

from .. import config

log = logging.getLogger(__name__)

# Optional fields rendered only when non-empty: (label, CPIC column).
_OPTIONAL_FIELDS = [
    ("Population", "population"),
    ("Implication", "implications"),
    ("Phenotype", "phenotypes"),
    ("Activity score", "activityscore"),
    ("Dosing", "dosinginformation"),
    ("Alternative drug available", "alternatedrugavailable"),
    ("Other guidance", "otherprescribingguidance"),
    ("Comment", "comments"),
]
_GENE_KEYED = {"implications", "phenotypes", "activityscore"}   # shown for tested genes only
_EMPTYISH = {"", "n/a", "none", "null", "{}", "[]"}


class CPICClient:
    def __init__(self, base=None):
        self.base = base or config.CPIC_BASE
        self._diplotypes = {}      # (gene, generesult) -> /diplotype rows

    def get(self, path, params=None):
        r = requests.get(f"{self.base}{path}", params=params or {},
                         headers={"Accept": "application/json"}, timeout=config.CPIC_TIMEOUT)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------ allele normalisation
    @staticmethod
    def _parse_diplotype(dip):
        s = str(dip)
        parts = [p.strip() for p in s.split("/")] if "/" in s else [s.strip()]
        out = {}
        for p in parts:
            if p:
                out[p] = out.get(p, 0) + 1
        return out

    def alleles_for_phenotype(self, gene, generesult):
        """Phenotype -> the alleles of a representative CPIC diplotype, for a report that gives
        a phenotype and no usable genotype."""
        key = (gene, generesult)
        if key not in self._diplotypes:
            try:
                self._diplotypes[key] = self.get("/diplotype", {
                    "genesymbol": f"eq.{gene}", "generesult": f"eq.{generesult}",
                    "select": "diplotype,lookupkey", "limit": "1000"})
            except Exception:
                self._diplotypes[key] = []
        rows = self._diplotypes[key]
        return self._parse_diplotype(rows[0].get("diplotype", "")) if rows else None

    def normalize(self, alleles):
        """{gene: {allele: count}} -> CPIC lookup keys.

        si_builder already writes every allele, serotype suffix and phenotype-derived genotype
        in CPIC's own names; a single reported copy is completed with the reference allele."""
        if not alleles:
            return alleles
        out = {}
        for gene, tokens in alleles.items():
            g = str(gene)
            if not isinstance(tokens, dict):
                out[gene] = tokens
                continue
            resolved = {str(t).strip(): cnt for t, cnt in tokens.items()}
            out[g] = resolved
        # Genes CPIC keys together are completed by si_builder from the table's gene groups,
        # before the lookup reaches here.
        return {g: v for g, v in out.items() if v or not isinstance(v, dict)}

    # ------------------------------------------------------------ lookup + formatting
    def recommendation_lookup(self, lookup):
        if not lookup:
            return []
        return self.get("/rpc/recommendation_lookup", {
            "diplotypelookup": json.dumps(lookup, separators=(",", ":"))})

    def search(self, scope_rows, alleles, tested_genes=None, supplied_genes=None,
               inferred_genes=None):
        """CPIC recommendations for the patient's diplotypes, as text for the reasoner.

        tested_genes names the genes the question itself reports. Per-gene fields are shown for
        those genes only: CPIC completes a gene it was not given with a default phenotype, and
        printing that would read as a result the patient does not have. Defaults to the genes
        in alleles.

        supplied_genes names genes the lookup carries because a named drug's CPIC key needs
        them, not because the patient was tested. They are listed in the output, so a
        recommendation that rests on one of them can be read as resting on an assumption.

        inferred_genes maps a gene to the phenotype the question gave for it, where the question
        gave no usable genotype. The lookup then stands on one CPIC diplotype that yields that
        phenotype, so a per-gene value that the phenotype does not fix is left out: it would
        describe that stand-in rather than the patient."""
        drugs = {r.get("Drug Name", "").strip().lower() for r in scope_rows if isinstance(r, dict)}
        try:
            rows = self.recommendation_lookup(self.normalize(alleles))
        except Exception as e:
            log.warning("recommendation_lookup failed: %s", e)
            rows = []
        drugs = {d for d in drugs if d}
        if drugs:                  # no drug named -> keep every row the genotype matched
            rows = [r for r in rows if r.get("drugname", "").lower() in drugs]
        tested = {str(g).upper() for g in (tested_genes if tested_genes is not None else (alleles or {}))}

        inferred = {str(g).upper(): str(p) for g, p in (inferred_genes or {}).items()}
        # For a gene known only by its phenotype, a value the phenotype does not fix belongs to
        # the stand-in diplotype. CPIC lists the values a phenotype leaves open, so they are
        # recognised rather than guessed at by field name.
        from . import si_builder
        open_values = {}
        for g, pheno in inferred.items():
            scores = (si_builder.table()['genes'].get(g, {})
                      .get('phenotype_activity_scores', {}).get(pheno, []))
            open_values[g] = {str(x).strip().lower() for x in scores}

        def fmt(v, gene_filter=None):
            if isinstance(v, dict):
                items = []
                for k, vv in v.items():
                    if not vv or (gene_filter is not None and str(k).upper() not in gene_filter):
                        continue
                    if str(vv).strip().lower() in open_values.get(str(k).upper(), ()):
                        continue
                    items.append((k, vv))
                return "; ".join(f"{k}: {vv}" for k, vv in items)
            return str(v).strip() if v is not None else ""

        lines = []
        for g, pheno in sorted(inferred.items()):
            if not lines:
                lines.append("=== GENES GIVEN ONLY AS A PHENOTYPE ===")
                lines.append("(Notes from this lookup tool, not text printed by CPIC.)")
            val = "; ".join(sorted((alleles or {}).get(g, {}))) or "none"
            lines.append(f"{g} \u2014 the question gives the phenotype '{pheno}' and no genotype "
                         f"this tool can use. CPIC is keyed on genotypes, so the lookup below "
                         f"used '{val}', one CPIC diplotype that yields that phenotype. Several "
                         f"genotypes yield it, so only the phenotype is reported for {g}; any "
                         f"other per-gene value would describe that stand-in, not this patient.")
        for g in sorted(supplied_genes or ()):
            if not lines:
                lines.append("=== GENES NOT TESTED ===")
                lines.append("(Notes from this lookup tool, not text printed by CPIC.)")
            val = "; ".join(sorted((alleles or {}).get(g, {})))
            lines.append(f"{g} \u2014 the question reports no result for this gene. CPIC gives no "
                         f"recommendation for the drug below without it, so it was looked up as "
                         f"'{val}', the value CPIC uses for a gene carrying no variant. Any "
                         f"recommendation below that depends on {g} rests on that assumption; "
                         f"this patient's {g} status is unknown.")
        lines.append(("\n" if lines else "") + "=== RECOMMENDATIONS ===")
        seen, n = set(), 0
        for r in rows:
            key = json.dumps(r, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            n += 1
            drug, guideline = fmt(r.get("drugname")), fmt(r.get("guidelinename"))
            rec, cls = fmt(r.get("drugrecommendation")), fmt(r.get("classification"))
            lines.append(f"- Drug: {drug}" + (f"  (Guideline: {guideline})" if guideline else ""))
            if rec:
                lines.append(f"  Recommendation: {rec}" + (f"  [{cls}]" if cls else ""))
            elif cls:
                lines.append(f"  Classification: {cls}")
            for label, col in _OPTIONAL_FIELDS:
                val = fmt(r.get(col), tested if col in _GENE_KEYED else None)
                if val and val.lower() not in _EMPTYISH:
                    lines.append(f"  {label}: {val}")
        if n == 0:
            lines.append("(none)")
        return "\n".join(lines)


client = CPICClient()
