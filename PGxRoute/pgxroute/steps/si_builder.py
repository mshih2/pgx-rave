"""Build the structured input (SI) of a question by lookup against the CPIC table.

The table (data/cpic_lookup.json) holds every gene, allele, drug and phenotype name of the
CPIC API, with the aliases CPIC itself prints for them. Nothing here is matched loosely: a
gene, allele or drug is recognised only if the question names it exactly as CPIC does (up to
case, the aliases in the table, and the normalisations below). Questions are assumed to name
genotypes correctly.

Each gene is read in its own clause: the text from the gene's name to the next gene the
question names (at most _WINDOW characters), so one gene's alleles never leak into another's.
In that clause the genotype is taken as, in order: a diplotype 'X/Y', one allele with
'positive'/'negative', one allele alone, a phenotype name.
A gene named with an explicit "no variant" takes the gene's CPIC reference genotype; a gene
named without any genotype (an untested panel, say) gets none, since CPIC cannot be asked
about a genotype the question does not give. One reported allele on an autosomal gene is
completed with the reference
allele, as CPIC's lookup needs two; on an X-linked gene a single call stands (hemizygous).
A variant name CPIC gives to exactly one gene (e.g. 'm.1555A>G') identifies that gene even
when the question does not print the symbol. Genes CPIC keys together (e.g. TPMT with NUDT15)
have no recommendation unless both are given, so a gene that a named drug's key needs and the
question does not report is supplied with CPIC's no-variant value for that gene. A supplied
gene is listed in `supplied_genes`, since the patient was not tested for it.

build(question) returns the SI text and the {gene: {allele: count}} lookup for the CPIC API.
"""
import functools
import glob
import json
import os
import re

from .. import config

# Reported as 'xN' or 'x≥2': a duplication of unknown count, so CPIC's smallest one is used.
_UNSPECIFIED_COPIES = re.compile(r'^(\*\d+)x(?:N|n|≥2)$')
# An explicit absence of variants, which a question may write before or after the gene name
# ('no <gene> variant detected'), so it is read from the gene's wider context.
_NO_VARIANT = re.compile(r'\bno\s+(?:\w+\s+){0,2}variants?\b', re.I)
# '(*1/*2)xN': a duplication reported outside the diplotype, so which allele is duplicated is
# unknown. No CPIC genotype follows from it, and the phenotype the report states is used.
_COPIES_UNASSIGNED = re.compile(r'\([^()]*\)\s*x\s*(?:N|n|≥?\d+)')
# Zygosity words a report uses when it prints one allele; like the polarity words, a question
# may write them before the gene name ('heterozygous for <gene> <allele>'), so they are read
# from the gene's wider context.
_HOMOZYGOUS = re.compile(r'\bhomozygous\b', re.I)
_HETEROZYGOUS = re.compile(r'\bhetero(?:zygous|zygote)\b', re.I)
# A serotype is reported as found or not found; CPIC words it '<allele> positive'/'negative'.
_POLARITY = re.compile(r'\b(positive|negative)\b', re.I)
_WINDOW = 160                      # characters after a gene mention that may carry its genotype
# Words a report puts between the gene symbol and its genotype.
_CONNECTOR = re.compile(r'^[\s:=,()-]*(?:genotype|diplotype|allele|alleles|result|results|is|are'
                        r'|was|were|reports?|reported|shows?|reveals?|of)?[\s:=,()-]*$', re.I)


@functools.lru_cache(maxsize=1)
def table():
    return json.loads(config.CPIC_TABLE.read_text(encoding='utf-8'))


@functools.lru_cache(maxsize=1)
def _patterns():
    """The gene, per-gene allele, drug and lone-variant regexes, compiled once.

    Each is one alternation over the table's names, longest first, so a longer name wins where
    two overlap; they are built here rather than per question, which the table's size makes
    worth doing once.
    """
    t = table()
    genes = sorted(t['genes'], key=len, reverse=True)
    gene_re = re.compile(r'(?<![A-Za-z0-9])(' + '|'.join(re.escape(g) for g in genes) + r')(?![A-Za-z0-9])')
    allele_re = {}
    for g, info in t['genes'].items():
        keys = sorted(set(info['alleles']) | set(info['allele_prefixes']), key=len, reverse=True)
        if keys:
            allele_re[g] = re.compile(r'(?<![A-Za-z0-9])(' + '|'.join(re.escape(k) for k in keys)
                                      + r')(?![A-Za-z0-9])', re.I)
    drugs = sorted(t['drugs'], key=len, reverse=True)
    drug_re = re.compile(r'(?<![A-Za-z])(' + '|'.join(re.escape(d) for d in drugs) + r')(?![A-Za-z])', re.I)
    # Variant names CPIC gives to one gene only, so the name identifies the gene on its own.
    # A name of letters only is left out: it also reads as ordinary prose.
    lone = {al: g for al, g in t['alleles_unique_to_one_gene'].items() if not al.isalpha()}
    lone_re = re.compile(r'(?<![A-Za-z0-9])(' + '|'.join(
        re.escape(k) for k in sorted(lone, key=len, reverse=True)) + r')(?![A-Za-z0-9])', re.I)
    return gene_re, allele_re, drug_re, lone_re, lone


def _clean(question):
    """Drop markdown that wraps a gene symbol, so '*TPMT*' and '`RYR1`' read as the symbol.
    Star alleles are untouched: only emphasis around a gene name in the table is removed."""
    q = question.replace('`', ' ')
    return re.sub(r'\*([A-Za-z][A-Za-z0-9-]{1,})\*',
                  lambda m: m.group(1) if m.group(1).upper() in table()['genes'] else m.group(0), q)


@functools.lru_cache(maxsize=1)
def _corpus_index():
    """Gene -> the text of its guideline and supplement documents, for names the CPIC tables
    do not carry. Empty when the corpus is not installed."""
    index = {}
    if not config.CORPUS_DIR.is_dir():
        return index
    for path in sorted(glob.glob(str(config.CORPUS_DIR / '*.json'))):
        name = os.path.basename(path)
        if name.startswith(('doc_manifest', '_')):
            continue
        try:
            doc = json.loads(open(path, encoding='utf-8').read())
        except Exception:
            continue
        text = json.dumps(doc.get('prose_chunks', [])) + json.dumps(doc.get('blocks', []))
        for g in doc.get('gene', []):
            index.setdefault(g.upper(), []).append(text)
    return {g: '\n'.join(v).lower() for g, v in index.items()}


def gene_of_unlisted_name(name):
    """The gene of a variant name CPIC's tables do not carry: the one whose guidelines and
    supplements mention it, when exactly one does. None if the corpus is absent or undecided."""
    index = _corpus_index()
    if not index or len(name) < 4:
        return None
    hits = [g for g, text in index.items() if name.lower() in text]
    return hits[0] if len(hits) == 1 else None


def _canonical(gene, token):
    """One allele name as CPIC writes it, or None."""
    info = table()['genes'][gene]
    tl = token.strip().lower()
    name = info['alleles'].get(tl) or info['allele_prefixes'].get(tl)
    if name:
        return name
    m = _UNSPECIFIED_COPIES.match(token.strip())
    if m:
        copies = info['duplications'].get(m.group(1))
        if copies:
            return copies[0]
    return None


def _in_genotype_position(clause, m):
    """True when a short, prose-like allele name ('A', 'B') stands where a genotype is written:
    joined by '/' to another allele, or directly after the gene symbol."""
    before, after = clause[:m.start()], clause[m.end():]
    if before.rstrip().endswith('/') or after.lstrip().startswith('/'):
        return True
    return bool(_CONNECTOR.match(before))


def _alleles_in(gene, clause):
    """-> (list of canonical allele names in reading order, end offset of the last one).

    A question may name one variant twice, in CPIC's star form and in cDNA, protein or dbSNP
    form. Those are one allele, so a canonical name repeats here only where the question joins
    the two with '/', which is how a diplotype of two identical alleles is written."""
    rx = _patterns()[1].get(gene)
    if not rx:
        return [], 0
    position_only = set(table()['genes'][gene]['position_only_aliases'])
    found, end = [], 0
    for m in rx.finditer(clause):
        if m.group(1).lower() in position_only and not _in_genotype_position(clause, m):
            continue
        name = _canonical(gene, m.group(1))
        if name:
            found.append((m.start(), m.end(), name))
            end = m.end()
    found.sort()
    out = []
    for start, stop, name in found:
        if out and out[-1][1] == name:
            between = clause[out[-1][0]:start]
            if '/' not in between:          # the same allele written a second way, not a pair
                continue
        out.append((stop, name))
    return [n for _, n in out], end


def _polarity(clause, at):
    """Whether a serotype was found: the nearest 'positive'/'negative' in the gene's own
    clause, measured from the allele. A serotype named with neither word was found."""
    best = (None, None)
    for m in _POLARITY.finditer(clause):
        d = 0 if m.start() <= at <= m.end() else min(abs(m.start() - at), abs(m.end() - at))
        if best[0] is None or d < best[0]:
            best = (d, m.group(1).lower())
    return best[1] or 'positive'


def _genotype(gene, clause, allele_at=0):
    """The genotype a question states for one gene: {allele: count}, or None.

    clause is the gene's own span of the question; allele_at is where that span starts inside
    the wider clause used to read 'positive'/'negative'."""
    info = table()['genes'][gene]
    ref = info['reference_allele']
    serotype = info['reported_as'] == 'serotype'

    alleles, end = _alleles_in(gene, clause)
    if _COPIES_UNASSIGNED.search(clause):
        alleles = []
    if alleles:
        if serotype:
            name = alleles[0]
            if not info['polarity_suffix'] or name.lower().endswith(('positive', 'negative')):
                return {name: 1}
            at = allele_at + clause.lower().find(alleles[0].lower())
            return {f'{name} {_polarity(_genotype.context, at)}': 1}
        if len(alleles) >= 2:                         # both copies are written out
            dip = {}
            for a in alleles[:2]:
                dip[a] = dip.get(a, 0) + 1
            return dip
        # Only one allele is written. A stated zygosity then says what the other copy is.
        zygosity = _genotype.context or clause
        if _HETEROZYGOUS.search(zygosity) and ref and alleles[0] != ref:
            return {alleles[0]: 1, ref: 1}
        if _HOMOZYGOUS.search(zygosity):              # one allele printed, both copies carry it
            return {alleles[0]: 2}
        if info.get('chromosome') == 'chrX':          # one call stands for a hemizygous gene
            return {alleles[0]: 1}
        if ref:                       # one allele reported: the other chromosome is reference
            return {alleles[0]: 1, ref: 1} if alleles[0] != ref else {ref: 2}
        return {alleles[0]: 1}

    for phrase, result in sorted(info['phenotypes'].items(), key=lambda kv: -len(kv[0])):
        if re.search(r'(?<![a-z])' + re.escape(phrase) + r'(?![a-z])', clause, re.I):
            return {'__phenotype__': result}

    if ref and _NO_VARIANT.search(_genotype.context or clause):
        return {ref: 2}
    return None


_genotype.context = ''                 # the clause 'positive'/'negative' is read from


def _resolve_phenotype(gene, result, client):
    """A report that gives a phenotype and no usable genotype (e.g. a duplication of unknown
    count): take the alleles of a CPIC diplotype that yields that phenotype."""
    return client.alleles_for_phenotype(gene, result) or None


def match(question, client=None):
    """-> (genes, genotypes, drugs, phenotypes, supplied). genotypes maps a gene to
    {allele: count}; phenotypes holds the genes the question states only as a phenotype;
    supplied names the genes a named drug's CPIC key needs that the question does not report."""
    t = table()
    q = _clean(question)
    gene_re, _, drug_re, lone_re, lone = _patterns()

    hits = [(m.group(1), m.start(), m.end()) for m in gene_re.finditer(q)]
    # A variant CPIC assigns to one gene only names that gene, so a question that prints the
    # variant without the symbol is still resolved.
    for m in lone_re.finditer(q):
        gene = lone[m.group(1)] if m.group(1) in lone else lone.get(m.group(1).lower())
        if gene is None:
            gene = next((g for al, g in lone.items() if al.lower() == m.group(1).lower()), None)
        if gene and not any(h[0] == gene for h in hits):
            hits.append((gene, m.start(), m.start()))
    hits.sort(key=lambda h: h[1])
    genes, genotypes, phenotypes = [], {}, {}
    for i, (gene, start, pos) in enumerate(hits):
        if gene not in genes:
            genes.append(gene)
        if gene in genotypes or gene in phenotypes:
            continue
        # The gene's own clause ends where the next gene is named, so a neighbouring gene's
        # alleles cannot be read as this one's.
        nxt = hits[i + 1][1] if i + 1 < len(hits) else len(q)
        stop = min(pos + _WINDOW, nxt)
        prev = hits[i - 1][2] if i else 0
        _genotype.context = q[prev:stop]
        g = _genotype(gene, q[pos:stop], allele_at=pos - prev)
        if not g:
            continue
        if '__phenotype__' in g:
            phenotypes[gene] = g['__phenotype__']
            resolved = _resolve_phenotype(gene, g['__phenotype__'], client) if client else None
            if resolved:
                genotypes[gene] = resolved
        else:
            genotypes[gene] = g

    drugs, seen = [], set()
    for m in drug_re.finditer(q):
        name = t['drugs'][m.group(1).lower()]
        if name.lower() not in seen:
            seen.add(name.lower())
            drugs.append(name)

    # CPIC keys some recommendations on two genes together and returns nothing unless both are
    # given. Only a drug the question names decides that: supplying a gene a named drug does not
    # need would answer from a guideline the question never asked about, and would assert a test
    # result the patient does not have. A supplied gene carries the value CPIC writes for a gene
    # with no variant, goes into the lookup only, and is reported in `supplied` so the answer can
    # say it was not tested.
    supplied = set()
    for drug in drugs:
        for gene in t['drug_lookup_genes'].get(drug, []):
            if gene in genotypes or gene in phenotypes or gene in supplied:
                continue
            if not any(g in genotypes or g in phenotypes
                       for g in t['drug_lookup_genes'][drug] if g != gene):
                continue            # the question reports none of this drug's genes
            info = t['genes'][gene]
            if info['reported_as'] == 'serotype':
                absent = t['serotype_absent_value'].get(gene)
                if absent:
                    genotypes[gene] = {absent: 1}
                    supplied.add(gene)
            elif info['reference_allele']:
                genotypes[gene] = {info['reference_allele']: 2}
                supplied.add(gene)
    return genes, genotypes, drugs, phenotypes, supplied


def _star_number(allele):
    m = re.match(r'^\*(\d+)', allele)
    return int(m.group(1)) if m else None


def _render(gene, dip):
    """'TPMT*6/*8' / 'HLA-B*15:02 negative' / 'CACNA1S c.3257G>A/Reference'. Star alleles are
    ordered by number and a reference allele last, as CPIC writes its diplotypes."""
    parts = []
    for allele, count in dip.items():
        parts += [allele] * count
    ref = table()['genes'][gene]['reference_allele']
    if all(_star_number(p) is not None for p in parts):
        parts.sort(key=_star_number)
    else:
        parts.sort(key=lambda p: (p == ref, p))
    joined = '/'.join(parts[:2]) if len(parts) > 1 else parts[0]
    return gene + ('' if joined.startswith('*') else ' ') + joined


def build(question, client=None):
    """-> {'si_text', 'genes', 'variants', 'drugs', 'lookup'}.

    'lookup' is the {gene: {allele: count}} argument of the CPIC recommendation lookup;
    'si_text' is the SI block the pipelines use as the retrieval query."""
    genes, genotypes, drugs, phenotypes, supplied = match(question, client)
    variants = []
    for g in genes:                                   # named genes only, in reading order
        if g in phenotypes:                           # the question stated a phenotype
            variants.append(f'{g} {phenotypes[g]}')
        elif g in genotypes and g not in supplied:
            variants.append(_render(g, genotypes[g]))
    si = (f"## Question\n{question.strip()}\n\n## PGx Analysis\n"
          f"- Key Elements: genes={genes!r}, variants={variants!r}, drugs={drugs!r}\n")
    return {'si_text': si, 'genes': genes, 'variants': variants, 'drugs': drugs,
            'lookup': genotypes, 'phenotypes': phenotypes, 'supplied_genes': sorted(supplied)}
