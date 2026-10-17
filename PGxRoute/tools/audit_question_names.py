"""Check a question bank against the CPIC lookup table, before the questions are used.

For every question it reports the genes, genotypes and drugs the table resolves, and the
gene-, variant- or drug-like names it cannot resolve. An unresolved name is looked for in the
guideline and supplement corpus; if it is not there either, the name in the question is taken
to be wrong, and the names the question's ground truth uses are printed as the likely intent.

This is a bank-preparation tool. It reads ground truth, so it must never run inside the answer
pipeline: the pipeline resolves a question from the table and the corpus alone.

  python tools/audit_question_names.py bank.jsonl

bank.jsonl: one {"question_id", "question", "ground_truth"} per line ("ground_truth" optional).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgxroute.steps import si_builder  # noqa: E402

# Tokens written like a gene symbol, an allele or a variant. A gene symbol must carry a digit,
# which keeps ordinary clinical acronyms (CPIC, LDL, SJS) out of the report.
NAME_LIKE = re.compile(r'(?<![A-Za-z0-9])('
                       r'[A-Z][A-Z0-9]*\d[A-Z0-9]*(?:-[A-Z0-9]+)?'   # gene-like symbol
                       r'|\*\d+[A-Za-z0-9+x≥:]*'                      # star allele
                       r'|[cmgp]\.[\w>+_.-]+'                         # HGVS
                       r'|rs\d+'                                      # dbSNP
                       r')')


def resolved_names(question):
    """Every table name the matcher reads out of the question, lower-cased."""
    si = si_builder.build(question)
    out = {g.lower() for g in si['genes']} | {d.lower() for d in si['drugs']}
    for v in si['variants']:
        for gene in si['genes']:
            if v.startswith(gene):
                v = v[len(gene):]
                break
        out |= {p.strip().lower() for p in re.split(r'[/\s]', v) if p.strip()}
    return si, out


def audit(question, ground_truth=None):
    si, seen = resolved_names(question)
    table = si_builder.table()
    unresolved = []
    for m in NAME_LIKE.finditer(si_builder._clean(question)):
        tok = m.group(1)
        if tok.lower() in seen or tok.upper() in table['genes'] or tok.lower() in table['drugs']:
            continue
        if any(tok.lower() in s for s in seen):          # part of a name already resolved
            continue
        # A name the table resolves for one of the question's genes is not unresolved, even
        # when the matcher wrote the diplotype in CPIC's own wording.
        if any(si_builder._canonical(g, tok) for g in si['genes'] if g in table['genes']):
            continue
        item = {'name': tok}
        gene = si_builder.gene_of_unlisted_name(tok)
        if gene:
            item['found_in_corpus_of_gene'] = gene
        elif ground_truth:
            gt_si, gt_names = resolved_names(ground_truth)
            item['not_in_cpic_or_corpus'] = True
            item['ground_truth_names'] = {'genes': gt_si['genes'],
                                          'variants': gt_si['variants'],
                                          'drugs': gt_si['drugs']}
        else:
            item['not_in_cpic_or_corpus'] = True
        unresolved.append(item)
    return {'genes': si['genes'], 'variants': si['variants'], 'drugs': si['drugs'],
            'unresolved': unresolved}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    flagged = 0
    for line in open(sys.argv[1], encoding='utf-8'):
        if not line.strip():
            continue
        item = json.loads(line)
        rep = audit(item['question'], item.get('ground_truth'))
        rep['question_id'] = item.get('question_id')
        if rep['unresolved']:
            flagged += 1
        print(json.dumps(rep, ensure_ascii=False))
    print(f'# questions with unresolved names: {flagged}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
