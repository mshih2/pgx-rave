"""Rule baseline: three rules over the key elements of the structured input (SI).

  no gene and no variant                                            -> reasoning_only
  exactly one gene, at least one variant, exactly one drug, no condition -> api_only
  otherwise                                                         -> full

Unlike the other routers it reads the SI, not only the question text, so every question (common
ones included) goes through the SI step first.

si_builder builds the SI by lookup against the CPIC table, which has no condition vocabulary,
so `route` passes no conditions; pass them to `pick` if you have them.
"""
from ..steps import cpic_api, si_builder


def pick(genes, variants, drugs, conditions=()):
    """The pipeline the three rules choose for these key elements."""
    if not genes and not variants:
        return "reasoning_only"
    if len(genes) == 1 and variants and len(drugs) == 1 and not conditions:
        return "api_only"
    return "full"


class RuleRouter:
    name = "rule"

    def route(self, question):
        si = si_builder.build(question, cpic_api.client)
        return pick(si["genes"], si["variants"], si["drugs"])
