"""Settings. Model IDs, paths and limits can be overridden from the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # Question-generation/
REPO_ROOT = ROOT.parent
PROMPT_DIR = ROOT / "prompts"
WORK_DIR = Path(os.environ.get("QGEN_WORK_DIR", str(ROOT / "work")))

# The two LLMs. Every selection and review step runs them as independent readers.
OPUS = os.environ.get("QGEN_OPUS_MODEL", "claude-opus-4-8")
FABLE = os.environ.get("QGEN_FABLE_MODEL", "claude-fable-5-1")
READERS = (OPUS, FABLE)          # blind passage selectors, GT reviewers, claim-coverage reviewers

QUESTION_WRITER = os.environ.get("QGEN_QUESTION_MODEL", OPUS)
GT_WRITER = FABLE                # hard GT, intermediate GT curation, revisions
CONVERGER = FABLE                # merges the two blind passage selections
SETTLER = OPUS                   # settles the two readers' findings
CATEGORIZER = OPUS               # sorts CPIC information into categories
CURATOR = FABLE                  # curates decomposed GT claims
COMMON_DRAFTER, COMMON_CRITIC = OPUS, FABLE

EFFORT = os.environ.get("QGEN_EFFORT", "high")
MAX_TOKENS = int(os.environ.get("QGEN_MAX_TOKENS", 64000))
# Claude Fable requests opt into the server-side refusal fallback; the model that actually
# served each call is recorded next to its output. QGEN_FALLBACKS=0 turns it off.
FALLBACKS = os.environ.get("QGEN_FALLBACKS", "1") != "0"
WORKERS = int(os.environ.get("QGEN_WORKERS", 4))

CURATION_ROUNDS = int(os.environ.get("QGEN_CURATION_ROUNDS", 6))     # claim curation, at most
COMMON_GT_ROUNDS = int(os.environ.get("QGEN_COMMON_GT_ROUNDS", 4))   # draft-critique exchanges
REVIEW_ROUNDS = int(os.environ.get("QGEN_REVIEW_ROUNDS", 2))         # GT review -> revise
SEED = int(os.environ.get("QGEN_SEED", 17))

# CPIC API tables the generator reads (fetched once into WORK_DIR/cpic_snapshot).
CPIC_BASE = os.environ.get("CPIC_BASE", "https://api.cpicpgx.org/v1")
CPIC_TABLES = {
    "recommendation": "*",
    "drug": "*",
    "guideline": "*",
    "pair": "*",
    "diplotype": "genesymbol,diplotype,generesult,lookupkey",
}
MAX_EXAMPLE_DIPLOTYPES = 8
