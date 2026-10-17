"""Settings of the atomic-claim verifier.

Every value read with _env can be overridden with the environment variable of the same name.
"""
import os
from pathlib import Path

# Repository root (the folder holding prompts/), one level above this package.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name, default, cast=str):
    v = os.getenv(name)
    return default if v is None or v == "" else cast(v)


PROMPT_DIR = Path(_env("PROMPT_DIR", str(REPO_ROOT / "prompts")))

# ---------------------------------------------------------------- decomposition (Gemma-4-31B-it)
# One OpenAI-compatible vLLM server runs both passes and the deduplication.
DECOMPOSER_URL = _env("DECOMPOSER_URL", "http://127.0.0.1:8006/v1").rstrip("/")
DECOMPOSER_MODEL = _env("DECOMPOSER_MODEL", "gemma-4-31b-it")
DECOMPOSER_TEMPERATURE = 0.0
DECOMPOSE_MAX_TOKENS = 4096                # per pass; raised once to the context limit if cut
DEDUP_MAX_TOKENS = 12288
DECOMPOSER_TIMEOUT = _env("DECOMPOSER_TIMEOUT", 600, int)

# ---------------------------------------------------------------- labelling judge (Grok 4.5)
XAI_BASE_URL = _env("XAI_BASE_URL", "https://api.x.ai/v1")
LABEL_MODEL = _env("LABEL_MODEL", "grok-4.5")
LABEL_TEMPERATURE = 0.0
LABEL_TIMEOUT = _env("LABEL_TIMEOUT", 600, int)
LABEL_RETRIES = 6                          # transient errors, empty replies, malformed JSON
LABEL_BACKOFF = 4.0                        # seconds, doubled per retry, capped at 30
LABEL_SAFETY_SLEEP = 2.0                   # before retrying a content-safety 403 or bad JSON
LABEL_CALL_SLEEP = 0.25                    # per worker, before every call
# The API key is read from XAI_API_KEY when the first labelling call is made.

# ---------------------------------------------------------------- grading judge (Mistral-Small-24B)
GRADER_URL = _env("GRADER_URL", "http://127.0.0.1:8012/v1").rstrip("/")
GRADER_MODEL = _env("GRADER_MODEL", "mistral-small-24b")
GRADER_TIMEOUT = _env("GRADER_TIMEOUT", 180, int)
GRADER_MAX_TOKENS = 80
GRADER_TOP_LOGPROBS = 15

# ---------------------------------------------------------------- score
EXTRA_WEIGHT = 0.1                         # an extra claim counts 0.1 of a false positive
F_BETA = 1.5

WORKERS = _env("WORKERS", 12, int)         # questions scored in parallel; grading calls per question
