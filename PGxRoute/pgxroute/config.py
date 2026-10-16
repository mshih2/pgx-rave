"""Settings for the three answer pipelines.

Every value can be overridden with the environment variable of the same name.
"""
import os
from pathlib import Path

# Repository root (the folder holding prompts/ and data/), one level above this package.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name, default, cast=str):
    v = os.getenv(name)
    return default if v is None or v == "" else cast(v)


def _flag(name, default):
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- paths
PROMPT_DIR = Path(_env("PROMPT_DIR", str(REPO_ROOT / "prompts")))
# CPIC lookup table: every gene, allele, drug and phenotype name of the CPIC API.
CPIC_TABLE = Path(_env("CPIC_TABLE", str(REPO_ROOT / "data" / "cpic_lookup.json")))
# Frozen retrieval corpus: one JSON per guideline/supplement document, plus
# doc_manifest.json and _provenance.json.
CORPUS_DIR = Path(_env("CORPUS_DIR", str(REPO_ROOT / "data" / "corpus")))
EMB_CACHE_DIR = Path(_env("EMB_CACHE_DIR", str(REPO_ROOT / "data" / "emb_cache")))

# ---------------------------------------------------------------- CPIC API
CPIC_BASE = _env("CPIC_BASE", "https://api.cpicpgx.org/v1")
CPIC_TIMEOUT = _env("CPIC_TIMEOUT", 20, int)
CPIC_MAX_CHARS = 8000                      # CPIC output passed to the reasoner and critic

# ---------------------------------------------------------------- reasoner (all pipelines)
REASONER_BASE_URL = _env("REASONER_BASE_URL", "http://127.0.0.1:8019/v1").rstrip("/")
REASONER_MODEL = _env("REASONER_MODEL", "qwen3.8-27b")       # vLLM served name of Qwen3.8-27B
REASONER_TEMPERATURE = _env("REASONER_TEMPERATURE", 0.05, float)
REASONER_TOP_P = _env("REASONER_TOP_P", 0.95, float)
REASONER_MAX_TOKENS = _env("REASONER_MAX_TOKENS", 24000, int)
REASONER_THINKING = _flag("REASONER_THINKING", True)
REASONER_EFFORT = _env("REASONER_EFFORT", "medium")          # Qwen3.8 template default is xhigh
REASONER_SEED = _env("REASONER_SEED", None, int)              # None: no seed is sent
REASONER_TIMEOUT = _env("REASONER_TIMEOUT", 1800, int)
REASONER_EMPTY_RETRIES = _env("REASONER_EMPTY_RETRIES", 3, int)
CONTEXT_MAX_CHARS = 32000                  # retrieved units passed to the reasoner and critic

# ---------------------------------------------------------------- retrieval (Full pipeline)
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = _env("EMBED_DEVICE", "cuda:0")
# Optional: run BGE-M3 scoring under another interpreter (e.g. a venv with FlagEmbedding and
# a transformers version the main environment cannot use). Empty = score in this process.
EMBED_PYTHON = _env("EMBED_PYTHON", "")
EMBED_TIMEOUT = _env("EMBED_TIMEOUT", 1800, int)
SPARSE_WEIGHT = 0.3                        # hybrid score = dense + 0.3 x sparse
POOL_SIZE = 60
TOP_N = 8
MIN_DRUG_SCOPE_UNITS = 8                   # drug-narrowed scope used only if it keeps >= 8 units

RERANKER_URL = _env("RERANKER_URL", REASONER_BASE_URL + "/chat/completions")
RERANKER_MODEL = _env("RERANKER_MODEL", REASONER_MODEL)
RERANKER_TOKENIZER = _env("RERANKER_TOKENIZER", "Qwen/Qwen3.8-27B")
RERANKER_THINKING = _flag("RERANKER_THINKING", False)
RERANKER_CONTEXT_WINDOW = 40960
RERANKER_MAX_TOKENS = (4000, 8000, 16000)  # raised only when the ranking is truncated
RERANKER_MARGIN = 64

# Stamp of the frozen corpus (sha256 over its file hashes). The corpus is not redistributed;
# set VERIFY_CORPUS=0 to run on a corpus you built yourself.
VERIFY_CORPUS = _flag("VERIFY_CORPUS", True)
EXPECTED_CORPUS_VERSION = _env(
    "EXPECTED_CORPUS_VERSION",
    "27b7de3a20fbea2fb36e381348576fc7e531a4dd497745b9ae26e7b84b045605")

# ---------------------------------------------------------------- critique (Full pipeline)
CRITIC_URL = _env("CRITIC_URL", "http://127.0.0.1:8006/v1/chat/completions")
CRITIC_MODEL = _env("CRITIC_MODEL", "gemma-4-31b-it")
CRITIC_TEMPERATURE = _env("CRITIC_TEMPERATURE", 0.05, float)
CRITIC_MAX_TOKENS = 2048
CRITIC_TIMEOUT = _env("CRITIC_TIMEOUT", 400, int)
CRITIC_ATTEMPTS = 3

# ---------------------------------------------------------------- routers
# LLM routers (routers/rl): Qwen3-8B, untrained or GRPO-trained, behind an OpenAI-compatible
# server.
ROUTER_BASE_URL = _env("ROUTER_BASE_URL", "http://127.0.0.1:8020/v1").rstrip("/")
ROUTER_MODEL = _env("ROUTER_MODEL", "Qwen/Qwen3-8B")      # or the served name of a trained checkpoint
ROUTER_TEMPERATURE = _env("ROUTER_TEMPERATURE", 0.6, float)
ROUTER_TOP_P = _env("ROUTER_TOP_P", 0.95, float)
ROUTER_MAX_TOKENS = _env("ROUTER_MAX_TOKENS", 2048, int)
ROUTER_RESAMPLES = 2                       # an unparsable reply is resampled up to twice
ROUTER_TIMEOUT = _env("ROUTER_TIMEOUT", 600, int)
# Text classifiers (routers/ml): a model saved by `python -m pgxroute.routers.ml.text_classifier`.
CLASSIFIER_DIR = Path(_env("CLASSIFIER_DIR", str(REPO_ROOT / "data" / "routers")))
