"""Question and ground-truth generation for the PGx question bank.

The CPIC section of a question is built with PGxRoute and GT claims are decomposed with the
Verifier, so both sibling folders are put on the import path here.
"""
import sys

from .config import REPO_ROOT

for _p in (REPO_ROOT / "PGxRoute", REPO_ROOT / "Verifier"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))
