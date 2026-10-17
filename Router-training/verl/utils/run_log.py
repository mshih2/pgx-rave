# SPDX-License-Identifier: Apache-2.0
"""Tee stdout/stderr to a log file (used by main_grpo_pgx, verl FSDP workers).

Expected on ``sys.path`` with ``TRAINING_DIR`` as cwd (see ``main_grpo_pgx``).
"""
from __future__ import annotations

import os
import sys
from typing import List, TextIO


class _Tee(TextIO):
    __slots__ = ("_streams",)

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
            try:
                s.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return getattr(self._streams[0], "isatty", lambda: False)()

    def fileno(self) -> int:
        # Ray calls ``faulthandler.enable()`` on ``sys.stderr`` and needs a real OS fd.
        # First stream is always ``sys.__stdout__`` / ``sys.__stderr__``.
        return self._streams[0].fileno()


_log_handles: List[TextIO] = []


def setup_run_log(log_path: str, mode: str = "w") -> None:
    """Mirror stdout/stderr to ``log_path`` while keeping the console.

    Safe to call more than once (e.g. driver + Ray TaskRunner): each call adds
    another log file to the tee chain.
    """
    path = os.path.abspath(log_path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    h = open(path, mode, encoding="utf-8", buffering=1)
    _log_handles.append(h)
    sys.stdout = _Tee(sys.__stdout__, *_log_handles)
    sys.stderr = _Tee(sys.__stderr__, *_log_handles)
