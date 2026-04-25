from __future__ import annotations

from typing import Any

import joblib


def ensure_sklearn_pickle_compat() -> None:
    """
    Add compatibility shims for loading pipelines serialized across sklearn versions.

    Some pickles reference private sklearn internals such as
    sklearn.compose._column_transformer._RemainderColsList. In certain runtime
    versions this symbol is missing, which breaks joblib.load().
    """
    try:
        from sklearn.compose import _column_transformer as column_transformer
    except Exception:
        # If sklearn itself cannot be imported we let joblib.load raise the
        # original error; this helper only patches known compatibility gaps.
        return

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compatibility shim for sklearn private type used in older pickles."""

        column_transformer._RemainderColsList = _RemainderColsList


def load_joblib_model(path: str) -> Any:
    ensure_sklearn_pickle_compat()
    return joblib.load(path)
