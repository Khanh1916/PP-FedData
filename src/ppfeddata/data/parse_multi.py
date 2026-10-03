"""Parsing of multi-value cells (e.g. "Publish Message,Publish Message", "40,41")."""
from __future__ import annotations

import pandas as pd

_EMPTY = {"", "nan", "none", "null"}


def parse_multi(cell: object) -> list[str]:
    """Split a cell into its stripped, non-empty elements. Empty / NaN / whitespace-only cells give [].

    >>> parse_multi("40,41")
    ['40', '41']
    >>> parse_multi(float("nan"))
    []
    """
    if cell is None:
        return []
    try:
        if pd.isna(cell):
            return []
    except (TypeError, ValueError):
        pass
    parts = [p.strip() for p in str(cell).split(",")]
    return [p for p in parts if p.lower() not in _EMPTY]


def first_values(series: pd.Series) -> pd.Series:
    """First element of every cell (NaN where the cell has none).

    G1 decision (multi_policy = first_only): the Normal export kept only the first occurrence of each field, so the
    first element is the only value comparable between Normal and Attack rows.
    """
    return series.map(lambda c: (parse_multi(c) or [None])[0])


def count_values(series: pd.Series) -> pd.Series:
    """Number of elements per cell (not used as a feature after G1; kept for audits and tests)."""
    return series.map(lambda c: len(parse_multi(c)))
