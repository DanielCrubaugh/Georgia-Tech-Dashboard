"""Local ML4T-style market-data loader used by this dashboard.

It keeps the course CSV convention: one file per ticker, a ``Date`` column,
and an ``Adj Close`` column.  ``base_dir`` is explicit so the dashboard does
not depend on a terminal environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd


def symbol_to_path(symbol: str, base_dir: str | None = None) -> str:
    """Return the CSV path for a ticker symbol."""
    directory = base_dir or os.environ.get("MARKET_DATA_DIR", "../data/")
    return os.path.join(directory, f"{symbol}.csv")


def get_data(
    symbols: Iterable[str],
    dates: pd.DatetimeIndex,
    addSPY: bool = True,
    colname: str = "Adj Close",
    base_dir: str | None = None,
) -> pd.DataFrame:
    """Load selected price columns from ML4T per-symbol CSV files.

    SPY is added by default and non-trading SPY dates are removed, matching
    the useful behavior of the supplied course utility.
    """
    tickers = [str(symbol).strip().upper() for symbol in symbols]
    if addSPY and "SPY" not in tickers:
        tickers.insert(0, "SPY")
    result = pd.DataFrame(index=dates)
    for ticker in tickers:
        csv_path = symbol_to_path(ticker, base_dir)
        prices = pd.read_csv(
            csv_path,
            index_col="Date",
            parse_dates=True,
            usecols=["Date", colname],
            na_values=["nan"],
        ).rename(columns={colname: ticker})
        result = result.join(prices, how="left")
        if ticker == "SPY":
            result = result.dropna(subset=["SPY"])
    return result
