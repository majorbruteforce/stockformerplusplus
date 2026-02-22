"""
Data fetching module using yfinance.
Fetches daily adjusted stock data.
"""

import os
import pandas as pd
import yfinance as yf
from typing import Optional
from config import DATA_CONFIG, DATA_DIR


def fetch_daily_adjusted(
    symbol: str, force_refresh: bool = False, years: int = 10
) -> pd.DataFrame:
    """
    Fetch daily adjusted time series using yfinance.

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'SPY')
        force_refresh: Force re-download even if cached data exists
        years: Number of years of historical data to fetch

    Returns:
        DataFrame with columns: open, high, low, close, adjusted_close, volume
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file = os.path.join(DATA_DIR, f"{symbol}_daily_adjusted.csv")

    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached data for {symbol} from {cache_file}")
        df = pd.read_csv(cache_file)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.set_index("Date")
            df.index = df.index.tz_localize(None)
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
            df.index = df.index.tz_localize(None)
        return df

    print(f"Fetching {symbol} data from yfinance...")

    ticker = yf.Ticker(symbol)

    df = ticker.history(period=f"{years}y", auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted_close",
            "Volume": "volume",
        }
    )

    df = df[["open", "high", "low", "close", "adjusted_close", "volume"]]

    df = df.dropna()

    df.to_csv(cache_file)
    print(f"Fetched {len(df)} days of data for {symbol}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def load_or_fetch_data(
    symbol: str = DATA_CONFIG["symbol"], force_refresh: bool = False, years: int = 10
) -> pd.DataFrame:
    """
    Load cached data or fetch fresh data.
    """
    return fetch_daily_adjusted(symbol, force_refresh, years)


if __name__ == "__main__":
    df = load_or_fetch_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
