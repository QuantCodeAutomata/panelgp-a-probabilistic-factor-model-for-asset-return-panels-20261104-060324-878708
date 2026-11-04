"""Shared pytest fixtures: market-data marker registration and a reduced
real-data fixture cached across test runs."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: requires DATALOADER_API_KEY/MASSIVE_API_KEY and a live Dataloader call"
    )


@pytest.fixture(scope="session")
def small_ohlcv(request):
    """A few tickers, two years of daily bars, loaded through
    data/data_loader.py and cached between test runs via pytest's cache."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set; skipping market-data test")
    from data.data_loader import load_ohlcv

    cache_dir = request.config.cache.mkdir("panelgp_small_ohlcv")
    cache_file = cache_dir / "bars.csv"
    if cache_file.exists():
        import pandas as pd
        bars = pd.read_csv(cache_file, parse_dates=["date"])
        return bars
    bars, _meta = load_ohlcv(["AAPL", "MSFT", "XOM"], "2015-01-01", "2016-12-31", "1d")
    bars.to_csv(cache_file, index=False)
    return bars
