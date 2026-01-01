#!/usr/bin/env python3
"""
All Weather portfolio backtest (risk parity + vol targeting).

Data sources:
- Stooq (default, no API key)
- Alpaca (requires APCA_API_KEY_ID / APCA_API_SECRET_KEY)

Notes:
- This is an ETF proxy of the ideas in the Bridgewater document.
- Trading and financing costs can be included; taxes and slippage are not modeled.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

DEFAULT_TICKERS = ["SPY", "TLT", "IEF", "GLD", "DBC"]
DEFAULT_FIXED_WEIGHTS = {
    "SPY": 0.30,
    "TLT": 0.40,
    "IEF": 0.15,
    "GLD": 0.075,
    "DBC": 0.075,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="All Weather backtest with risk parity + volatility targeting",
    )
    parser.add_argument(
        "--data-source",
        choices=["stooq", "alpaca"],
        default="stooq",
        help="Price data source",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for cached price data (set empty to disable)",
    )
    parser.add_argument(
        "--data-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Cache format for price data",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh cached price data",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ETF tickers (e.g. SPY,TLT,IEF,GLD,DBC)",
    )
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker (set to 'none' to disable)",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--portfolio",
        choices=["risk-parity", "inverse-vol", "fixed"],
        default="risk-parity",
        help="Weighting scheme",
    )
    parser.add_argument(
        "--fixed-weights",
        default=None,
        help='Fixed weights like "SPY=0.3,TLT=0.4,IEF=0.15,GLD=0.075,DBC=0.075"',
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback window for risk estimates (trading days)",
    )
    parser.add_argument(
        "--rebalance",
        default="M",
        help="Rebalance frequency (pandas offset alias, e.g. M, W-FRI)",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=0.10,
        help="Target annualized volatility (e.g. 0.10 for 10%%)",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=2.0,
        help="Max gross leverage (e.g. 2.0 means 2x)",
    )
    parser.add_argument(
        "--ann-factor",
        type=int,
        default=252,
        help="Annualization factor for daily returns",
    )
    parser.add_argument(
        "--alpaca-feed",
        default=None,
        help="Alpaca feed (iex or sip). Optional.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for CSVs (optional)",
    )
    parser.add_argument(
        "--trade-cost",
        type=float,
        default=0.0,
        help="Trading cost per $ turnover (e.g. 0.0005 = 5 bps)",
    )
    parser.add_argument(
        "--borrow-rate",
        type=float,
        default=0.0,
        help="Annualized borrow rate on leverage (e.g. 0.03 for 3%%)",
    )
    parser.add_argument(
        "--cash-rate",
        type=float,
        default=0.0,
        help="Annualized interest on idle cash (e.g. 0.02 for 2%%)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save equity + drawdown plot (uses --out or cwd)",
    )
    return parser.parse_args()


def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _parse_tickers(value: str) -> List[str]:
    tickers = [t.strip().upper() for t in value.split(",") if t.strip()]
    if not tickers:
        raise ValueError("No tickers provided.")
    return tickers


def _parse_benchmark(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    symbol = value.strip().upper()
    if symbol in {"", "NONE", "OFF", "NO"}:
        return None
    return symbol


def _parse_fixed_weights(value: str) -> Dict[str, float]:
    weights = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid fixed weight entry: {item}")
        key, val = item.split("=", 1)
        key = key.strip().upper()
        weights[key] = float(val)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Fixed weights must sum to a positive value.")
    return {k: v / total for k, v in weights.items()}


def _ensure_parquet_support() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support requires pyarrow or fastparquet. "
                "Install with `pip install pyarrow` or `uv sync --extra data`."
            ) from exc


def _cache_path(
    data_dir: Optional[Path],
    source: str,
    ticker: str,
    data_format: str,
) -> Optional[Path]:
    if data_dir is None:
        return None
    ext = "csv" if data_format == "csv" else "parquet"
    safe_ticker = ticker.replace("/", "_")
    return data_dir / f"{source}_{safe_ticker}.{ext}"


def _cache_covers_range(
    frame: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> bool:
    if frame.empty:
        return False
    idx = frame.index
    if start is not None and idx.min() > start:
        return False
    if end is not None and idx.max() < end:
        return False
    return True


def _load_cached_frame(path: Path, ticker: str, data_format: str) -> pd.DataFrame:
    if data_format == "csv":
        df = pd.read_csv(path, index_col=0, parse_dates=[0])
    else:
        _ensure_parquet_support()
        df = pd.read_parquet(path)
    if df.empty:
        return df
    if ticker not in df.columns and df.shape[1] == 1:
        df.columns = [ticker]
    if ticker not in df.columns:
        raise ValueError(f"Cache file {path} missing column {ticker}.")
    df = df[[ticker]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index.name = "Date"
    return df


def _save_cached_frame(path: Path, frame: pd.DataFrame, data_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    if data_format == "csv":
        frame.to_csv(path)
    else:
        _ensure_parquet_support()
        frame.to_parquet(path)


def _merge_cached_frame(
    cached: Optional[pd.DataFrame],
    fresh: pd.DataFrame,
) -> pd.DataFrame:
    if cached is None or cached.empty:
        combined = fresh.copy()
    else:
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _read_stooq_csv(url: str, max_retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            if not text.strip():
                raise ValueError("Empty response from Stooq.")
            return pd.read_csv(io.StringIO(text))
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            pd.errors.EmptyDataError,
            ValueError,
        ) as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            else:
                break

    try:
        return pd.read_csv(url)
    except Exception as exc:
        if last_err is not None:
            raise RuntimeError(
                f"Failed to fetch Stooq data after {max_retries} attempts: {last_err}"
            ) from exc
        raise


def fetch_stooq_prices(
    tickers: Iterable[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    data_dir: Optional[Path],
    data_format: str,
    refresh_data: bool,
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        cache_path = _cache_path(data_dir, "stooq", ticker, data_format)
        cached = None
        if cache_path and cache_path.exists() and not refresh_data:
            cached = _load_cached_frame(cache_path, ticker, data_format)

        if cached is None or not _cache_covers_range(cached, start, end):
            symbol = f"{ticker.lower()}.us"
            url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
            df = _read_stooq_csv(url)
            if df.empty:
                raise ValueError(f"No Stooq data for {ticker}.")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            df = df[["Close"]].rename(columns={"Close": ticker})
            if cache_path:
                _save_cached_frame(cache_path, df, data_format)
            frames.append(df)
        else:
            frames.append(cached)

    prices = pd.concat(frames, axis=1).sort_index()
    if start is not None:
        prices = prices.loc[start:]
    if end is not None:
        prices = prices.loc[:end]
    return prices.dropna()


def _to_rfc3339(ts: pd.Timestamp) -> str:
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def _fetch_alpaca_prices_remote(
    tickers: Iterable[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    feed: Optional[str],
) -> pd.DataFrame:
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError("Missing Alpaca API keys. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    params = {
        "symbols": ",".join(tickers),
        "timeframe": "1Day",
        "adjustment": "all",
        "limit": 10000,
    }
    if feed:
        params["feed"] = feed
    if start is not None:
        params["start"] = _to_rfc3339(start)
    if end is not None:
        params["end"] = _to_rfc3339(end)

    data_by_symbol: Dict[str, List[Tuple[pd.Timestamp, float]]] = {t: [] for t in tickers}
    next_page_token = None

    while True:
        if next_page_token:
            params["page_token"] = next_page_token
        resp = requests.get(base_url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        for bar in payload.get("bars", []):
            symbol = bar.get("S")
            close = bar.get("c")
            ts = pd.to_datetime(bar.get("t"), utc=True).tz_convert(None).normalize()
            if symbol in data_by_symbol and close is not None:
                data_by_symbol[symbol].append((ts, close))

        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break

    frames = []
    for ticker, rows in data_by_symbol.items():
        if not rows:
            raise ValueError(f"No Alpaca data for {ticker}.")
        df = pd.DataFrame(rows, columns=["Date", ticker]).drop_duplicates("Date")
        df = df.set_index("Date").sort_index()
        frames.append(df)

    prices = pd.concat(frames, axis=1).sort_index()
    return prices.dropna()


def fetch_alpaca_prices(
    tickers: Iterable[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    feed: Optional[str],
    data_dir: Optional[Path],
    data_format: str,
    refresh_data: bool,
) -> pd.DataFrame:
    tickers = list(tickers)
    if data_dir is None:
        return _fetch_alpaca_prices_remote(tickers, start, end, feed)

    cached_frames: Dict[str, Optional[pd.DataFrame]] = {}
    missing: List[str] = []

    for ticker in tickers:
        cache_path = _cache_path(data_dir, "alpaca", ticker, data_format)
        cached = None
        if cache_path and cache_path.exists() and not refresh_data:
            cached = _load_cached_frame(cache_path, ticker, data_format)
        if cached is None or not _cache_covers_range(cached, start, end):
            missing.append(ticker)
        cached_frames[ticker] = cached

    if missing:
        fresh_prices = _fetch_alpaca_prices_remote(missing, start, end, feed)
        for ticker in missing:
            if ticker not in fresh_prices.columns:
                raise ValueError(f"No Alpaca data for {ticker}.")
            fresh = fresh_prices[[ticker]].dropna()
            combined = _merge_cached_frame(cached_frames[ticker], fresh)
            cache_path = _cache_path(data_dir, "alpaca", ticker, data_format)
            if cache_path:
                _save_cached_frame(cache_path, combined, data_format)
            cached_frames[ticker] = combined

    frames = []
    for ticker in tickers:
        frame = cached_frames[ticker]
        if frame is None or frame.empty:
            raise ValueError(f"No Alpaca data for {ticker}.")
        frames.append(frame)

    prices = pd.concat(frames, axis=1).sort_index()
    if start is not None:
        prices = prices.loc[start:]
    if end is not None:
        prices = prices.loc[:end]
    return prices.dropna()


def inverse_vol_weights(window: pd.DataFrame, ann_factor: int) -> pd.Series:
    vol = window.std() * np.sqrt(ann_factor)
    vol = vol.replace(0.0, np.nan)
    inv = 1.0 / vol
    weights = inv / inv.sum()
    return weights.fillna(0.0)


def risk_parity_weights(cov: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    n = cov.shape[0]
    weights = np.ones(n) / n
    target = np.ones(n) / n

    for _ in range(max_iter):
        prev = weights.copy()
        port_var = weights @ cov @ weights
        if port_var <= 0:
            break
        mrc = cov @ weights
        rc = weights * mrc
        rc_target = port_var * target
        weights = weights * (rc_target / (rc + 1e-12))
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()
        if np.max(np.abs(weights - prev)) < tol:
            break

    return weights


def compute_weights(
    window: pd.DataFrame,
    portfolio: str,
    fixed_weights: Optional[pd.Series],
    ann_factor: int,
) -> pd.Series:
    if portfolio == "fixed":
        if fixed_weights is None:
            raise ValueError("Fixed weights required for fixed portfolio.")
        return fixed_weights
    if portfolio == "inverse-vol":
        return inverse_vol_weights(window, ann_factor)
    if portfolio == "risk-parity":
        cov = window.cov().values * ann_factor
        weights = risk_parity_weights(cov)
        return pd.Series(weights, index=window.columns)
    raise ValueError(f"Unknown portfolio type: {portfolio}")


def compute_portfolio(
    prices: pd.DataFrame,
    lookback: int,
    rebalance: str,
    portfolio: str,
    fixed_weights: Optional[pd.Series],
    target_vol: float,
    max_leverage: float,
    ann_factor: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().dropna()
    rebalance_dates = returns.resample(rebalance).last().index

    weights_hist = pd.DataFrame(index=rebalance_dates, columns=returns.columns, dtype=float)

    for date in rebalance_dates:
        window = returns.loc[:date].tail(lookback)
        if len(window) < lookback:
            continue
        weights = compute_weights(window, portfolio, fixed_weights, ann_factor)

        cov = window.cov().values * ann_factor
        port_vol = np.sqrt(weights.values @ cov @ weights.values)
        if target_vol and port_vol > 0:
            leverage = min(max_leverage, target_vol / port_vol)
            weights = weights * leverage

        weights_hist.loc[date] = weights

    weights_daily = weights_hist.reindex(returns.index).ffill()
    weights_daily = weights_daily.shift(1).dropna()
    port_returns = (returns.loc[weights_daily.index] * weights_daily).sum(axis=1)
    return port_returns, weights_daily


def compute_equity_drawdown(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return equity, drawdown


def apply_costs(
    returns: pd.Series,
    weights: pd.DataFrame,
    trade_cost: float,
    borrow_rate: float,
    cash_rate: float,
    ann_factor: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    if trade_cost < 0 or borrow_rate < 0 or cash_rate < 0:
        raise ValueError("Costs and rates must be non-negative.")
    weights = weights.reindex(returns.index)
    gross_leverage = weights.abs().sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    if not turnover.empty:
        # Assume initial allocation from cash.
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    trading_cost = turnover * trade_cost
    borrow_cost = (gross_leverage - 1.0).clip(lower=0.0) * (borrow_rate / ann_factor)
    cash_yield = (1.0 - gross_leverage).clip(lower=0.0) * (cash_rate / ann_factor)
    net_returns = returns - trading_cost - borrow_cost + cash_yield
    costs = pd.DataFrame(
        {
            "turnover": turnover,
            "trading_cost": trading_cost,
            "borrow_cost": borrow_cost,
            "cash_yield": cash_yield,
            "financing_impact": cash_yield - borrow_cost,
        },
        index=returns.index,
    )
    return net_returns, costs


def performance_stats(returns: pd.Series, ann_factor: int) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("No returns to analyze.")

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    _, drawdown = compute_equity_drawdown(returns)
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def plot_performance(
    returns: pd.Series,
    out_path: Path,
    title: str,
    benchmark_returns: Optional[pd.Series] = None,
    benchmark_label: Optional[str] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot. Install with "
            "`pip install matplotlib` or `pip install -e '.[plot]'`."
        ) from exc

    equity, drawdown = compute_equity_drawdown(returns)
    benchmark_equity = None
    benchmark_drawdown = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark_equity, benchmark_drawdown = compute_equity_drawdown(benchmark_returns)
    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_eq.plot(equity.index, equity.values, color="#1f77b4", label="Portfolio")
    if benchmark_equity is not None:
        label = f"Benchmark ({benchmark_label})" if benchmark_label else "Benchmark"
        ax_eq.plot(
            benchmark_equity.index,
            benchmark_equity.values,
            color="#ff7f0e",
            label=label,
        )
    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(loc="upper left")

    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.3, label="Portfolio")
    if benchmark_drawdown is not None:
        label = f"Benchmark ({benchmark_label})" if benchmark_label else "Benchmark"
        ax_dd.plot(
            benchmark_drawdown.index,
            benchmark_drawdown.values,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label=label,
        )
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(True, alpha=0.3)
    if benchmark_drawdown is not None:
        ax_dd.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.trade_cost < 0 or args.borrow_rate < 0 or args.cash_rate < 0:
        raise ValueError("Cost parameters must be non-negative.")

    tickers = _parse_tickers(args.tickers)
    benchmark = _parse_benchmark(args.benchmark)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else None

    fixed_weights = None
    if args.portfolio == "fixed":
        if args.fixed_weights:
            weights_map = _parse_fixed_weights(args.fixed_weights)
        else:
            weights_map = DEFAULT_FIXED_WEIGHTS
        missing = [t for t in tickers if t not in weights_map]
        if missing:
            raise ValueError(
                "Fixed weights missing tickers: " + ", ".join(missing)
            )
        fixed_weights = pd.Series({t: weights_map[t] for t in tickers})

    if args.data_source == "stooq":
        prices = fetch_stooq_prices(
            tickers,
            start,
            end,
            data_dir=data_dir,
            data_format=args.data_format,
            refresh_data=args.refresh_data,
        )
    else:
        prices = fetch_alpaca_prices(
            tickers,
            start,
            end,
            args.alpaca_feed,
            data_dir=data_dir,
            data_format=args.data_format,
            refresh_data=args.refresh_data,
        )

    gross_returns, weights_daily = compute_portfolio(
        prices=prices,
        lookback=args.lookback,
        rebalance=args.rebalance,
        portfolio=args.portfolio,
        fixed_weights=fixed_weights,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        ann_factor=args.ann_factor,
    )

    net_returns = gross_returns
    costs = None
    if args.trade_cost > 0 or args.borrow_rate > 0 or args.cash_rate > 0:
        net_returns, costs = apply_costs(
            gross_returns,
            weights_daily,
            trade_cost=args.trade_cost,
            borrow_rate=args.borrow_rate,
            cash_rate=args.cash_rate,
            ann_factor=args.ann_factor,
        )

    benchmark_returns = None
    benchmark_stats = None
    if benchmark:
        if benchmark in prices.columns:
            benchmark_prices = prices[benchmark].dropna()
        elif args.data_source == "stooq":
            benchmark_prices = fetch_stooq_prices(
                [benchmark],
                start,
                end,
                data_dir=data_dir,
                data_format=args.data_format,
                refresh_data=args.refresh_data,
            )[benchmark]
        else:
            benchmark_prices = fetch_alpaca_prices(
                [benchmark],
                start,
                end,
                args.alpaca_feed,
                data_dir=data_dir,
                data_format=args.data_format,
                refresh_data=args.refresh_data,
            )[benchmark]
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_returns = benchmark_returns.reindex(net_returns.index).dropna()
        if benchmark_returns.empty:
            raise ValueError(f"No benchmark data for {benchmark} in backtest period.")

    gross_stats = performance_stats(gross_returns, args.ann_factor)
    net_stats = performance_stats(net_returns, args.ann_factor)
    if benchmark_returns is not None:
        benchmark_stats = performance_stats(benchmark_returns, args.ann_factor)
    gross_leverage = weights_daily.abs().sum(axis=1)

    print("Backtest Summary")
    print("---------------")
    print(f"Data source     : {args.data_source}")
    print(f"Tickers         : {', '.join(tickers)}")
    if benchmark:
        print(f"Benchmark       : {benchmark}")
    print(f"Period          : {net_returns.index[0].date()} -> {net_returns.index[-1].date()}")
    print(f"Rebalance       : {args.rebalance}")
    print(f"Portfolio       : {args.portfolio}")
    print(f"Data cache      : {data_dir if data_dir else 'disabled'}")
    if data_dir:
        print(f"Data format     : {args.data_format}")
        print(f"Data refresh    : {'on' if args.refresh_data else 'off'}")
    print(f"Target vol      : {args.target_vol:.2%}")
    print(f"Max leverage    : {args.max_leverage:.2f}x")
    print(f"Trade cost      : {args.trade_cost * 1e4:.1f} bps")
    print(f"Borrow rate     : {args.borrow_rate:.2%}")
    print(f"Cash rate       : {args.cash_rate:.2%}")
    print("")
    print(f"Avg Gross Lev   : {gross_leverage.mean():.2f}x")
    print(f"Max Gross Lev   : {gross_leverage.max():.2f}x")
    if costs is not None:
        print(f"Avg Turnover    : {costs['turnover'].mean():.2f}x")
        print(f"Max Turnover    : {costs['turnover'].max():.2f}x")
    print("")
    if costs is not None:
        print("Performance (net)")
    else:
        print("Performance")
    print(f"Total Return    : {net_stats['total_return']:.2%}")
    print(f"CAGR            : {net_stats['ann_return']:.2%}")
    print(f"Ann. Vol        : {net_stats['ann_vol']:.2%}")
    print(f"Sharpe (rf=0)   : {net_stats['sharpe']:.2f}")
    print(f"Max Drawdown    : {net_stats['max_drawdown']:.2%}")
    if costs is not None:
        print("")
        print("Performance (gross)")
        print(f"Total Return    : {gross_stats['total_return']:.2%}")
        print(f"CAGR            : {gross_stats['ann_return']:.2%}")
        print(f"Ann. Vol        : {gross_stats['ann_vol']:.2%}")
        print(f"Sharpe (rf=0)   : {gross_stats['sharpe']:.2f}")
        print(f"Max Drawdown    : {gross_stats['max_drawdown']:.2%}")
    if benchmark_stats is not None:
        print("")
        print(f"Benchmark ({benchmark})")
        print(f"Total Return    : {benchmark_stats['total_return']:.2%}")
        print(f"CAGR            : {benchmark_stats['ann_return']:.2%}")
        print(f"Ann. Vol        : {benchmark_stats['ann_vol']:.2%}")
        print(f"Sharpe (rf=0)   : {benchmark_stats['sharpe']:.2f}")
        print(f"Max Drawdown    : {benchmark_stats['max_drawdown']:.2%}")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        net_returns.to_csv(out_dir / "portfolio_returns.csv", header=["return"])
        weights_daily.to_csv(out_dir / "weights_daily.csv")
        (1 + net_returns).cumprod().to_csv(out_dir / "equity_curve.csv", header=["equity"])
        if benchmark_returns is not None:
            benchmark_returns.to_csv(out_dir / "benchmark_returns.csv", header=["return"])
            (1 + benchmark_returns).cumprod().to_csv(
                out_dir / "benchmark_equity_curve.csv",
                header=["equity"],
            )
        if costs is not None:
            gross_returns.to_csv(out_dir / "portfolio_returns_gross.csv", header=["return"])
            (1 + gross_returns).cumprod().to_csv(
                out_dir / "equity_curve_gross.csv",
                header=["equity"],
            )
            costs.to_csv(out_dir / "costs_daily.csv")
        meta = {
            "data_source": args.data_source,
            "tickers": tickers,
            "period_start": str(net_returns.index[0].date()),
            "period_end": str(net_returns.index[-1].date()),
            "rebalance": args.rebalance,
            "portfolio": args.portfolio,
            "lookback": args.lookback,
            "benchmark": benchmark,
            "target_vol": args.target_vol,
            "max_leverage": args.max_leverage,
            "data_dir": str(data_dir) if data_dir else None,
            "data_format": args.data_format,
            "refresh_data": args.refresh_data,
            "trade_cost": args.trade_cost,
            "borrow_rate": args.borrow_rate,
            "cash_rate": args.cash_rate,
            "execution_lag_days": 1,
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if args.plot:
        plot_dir = Path(args.out) if args.out else Path.cwd()
        plot_path = plot_dir / "equity_drawdown.png"
        plot_performance(
            net_returns,
            plot_path,
            f"All Weather ({args.portfolio})",
            benchmark_returns=benchmark_returns,
            benchmark_label=benchmark,
        )
        print(f"Plot saved to   : {plot_path}")


if __name__ == "__main__":
    main()
