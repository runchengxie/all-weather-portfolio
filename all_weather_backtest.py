#!/usr/bin/env python3
"""
All Weather portfolio backtest (risk parity + vol targeting).

Data sources:
- Stooq (default, no API key)
- Alpaca (requires APCA_API_KEY_ID / APCA_API_SECRET_KEY)

Notes:
- This is an ETF proxy of the ideas in the Bridgewater document.
- Leverage costs, taxes, and trading slippage are not modeled.
"""

from __future__ import annotations

import argparse
import json
import os
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
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ETF tickers (e.g. SPY,TLT,IEF,GLD,DBC)",
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


def fetch_stooq_prices(
    tickers: Iterable[str],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        symbol = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError(f"No Stooq data for {ticker}.")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df = df[["Close"]].rename(columns={"Close": ticker})
        frames.append(df)

    prices = pd.concat(frames, axis=1).sort_index()
    if start is not None:
        prices = prices.loc[start:]
    if end is not None:
        prices = prices.loc[:end]
    return prices.dropna()


def _to_rfc3339(ts: pd.Timestamp) -> str:
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_alpaca_prices(
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

    weights_daily = weights_hist.reindex(returns.index).ffill().dropna()
    port_returns = (returns.loc[weights_daily.index] * weights_daily).sum(axis=1)
    return port_returns, weights_daily


def performance_stats(returns: pd.Series, ann_factor: int) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("No returns to analyze.")

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def main() -> None:
    args = parse_args()

    tickers = _parse_tickers(args.tickers)
    start = _parse_date(args.start)
    end = _parse_date(args.end)

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
        prices = fetch_stooq_prices(tickers, start, end)
    else:
        prices = fetch_alpaca_prices(tickers, start, end, args.alpaca_feed)

    port_returns, weights_daily = compute_portfolio(
        prices=prices,
        lookback=args.lookback,
        rebalance=args.rebalance,
        portfolio=args.portfolio,
        fixed_weights=fixed_weights,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        ann_factor=args.ann_factor,
    )

    stats = performance_stats(port_returns, args.ann_factor)
    gross_leverage = weights_daily.sum(axis=1)

    print("Backtest Summary")
    print("---------------")
    print(f"Data source     : {args.data_source}")
    print(f"Tickers         : {', '.join(tickers)}")
    print(f"Period          : {port_returns.index[0].date()} -> {port_returns.index[-1].date()}")
    print(f"Rebalance       : {args.rebalance}")
    print(f"Portfolio       : {args.portfolio}")
    print(f"Target vol      : {args.target_vol:.2%}")
    print(f"Max leverage    : {args.max_leverage:.2f}x")
    print("")
    print("Performance")
    print(f"Total Return    : {stats['total_return']:.2%}")
    print(f"CAGR            : {stats['ann_return']:.2%}")
    print(f"Ann. Vol        : {stats['ann_vol']:.2%}")
    print(f"Sharpe (rf=0)   : {stats['sharpe']:.2f}")
    print(f"Max Drawdown    : {stats['max_drawdown']:.2%}")
    print(f"Avg Gross Lev   : {gross_leverage.mean():.2f}x")
    print(f"Max Gross Lev   : {gross_leverage.max():.2f}x")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        port_returns.to_csv(out_dir / "portfolio_returns.csv", header=["return"])
        weights_daily.to_csv(out_dir / "weights_daily.csv")
        (1 + port_returns).cumprod().to_csv(out_dir / "equity_curve.csv", header=["equity"])
        meta = {
            "data_source": args.data_source,
            "tickers": tickers,
            "period_start": str(port_returns.index[0].date()),
            "period_end": str(port_returns.index[-1].date()),
            "rebalance": args.rebalance,
            "portfolio": args.portfolio,
            "lookback": args.lookback,
            "target_vol": args.target_vol,
            "max_leverage": args.max_leverage,
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
