import numpy as np
import pandas as pd

from all_weather_backtest import (
    _parse_fixed_weights,
    _parse_tickers,
    apply_costs,
    compute_portfolio,
    inverse_vol_weights,
    performance_stats,
    risk_parity_weights,
)


def test_parse_tickers_uppercase_and_strip() -> None:
    tickers = _parse_tickers(" spy, tlt , ief ")
    assert tickers == ["SPY", "TLT", "IEF"]


def test_parse_fixed_weights_normalize() -> None:
    weights = _parse_fixed_weights("SPY=1,TLT=1")
    assert np.isclose(weights["SPY"], 0.5)
    assert np.isclose(weights["TLT"], 0.5)
    assert np.isclose(sum(weights.values()), 1.0)


def test_inverse_vol_weights_handles_zero_vol() -> None:
    window = pd.DataFrame(
        {
            "A": [0.0, 0.0, 0.0, 0.0],
            "B": [0.01, -0.01, 0.01, -0.01],
            "C": [0.02, -0.02, 0.02, -0.02],
        }
    )
    weights = inverse_vol_weights(window, ann_factor=252)
    assert weights["A"] == 0.0
    assert weights["C"] > 0.0
    ratio = weights["B"] / weights["C"]
    assert 1.5 < ratio < 2.5
    assert np.isclose(weights.sum(), 1.0)


def test_risk_parity_weights_diagonal_cov() -> None:
    cov = np.array([[1.0, 0.0], [0.0, 4.0]])
    weights = risk_parity_weights(cov)
    expected = np.array([2.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(weights, expected, atol=1e-2)


def test_compute_portfolio_inverse_vol_daily_rebalance() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "BBB": [50, 50.5, 51.0, 51.4, 51.8, 52.5, 52.9, 53.3, 54.0, 54.5],
        },
        index=dates,
    )

    port_returns, weights = compute_portfolio(
        prices=prices,
        lookback=3,
        rebalance="D",
        portfolio="inverse-vol",
        fixed_weights=None,
        target_vol=0.0,
        max_leverage=2.0,
        ann_factor=252,
    )

    assert not port_returns.empty
    assert port_returns.index.equals(weights.index)
    assert set(weights.columns) == {"AAA", "BBB"}
    assert np.allclose(weights.sum(axis=1).values, 1.0)


def test_compute_portfolio_applies_one_day_lag() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"AAA": [100, 101, 102, 103, 104, 105]}, index=dates)
    fixed_weights = pd.Series({"AAA": 1.0})

    port_returns, weights = compute_portfolio(
        prices=prices,
        lookback=2,
        rebalance="D",
        portfolio="fixed",
        fixed_weights=fixed_weights,
        target_vol=0.0,
        max_leverage=2.0,
        ann_factor=252,
    )

    assert port_returns.index[0] == dates[3]
    assert weights.index[0] == dates[3]


def test_apply_costs_trading_and_cash() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, 0.01, 0.01], index=dates)
    weights = pd.DataFrame({"AAA": [1.0, 1.0, 0.5]}, index=dates)

    net_returns, costs = apply_costs(
        returns,
        weights,
        trade_cost=0.001,
        borrow_rate=0.1,
        cash_rate=0.05,
        ann_factor=10,
    )

    expected_turnover = pd.Series([1.0, 0.0, 0.5], index=dates)
    expected_trading_cost = expected_turnover * 0.001
    expected_cash_yield = pd.Series([0.0, 0.0, 0.5 * 0.05 / 10], index=dates)
    expected_net = returns - expected_trading_cost + expected_cash_yield

    assert np.allclose(costs["turnover"].values, expected_turnover.values)
    assert np.allclose(net_returns.values, expected_net.values)


def test_apply_costs_borrow_cost() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    returns = pd.Series([0.0, 0.0], index=dates)
    weights = pd.DataFrame({"AAA": [1.5, 1.5]}, index=dates)

    net_returns, costs = apply_costs(
        returns,
        weights,
        trade_cost=0.0,
        borrow_rate=0.1,
        cash_rate=0.0,
        ann_factor=10,
    )

    expected_borrow = pd.Series([0.5 * 0.1 / 10, 0.5 * 0.1 / 10], index=dates)
    assert np.allclose(costs["borrow_cost"].values, expected_borrow.values)
    assert np.allclose(net_returns.values, (-expected_borrow).values)


def test_performance_stats_basic() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    stats = performance_stats(returns, ann_factor=252)

    expected_total = (1.01 * 0.98 * 1.03) - 1.0
    expected_ann_return = (1 + expected_total) ** (252 / len(returns)) - 1.0
    expected_ann_vol = returns.std() * np.sqrt(252)

    assert np.isclose(stats["total_return"], expected_total)
    assert np.isclose(stats["ann_return"], expected_ann_return)
    assert np.isclose(stats["ann_vol"], expected_ann_vol)
    assert np.isclose(stats["max_drawdown"], -0.02)
    assert np.isfinite(stats["sharpe"])
