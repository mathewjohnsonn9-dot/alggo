"""Compare entry hypotheses and one loss rule with the uploaded strategy.

Run with one stock's CSV files or folder. Cash and positions carry between days.
All signals use information available at that close; fills use the next row's open.
Slippage is an assumed cost, not a measured spread. Profit-only sale fills are an
opening-price approximation to conditional/limit orders; actual fills can differ.
All supplied bars are used in their supplied timezone. No settlement, tax, interest,
dividend, or corporate-action model is included. Use consistently adjusted data.
"""
import argparse
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "/home/fariborz/Downloads/live/test_stock/AAPL/AAPL_2025-07-07_1min.csv"
INITIAL_CAPITAL = 10_000.0
BUY_LIMIT = 100
CHUNK_SHARES = 25
WINDOW = 25
DAILY_LOSS_LIMIT = 0.005  # Illustrative 0.5% trigger; applies only to Daily loss rule
STRATEGIES = ("Original", "Recovery", "Trend", "Recovery + trend", "Daily loss rule", "Buy and hold")


def load_prices(paths):
    files = []
    for name in paths:
        path = Path(name)
        files.extend(sorted(path.glob("*_1min*.csv")) if path.is_dir() else [path])
    files = sorted(set(files))
    if not files:
        raise ValueError("No *_1min*.csv files found")
    tickers = {match.group(1) for file in files
               if (match := re.match(r"(.+)_\d{4}-\d{2}-\d{2}_1min", file.name))}
    if len(tickers) > 1:
        raise ValueError("Supply only one stock at a time")
    data = pd.concat([pd.read_csv(file, usecols=["timestamp", "open", "close"])
                      for file in files], ignore_index=True).drop_duplicates()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data[["open", "close"]] = data[["open", "close"]].apply(pd.to_numeric)
    if data.empty or data["timestamp"].isna().any():
        raise ValueError("Missing prices or timestamps")
    if data["timestamp"].duplicated().any():
        raise ValueError("Conflicting rows have the same timestamp")
    if not np.isfinite(data[["open", "close"]]).all().all() or (data[["open", "close"]] <= 0).any().any():
        raise ValueError("Open and close prices must be finite and positive")
    return data.sort_values("timestamp").reset_index(drop=True)


def add_signals(data):
    data = data.copy()
    close = data["close"]
    day = data["timestamp"].dt.date
    # Exactly the original 25-price window ending 25 bars before this close.
    local_mean = close.rolling(WINDOW).mean().shift(WINDOW)
    local_std = close.rolling(WINDOW).std(ddof=0).shift(WINDOW)
    data["lower"] = local_mean - 2 * local_std
    data["upper"] = local_mean + 2 * local_std
    day_mean = close.groupby(day).transform(lambda s: s.expanding().mean())
    day_std = close.groupby(day).transform(lambda s: s.expanding().std(ddof=0))
    data["global_upper"] = day_mean + day_std
    global_ok = close < data["global_upper"]
    touch = close <= data["lower"]
    # Buy only on a return inside the lower band, after a breach on the prior bar.
    recovery = (close.shift(1) <= data["lower"].shift(1)) & (close > data["lower"])
    # Compare the day's mean with its value 25 bars earlier in the SAME day.
    trend = day_mean >= day_mean.groupby(day).shift(WINDOW)
    data["Original"] = touch & global_ok
    data["Recovery"] = recovery & global_ok
    data["Trend"] = touch & global_ok & trend
    data["Recovery + trend"] = recovery & global_ok & trend
    data["sell_signal"] = close >= data["upper"]
    return data


def simulate(data, strategy, slippage_bps=2.0, fee_per_share=0.0):
    if not 0 <= slippage_bps < 10_000 or fee_per_share < 0:
        raise ValueError("Costs must be nonnegative and slippage below 100%")
    slip = slippage_bps / 10_000
    cash = INITIAL_CAPITAL
    held = []
    realized = 0.0
    pending = "BUY" if strategy == "Buy and hold" else None
    equity, fills = [], []
    current_day, day_start_value, paused = None, INITIAL_CAPITAL, False
    for i, bar in enumerate(data.itertuples(index=False)):
        if bar.timestamp.date() != current_day:
            current_day = bar.timestamp.date()
            day_start_value = equity[-1] if equity else INITIAL_CAPITAL
            paused = False
        if pending == "BUY":
            price = float(bar.open) * (1 + slip) + fee_per_share
            order_size = BUY_LIMIT if strategy == "Buy and hold" else CHUNK_SHARES
            qty = min(order_size, BUY_LIMIT - len(held), max(0, int(cash / price)))
            if qty:
                cash -= qty * price
                held.extend([price] * qty)
                fills.append((bar.timestamp, "BUY", qty, price, 0.0))
        elif pending in ("SELL", "RISK_EXIT"):
            price = float(bar.open) * (1 - slip) - fee_per_share
            selected = held.copy() if pending == "RISK_EXIT" else [entry for entry in held if entry < price]
            if selected:
                for entry in selected:
                    held.remove(entry)
                pnl = len(selected) * price - sum(selected)
                cash += len(selected) * price
                realized += pnl
                action = "RISK_EXIT" if pending == "RISK_EXIT" else "SELL"
                fills.append((bar.timestamp, action, len(selected), price, pnl))
        pending = None
        equity.append(cash + len(held) * float(bar.close))
        if strategy != "Buy and hold":
            signal_name = "Original" if strategy == "Daily loss rule" else strategy
            # A close-based trigger fills next open; gaps can exceed the 0.5% loss.
            if strategy == "Daily loss rule" and equity[-1] <= day_start_value * (1 - DAILY_LOSS_LIMIT):
                paused = True
                pending = "RISK_EXIT" if held else None
            elif bool(data.at[i, signal_name]) and len(held) < BUY_LIMIT and not paused:
                pending = "BUY"
            elif bool(data.at[i, "sell_signal"]) and held:
                pending = "SELL"

    curve = pd.Series(equity, index=data["timestamp"], name=strategy)
    peaks = curve.cummax().clip(lower=INITIAL_CAPITAL)
    open_pnl = sum(float(data["close"].iloc[-1]) - entry for entry in held)
    net_pnl = curve.iloc[-1] - INITIAL_CAPITAL
    if cash < -1e-7 or len(held) > BUY_LIMIT or not math.isclose(net_pnl, realized + open_pnl, abs_tol=1e-7):
        raise AssertionError("Cash or position accounting failed")
    metrics = {"Strategy": strategy, "Ending account": curve.iloc[-1], "Net PnL": net_pnl,
               "Return %": 100 * net_pnl / INITIAL_CAPITAL,
               "Max drawdown %": 100 * (1 - curve / peaks).max(),
               "Realized PnL": realized, "Open PnL": open_pnl, "Open shares": len(held),
               "Buy orders": sum(fill[1] == "BUY" for fill in fills),
               "Sell orders": sum(fill[1] != "BUY" for fill in fills),
               "Risk exits": sum(fill[1] == "RISK_EXIT" for fill in fills)}
    trades = pd.DataFrame(fills, columns=["timestamp", "action", "shares", "fill_price", "net_pnl"])
    return metrics, curve, trades


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=[DATA_PATH], help="One stock's CSV files or folder")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Assumed cost per side; 2 = 0.02%%")
    parser.add_argument("--fee-per-share", type=float, default=0.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    data = add_signals(load_prices(args.paths))
    summaries, curves = [], []
    for strategy in STRATEGIES:
        metrics, curve, _ = simulate(data, strategy, args.slippage_bps, args.fee_per_share)
        summaries.append(metrics)
        curves.append(curve)
    print(f"Dates: {data['timestamp'].iloc[0]} through {data['timestamp'].iloc[-1]}")
    print(f"Starting account: ${INITIAL_CAPITAL:,.2f}; slippage: {args.slippage_bps:g} bps per side")
    print("Cash and positions carry across days. All supplied bars are used.")
    print(f"Daily loss rule permits losing sales after a {100 * DAILY_LOSS_LIMIT:g}% daily account drop; gaps can exceed this.")
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("Net PnL includes unsold shares valued at the final close; there is no forced final sale.")
    print("Exploratory comparison: no configuration is established as profitable on unseen data.")
    if not args.no_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for column, label in [("close", "Close"), ("lower", "Local lower"),
                              ("upper", "Local upper"), ("global_upper", "Day mean + std")]:
            axes[0].plot(data["timestamp"], data[column], label=label)
        axes[0].set_ylabel("Price ($)")
        axes[0].legend()
        for curve in curves:
            axes[1].plot(curve.index, curve.values, label=curve.name)
        axes[1].axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Account value ($)")
        axes[1].set_xlabel("Time (CSV timestamps)")
        axes[1].legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
