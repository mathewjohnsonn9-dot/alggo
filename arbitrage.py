#!/usr/bin/env python3
"""Rolling two-asset price-gap backtest. No AE or separate training script.

Install:  python -m pip install numpy pandas matplotlib
Check:    python trade.py --self-test
Run:      python trade.py --data-root data --window 60 --mode percent --threshold-pct 0.2
Or:       python trade.py --asset-data AAPL.csv --reference-data VOO.csv --mode zscore

Input: data/AAPL/*.csv and data/VOO/*.csv (nested year folders also work).
Each CSV needs timestamp, open, close. With --execution next_close only
timestamp and close are required. Naive timestamps default to UTC; override
with --input-timezone America/New_York if necessary. Bars must be one minute.

At close t, fit AAPL = alpha * VOO + beta on the preceding W consecutive
minutes, excluding t. Gap = actual AAPL close - estimated AAPL close.
Percentage gap = 100 * gap / estimate. Z-score uses the preceding Z gaps,
each made with its own historical fit; current gap is excluded from its stats.
Negative gap below the lower threshold buys; positive gap above the upper
threshold sells one profitable lot/chunk. VOO is a reference, never traded.

Signals use completed closes. Orders are eligible only at the next minute's
open (or close with --execution next_close), and expire on missing minutes
or overnight. Sell orders have a net breakeven limit fixed at submission.
They expire if the next quote cannot fill profitably; intrabar highs are
never used. Fees and adverse slippage apply once to each fill.

Cash and positions persist across days. Fits and gap statistics restart
after a missing minute or session boundary. Unsold positions are marked to
the final close, even at a loss. No forced liquidation or short selling.

Outputs in one folder: results.csv, trades.csv, summary.json, equity.png.
Buy-and-hold starts on the same executable bar, with the same initial cash
and costs: full-budget AAPL, full-budget VOO, and share-cap-matched AAPL.
Benchmarks are price returns. Dividends, splits and taxes are not simulated;
use consistently adjusted OHLC input or account for corporate actions first.
Minute timestamps must identify bars consistently for both assets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
import json
import math
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    asset: str = "AAPL"
    reference: str = "VOO"
    window: int = 60
    mode: str = "percent"
    threshold_pct: float = 0.2
    z_threshold: float = 2.0
    z_window: int | None = None
    initial_cash: float = 10_000.0
    chunk: int = 5
    max_shares: int = 30
    commission_bps: float = 0.0
    fee: float = 0.0
    slippage_bps: float = 1.0
    execution: str = "next_open"
    input_timezone: str = "UTC"
    market_timezone: str = "America/New_York"
    start: str | None = None
    end: str | None = None

    def validate(self):
        if self.asset == self.reference:
            raise ValueError("Asset and reference must be different tickers.")
        if self.window < 3 or (self.z_window is not None and self.z_window < 2):
            raise ValueError("window must be >= 3 and z-window >= 2.")
        if self.chunk < 1 or self.max_shares < 1:
            raise ValueError("chunk and max-shares must be positive integers.")
        for name in ("initial_cash", "threshold_pct", "z_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.fee) or self.fee < 0:
            raise ValueError("fee must be finite and nonnegative.")
        for name in ("commission_bps", "slippage_bps"):
            if not 0 <= getattr(self, name) < 10_000:
                raise ValueError(f"{name} must be in [0, 10000).")
        ZoneInfo(self.input_timezone)
        ZoneInfo(self.market_timezone)
        for value in (self.start, self.end):
            if value is not None:
                date.fromisoformat(value)
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end.")


def read_asset(path: Path, cfg: Config) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No CSV files found in {path}")
    pieces = []
    required = ["close", "open"] if cfg.execution == "next_open" else ["close"]
    for file in files:
        raw = pd.read_csv(file)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        ts_col = next((c for c in ("timestamp", "datetime", "date", "time")
                       if c in raw), None)
        if ts_col is None or any(c not in raw for c in required):
            raise ValueError(f"{file}: need timestamp and {', '.join(required)} columns")
        text = raw[ts_col].astype(str).str.strip()
        aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        if aware.any() and not aware.all():
            raise ValueError(f"{file}: do not mix aware and naive timestamps")
        if pd.api.types.is_numeric_dtype(raw[ts_col]):
            raise ValueError(f"{file}: timestamp must be an ISO date/time, not an epoch number")
        if aware.all():
            idx = pd.DatetimeIndex(pd.to_datetime(text, format="mixed", utc=True))
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(text, format="mixed"))
            idx = idx.tz_localize(cfg.input_timezone, ambiguous="raise", nonexistent="raise")
            idx = idx.tz_convert("UTC")
        if idx.hasnans or not (idx == idx.floor("min")).all():
            raise ValueError(f"{file}: invalid timestamp or bar not on a minute boundary")
        frame = raw[required].apply(pd.to_numeric, errors="raise")
        frame.index = idx
        if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError(f"{file}: prices must be finite and strictly positive")
        pieces.append(frame)
    result = pd.concat(pieces).sort_index()
    duplicated = result.index.duplicated(keep=False)
    if duplicated.any():
        if (result[duplicated].groupby(level=0).nunique() > 1).any().any():
            raise ValueError(f"{path}: conflicting prices at a duplicate timestamp")
        result = result[~result.index.duplicated(keep="last")]
    result.index.name = "timestamp"
    return result.astype(float)


def load_pair(asset_path: Path, reference_path: Path, cfg: Config) -> pd.DataFrame:
    left = read_asset(asset_path, cfg).add_prefix("asset_")
    right = read_asset(reference_path, cfg).add_prefix("reference_")
    data = left.join(right, how="inner").sort_index()
    local = data.index.tz_convert(cfg.market_timezone)
    minute = local.hour * 60 + local.minute
    # Actual supplied bars determine holidays and early closes; no prices are filled in.
    data = data[(local.dayofweek < 5) & (minute >= 570) & (minute < 960)]
    if data.empty:
        raise ValueError("No aligned regular-session bars. Check timestamps/timezones.")
    return data


def make_signals(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = data.copy()
    names = ("alpha", "beta", "estimate", "gap", "gap_pct", "gap_mean", "gap_std", "zscore")
    values = np.full((len(out), len(names)), np.nan)
    times = out.index.asi8
    starts = np.r_[0, np.flatnonzero(np.diff(times) != 60_000_000_000) + 1]
    stops = np.r_[starts[1:], len(out)]
    zw = cfg.z_window or cfg.window
    for begin, stop in zip(starts, stops):
        block = out.iloc[begin:stop]
        x = block["reference_close"].to_numpy()
        y = block["asset_close"].to_numpy()
        # Center at the first price to reduce cancellation in price-level covariance.
        px, py = pd.Series(x - x[0]).shift(1), pd.Series(y - y[0]).shift(1)
        mx, my = px.rolling(cfg.window).mean(), py.rolling(cfg.window).mean()
        variance = px.rolling(cfg.window).var(ddof=0)
        covariance = (px * py).rolling(cfg.window).mean() - mx * my
        alpha = covariance / variance.where(variance > 1e-14)
        beta = (y[0] + my) - alpha * (x[0] + mx)
        estimate = alpha * x + beta
        estimate = estimate.where(estimate > 0)
        gap = pd.Series(y) - estimate
        history = gap.shift(1).rolling(zw)
        mean, std = history.mean(), history.std(ddof=1)
        z = (gap - mean) / std.where(std > 1e-12)
        values[begin:stop] = np.column_stack(
            [alpha, beta, estimate, gap, 100 * gap / estimate, mean, std, z])
    for col, val in zip(names, values.T):
        out[col] = val
    metric = out["gap_pct"] if cfg.mode == "percent" else out["zscore"]
    threshold = cfg.threshold_pct if cfg.mode == "percent" else cfg.z_threshold
    out["ready"] = np.isfinite(metric) & np.isfinite(out["estimate"])
    out["buy_signal"] = out["ready"] & (out["gap"] < 0) & (metric < -threshold)
    out["sell_signal"] = out["ready"] & (out["gap"] > 0) & (metric > threshold)
    return out


@dataclass
class Lot:
    lot_id: int
    qty: int
    unit_cost: float  # Includes allocated entry costs.


class Account:
    def __init__(self, cfg: Config, cap: int | None):
        self.cfg, self.cap = cfg, cap
        self.cash = cfg.initial_cash
        self.realized = 0.0
        self.lots: list[Lot] = []
        self.next_id = 1

    @property
    def shares(self):
        return sum(lot.qty for lot in self.lots)

    @property
    def basis(self):
        return sum(lot.qty * lot.unit_cost for lot in self.lots)

    def available(self, raw_price: float, requested: int):
        unit = raw_price * (1 + self.cfg.slippage_bps / 10_000)
        unit *= 1 + self.cfg.commission_bps / 10_000
        affordable = max(0, int(math.floor((self.cash - self.cfg.fee) / unit)))
        room = requested if self.cap is None else max(0, self.cap - self.shares)
        return min(requested, room, affordable)

    def buy(self, raw_price: float, requested: int):
        qty = self.available(raw_price, requested)
        if qty <= 0:
            return None
        price = raw_price * (1 + self.cfg.slippage_bps / 10_000)
        fee = qty * price * self.cfg.commission_bps / 10_000 + self.cfg.fee
        cost = qty * price + fee
        self.cash -= cost
        lot = Lot(self.next_id, qty, cost / qty)
        self.lots.append(lot)
        self.next_id += 1
        return dict(side="buy", lot_id=lot.lot_id, qty=qty, price=price, fee=fee,
                    realized_pnl=0.0, cash_after=self.cash, shares_after=self.shares)

    def sell_order(self, close: float):
        # One oldest profitable lot per signal; losing lots stay open.
        multiplier = (1 - self.cfg.slippage_bps / 10_000)
        multiplier *= (1 - self.cfg.commission_bps / 10_000)
        for lot in self.lots:
            qty = min(self.cfg.chunk, lot.qty)
            limit = (qty * lot.unit_cost + self.cfg.fee) / (qty * multiplier)
            limit *= 1 + 1e-12  # Strictly profitable, rather than exactly breakeven.
            if close >= limit:
                return dict(side="sell", lot_id=lot.lot_id, qty=qty, limit=limit)
        return None

    def sell(self, order: dict, raw_price: float):
        if raw_price < order["limit"]:
            return None
        lot = next(lot for lot in self.lots if lot.lot_id == order["lot_id"])
        qty = order["qty"]
        assert 0 < qty <= lot.qty
        price = raw_price * (1 - self.cfg.slippage_bps / 10_000)
        fee = qty * price * self.cfg.commission_bps / 10_000 + self.cfg.fee
        proceeds = qty * price - fee
        pnl = proceeds - qty * lot.unit_cost
        assert pnl > 0, "Sell limit failed to preserve positive net profit"
        self.cash += proceeds
        self.realized += pnl
        lot.qty -= qty
        if lot.qty == 0:
            self.lots.remove(lot)
        return dict(side="sell", lot_id=lot.lot_id, qty=qty, price=price, fee=fee,
                    realized_pnl=pnl, cash_after=self.cash, shares_after=self.shares)


def simulate(features: pd.DataFrame, cfg: Config):
    local_dates = features.index.tz_convert(cfg.market_timezone).strftime("%Y-%m-%d")
    mask = np.ones(len(features), dtype=bool)
    if cfg.start:
        mask &= local_dates >= cfg.start
    if cfg.end:
        mask &= local_dates <= cfg.end
    data = features.loc[mask].copy()
    if len(data) < 2:
        raise ValueError("Fewer than two aligned bars within the requested evaluation dates.")
    next_ok = np.r_[np.diff(data.index.asi8) == 60_000_000_000, False]
    first = np.flatnonzero(data["ready"].to_numpy() & next_ok)
    if len(first) == 0:
        raise ValueError("No executable evaluation period after indicator warm-up. "
                         "Need consecutive one-minute bars (about 2*window for zscore).")
    data = data.iloc[first[0]:].copy()
    book = Account(cfg, cfg.max_shares)
    benchmarks = [Account(cfg, None), Account(cfg, None), Account(cfg, cfg.max_shares)]
    pending = None
    trades = []
    counts = Counter()
    balances = np.zeros((len(data), 9))
    decisions = []
    order_states = []
    limits = np.full(len(data), np.nan)
    close = data["asset_close"].to_numpy()
    reference_close = data["reference_close"].to_numpy()
    fill_col = "open" if cfg.execution == "next_open" else "close"
    asset_quotes = data[f"asset_{fill_col}"].to_numpy()
    reference_quotes = data[f"reference_{fill_col}"].to_numpy()
    buy, sell = data["buy_signal"].to_numpy(), data["sell_signal"].to_numpy()
    times = data.index
    for i, ts in enumerate(times):
        state = "none"
        contiguous = i > 0 and ts - times[i - 1] == pd.Timedelta(minutes=1)
        if i == 1:
            # first[0] guarantees this is the shared next-minute execution bar.
            assert contiguous
            for b, quote in zip(benchmarks, (asset_quotes[i], reference_quotes[i], asset_quotes[i])):
                b.buy(quote, 2**60)
        if pending is not None:
            if not contiguous:
                state = "expired_missing_minute_or_session"
            else:
                if pending["side"] == "buy":
                    event = book.buy(asset_quotes[i], pending["qty"])
                    state = "filled_buy" if event else "blocked_cash_at_fill"
                else:
                    event = book.sell(pending, asset_quotes[i])
                    state = "filled_sell" if event else "expired_profit_limit"
                if event is not None:
                    event.update(signal_timestamp=pending["signal_timestamp"].isoformat(),
                                 fill_timestamp=ts.isoformat(),
                                 limit=pending.get("limit", None))
                    trades.append(event)
            counts[state] += 1
            pending = None

        decision = "no_signal" if data["ready"].iloc[i] else "warmup"
        if buy[i]:
            if book.shares >= cfg.max_shares:
                decision = "blocked_share_cap"
            elif book.available(close[i], cfg.chunk) == 0:
                decision = "blocked_cash"
            else:
                pending = dict(side="buy", qty=min(cfg.chunk, cfg.max_shares - book.shares),
                               signal_timestamp=ts)
                decision = "submit_buy"
        elif sell[i]:
            pending = book.sell_order(close[i])
            if pending:
                pending["signal_timestamp"] = ts
                limits[i] = pending["limit"]
                decision = "submit_sell_limit"
            else:
                decision = "no_profitable_lot"
        counts[decision] += 1
        equity = book.cash + book.shares * close[i]
        unrealized = book.shares * close[i] - book.basis
        if book.cash < -1e-7 or book.shares > cfg.max_shares:
            raise AssertionError("Cash or share constraint violated")
        if not math.isclose(equity - cfg.initial_cash, book.realized + unrealized, abs_tol=1e-6):
            raise AssertionError("Cash, realized P&L and inventory basis do not reconcile")
        bh = [b.cash + b.shares * p for b, p in
              zip(benchmarks, (close[i], reference_close[i], close[i]))]
        balances[i] = [book.cash, book.shares, book.basis, book.realized, unrealized, equity, *bh]
        decisions.append(decision)
        order_states.append(state)
    if pending is not None:
        counts["cancelled_at_end"] += 1
        decisions[-1] += "_cancelled_at_end"
    columns = ["cash", "shares", "cost_basis", "realized_pnl", "unrealized_pnl", "equity",
               "buy_hold_asset", "buy_hold_reference", "buy_hold_asset_capped"]
    for name, val in zip(columns, balances.T):
        data[name] = val
    data["decision"] = decisions
    data["order_status"] = order_states
    data["submitted_sell_limit"] = limits
    trade_columns = ["signal_timestamp", "fill_timestamp", "side", "lot_id", "qty", "price",
                     "fee", "realized_pnl", "cash_after", "shares_after", "limit"]
    trades = pd.DataFrame(trades, columns=trade_columns)

    def performance(curve):
        values = curve.to_numpy()
        days = (times[-1] - times[0]).total_seconds() / 86400
        return dict(final_equity=float(values[-1]), pnl=float(values[-1] - cfg.initial_cash),
                    return_pct=float(100 * (values[-1] / cfg.initial_cash - 1)),
                    max_drawdown_pct=float(100 * np.min(values / np.maximum.accumulate(values) - 1)),
                    cagr_pct=float(100 * ((values[-1] / cfg.initial_cash) ** (365.25 / days) - 1))
                    if days >= 365.25 else None)

    curves = ["equity", "buy_hold_asset", "buy_hold_reference", "buy_hold_asset_capped"]
    summary = dict(parameters=asdict(cfg), start=times[0].isoformat(), end=times[-1].isoformat(),
                   evaluated_bars=len(data), strategy=performance(data["equity"]),
                   benchmarks={name: performance(data[name]) for name in curves[1:]},
                   final_cash=book.cash, final_shares=book.shares,
                   realized_pnl=book.realized, unrealized_pnl=float(data["unrealized_pnl"].iloc[-1]),
                   buy_fills=int((trades["side"] == "buy").sum()),
                   sell_fills=int((trades["side"] == "sell").sum()), order_counts=dict(counts),
                   notes=["Regression and z-score statistics exclude the current observation.",
                          "Fits restart after missing minutes; inventory and cash do not reset.",
                          "Profit-only exits can leave losing positions open; equity includes them.",
                          "Price returns only: dividends, corporate actions and taxes are not modeled.",
                          "All benchmarks start on the same next-minute quote, using the same costs."])
    yearly = []
    previous = {name: cfg.initial_cash for name in curves}
    for year, group in data.groupby(data.index.tz_convert(cfg.market_timezone).year):
        row = {"year": int(year), "first_timestamp": group.index[0].isoformat(),
               "last_timestamp": group.index[-1].isoformat()}
        for name in curves:
            final = float(group[name].iloc[-1])
            row[name + "_return_pct"] = 100 * (final / previous[name] - 1)
            previous[name] = final
        yearly.append(row)
    summary["yearly_returns"] = yearly
    return data, trades, summary


def save_results(data, trades, summary, folder: Path, plot=True):
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "results.csv", index_label="timestamp")
    trades.to_csv(folder / "trades.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shown = data.iloc[np.unique(np.r_[np.linspace(0, len(data) - 1, min(4000, len(data)), dtype=int),
                                         len(data) - 1])]
        cfg = summary["parameters"]
        fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
        for name, label in [("equity", "Gap strategy"), ("buy_hold_asset", cfg["asset"] + " buy & hold"),
                            ("buy_hold_reference", cfg["reference"] + " buy & hold"),
                            ("buy_hold_asset_capped", cfg["asset"] + " buy & hold, same share cap")]:
            ax.plot(shown.index, shown[name], label=label, linewidth=1.2)
        ax.set(title=f"{cfg['asset']} / {cfg['reference']}: marked portfolio equity",
               ylabel="Equity ($)", xlabel="Time (UTC)")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(folder / "equity.png", dpi=160)
        plt.close(fig)


def self_test():
    """Offline behavioral checks, including execution gaps and future-data invariance."""
    from dataclasses import replace
    cfg = Config(window=5, z_window=4, slippage_bps=0)
    cfg.validate()
    idx = pd.date_range("2025-01-02 14:30", periods=90, freq="min", tz="UTC")
    x = 500 + np.arange(90) * 0.02 + np.sin(np.arange(90)) * 0.03
    y = 2 * x + 10
    y[15] -= 5
    frame = pd.DataFrame(dict(asset_close=y, asset_open=y,
                              reference_close=x, reference_open=x), index=idx)
    signals = make_signals(frame, cfg)
    assert abs(signals.iloc[15]["alpha"] - 2) < 1e-8
    assert abs(signals.iloc[15]["beta"] - 10) < 1e-5
    assert abs(signals.iloc[15]["gap"] + 5) < 1e-8
    assert signals.iloc[15]["buy_signal"]
    # Independent direct least-squares and z-score calculations.
    rng = np.random.default_rng(17)
    noisy = frame.copy()
    noisy["asset_close"] += rng.normal(0, 0.6, len(frame))
    for mode in ("percent", "zscore"):
        c = replace(cfg, mode=mode)
        full = make_signals(noisy, c)
        part = make_signals(noisy.iloc[:60], c)
        pd.testing.assert_frame_equal(full.iloc[:60], part)
        j = 40
        fit = np.linalg.lstsq(np.c_[x[j-c.window:j], np.ones(c.window)],
                              noisy["asset_close"].iloc[j-c.window:j], rcond=None)[0]
        assert np.allclose(full.iloc[j][["alpha", "beta"]].astype(float), fit, atol=1e-6)
        hist = full["gap"].iloc[j-c.z_window:j]
        assert abs(full["zscore"].iloc[j] - (full["gap"].iloc[j] - hist.mean()) / hist.std()) < 1e-8
        a, ta, _ = simulate(full, c)
        b, tb, _ = simulate(part, c)
        pd.testing.assert_frame_equal(a.loc[b.index, ["cash", "shares", "equity"]],
                                      b[["cash", "shares", "equity"]])
        pd.testing.assert_frame_equal(ta[ta["fill_timestamp"] <= b.index[-1].isoformat()].reset_index(drop=True),
                                      tb.reset_index(drop=True))

    # Force: buy, profitable sell signal, unprofitable next-open expiration,
    # profitable later fill, then an unsold loss held across a market night.
    times = idx[:6].append(pd.date_range("2025-01-03 14:30", periods=2, freq="min", tz="UTC"))
    fixture = pd.DataFrame(dict(asset_open=[100, 100, 98, 104, 100, 90, 80, 80],
                                asset_close=[100, 102, 103, 100, 90, 90, 80, 80],
                                reference_open=200., reference_close=200., ready=True,
                                buy_signal=[True, False, False, True, False, True, False, False],
                                sell_signal=[False, True, True, False, False, False, False, False]), index=times)
    result, trades, summary = simulate(fixture, cfg)
    assert trades["side"].tolist() == ["buy", "sell", "buy"]
    assert (pd.to_datetime(trades.fill_timestamp) > pd.to_datetime(trades.signal_timestamp)).all()
    assert summary["order_counts"]["expired_profit_limit"] == 1
    assert summary["order_counts"]["expired_missing_minute_or_session"] == 1
    assert math.isclose(summary["realized_pnl"], 20)
    assert math.isclose(summary["unrealized_pnl"], -100)
    assert math.isclose(summary["strategy"]["pnl"], -80)
    assert summary["final_shares"] == 5
    assert result.buy_hold_asset.iloc[0] == cfg.initial_cash
    assert result.buy_hold_asset.iloc[1] == 10_200

    # Cash, cap, costs, and selecting a profitable lot while retaining a losing one.
    c = replace(cfg, initial_cash=1240, max_shares=12, fee=1, commission_bps=5, slippage_bps=2)
    book = Account(c, c.max_shares)
    assert book.buy(100, 5)["qty"] == 5
    assert book.buy(100, 5)["qty"] == 5
    assert book.buy(100, 5)["qty"] == 2
    assert book.buy(100, 5) is None and book.cash >= 0
    book = Account(c, c.max_shares)
    book.buy(110, 5)
    book.buy(90, 5)
    order = book.sell_order(100)
    assert order["lot_id"] == 2
    assert book.sell(order, order["limit"] - 0.01) is None
    sold = book.sell(order, 100)
    assert sold["realized_pnl"] > 0 and book.shares == 5
    equity = book.cash + book.shares * 80
    assert math.isclose(equity - c.initial_cash, book.realized + 5 * 80 - book.basis, abs_tol=1e-8)
    poor = Account(replace(c, initial_cash=250), 30)
    assert poor.buy(100, 5)["qty"] == 2 and poor.cash >= 0

    # Missing minutes reset fits; loading respects time zones, overlaps and years.
    missing = make_signals(frame.drop(idx[20]), cfg)
    assert not missing.loc[idx[21:26], "ready"].any()
    try:
        simulate(signals, replace(cfg, start="2026-01-01"))
    except ValueError as exc:
        assert "evaluation dates" in str(exc)
    else:
        raise AssertionError("Empty evaluation dates were accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for ticker, prefix in (("AAPL", "asset"), ("VOO", "reference")):
            folder = root / ticker / "2025"
            folder.mkdir(parents=True)
            raw = frame[[prefix + "_open", prefix + "_close"]].copy()
            raw.columns = ["open", "close"]
            raw.index = raw.index.tz_convert("America/New_York").tz_localize(None)
            raw.to_csv(folder / "day.csv", index_label="timestamp")
            raw.iloc[:3].to_csv(folder / "overlap.csv", index_label="timestamp")
        loaded = load_pair(root / "AAPL", root / "VOO", replace(cfg, input_timezone="America/New_York"))
        pd.testing.assert_frame_equal(loaded, frame, check_freq=False, check_names=False)
        raw.iloc[:3].assign(close=1).to_csv(folder / "conflict.csv", index_label="timestamp")
        try:
            read_asset(root / "VOO", replace(cfg, input_timezone="America/New_York"))
        except ValueError as exc:
            assert "conflicting" in str(exc)
        else:
            raise AssertionError("Conflicting duplicates were accepted")
        save_results(result, trades, summary, root / "output", plot=True)
        assert (root / "output" / "equity.png").stat().st_size > 0
        json.loads((root / "output" / "summary.json").read_text())
    print("PASS: rolling OLS, prior-gap z-score, future-data invariance, next-minute fills,")
    print("profit limits, cash/cap/cost accounting, overnight inventory, benchmarks,")
    print("missing-minute resets, input timezones/overlaps, and saved outputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, help="Contains AAPL/ and VOO/ subfolders")
    parser.add_argument("--asset-data", type=Path, help="Alternative: asset CSV or directory")
    parser.add_argument("--reference-data", type=Path, help="Alternative: reference CSV or directory")
    parser.add_argument("--asset", default="AXP")
    parser.add_argument("--reference", default="VOO")
    parser.add_argument("--window", type=int, default=20, help="Preceding consecutive minutes for OLS")
    parser.add_argument("--mode", choices=["percent", "zscore"], default="zscore")
    parser.add_argument("--threshold-pct", type=float, default=1, help="0.2 means 0.2%%, symmetric bands")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--z-window", type=int, help="Preceding gap observations; default = window")
    parser.add_argument("--initial-cash", type=float, default=10_000)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--max-shares", type=int, default=3000)
    parser.add_argument("--commission-bps", type=float, default=0)
    parser.add_argument("--fee", type=float, default=0, help="Fixed dollars per filled order")
    parser.add_argument("--slippage-bps", type=float, default=0)
    parser.add_argument("--execution", choices=["next_open", "next_close"], default="next_open")
    parser.add_argument("--input-timezone", default="UTC", help="Timezone of naive CSV timestamps")
    parser.add_argument("--market-timezone", default="America/New_York")
    parser.add_argument("--start", help="First evaluation date YYYY-MM-DD; earlier data can warm up fits")
    parser.add_argument("--end", help="Last evaluation date YYYY-MM-DD, inclusive")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", default = True,  action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run offline behavioral checks")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    try:
        cfg = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
        cfg.validate()
        if bool(args.asset_data) != bool(args.reference_data):
            raise ValueError("Supply both --asset-data and --reference-data")
        if args.data_root and args.asset_data:
            raise ValueError("Choose --data-root or the two explicit data paths")
        if args.data_root:
            asset_path, reference_path = args.data_root / cfg.asset, args.data_root / cfg.reference
        elif args.asset_data:
            asset_path, reference_path = args.asset_data, args.reference_data
        else:
            raise ValueError("Supply --data-root, or --asset-data and --reference-data")
        data = load_pair(asset_path, reference_path, cfg)
        results, trades, summary = simulate(make_signals(data, cfg), cfg)
        save_results(results, trades, summary, args.output, plot=not args.no_plot)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    p = summary["strategy"]
    print(f"{summary['start']} to {summary['end']} ({len(results):,} aligned evaluation bars)")
    print(f"Strategy equity ${p['final_equity']:,.2f}; P&L ${p['pnl']:,.2f}; return {p['return_pct']:.2f}%")
    print(f"Realized ${summary['realized_pnl']:,.2f}; unrealized ${summary['unrealized_pnl']:,.2f}; "
          f"open shares {summary['final_shares']}")
    for name, stats in summary["benchmarks"].items():
        print(f"{name}: {stats['return_pct']:.2f}%")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
