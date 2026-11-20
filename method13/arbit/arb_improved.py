#!/usr/bin/env python3
"""arb_improved.py - long-only AAPL/VOO minute-bar backtest: hold the asset, step aside while it is
rich versus the reference. Rebuilt from arb.py after an audit of 2025-05-09..2025-07-03 data.

Install:  python -m pip install numpy pandas matplotlib
Check:    python arb_improved.py --self-test
Run:      python arb_improved.py --data-root data --output results
Compare:  python arb_improved.py --data-root data --signal gap --exec-mode legacy --slippage-bps 0 \\
              --no-early-close-calendar          (reproduces arb.py with its CLI defaults, bar for bar)

WHAT THE DEFAULT STRATEGY DOES
  1. Hold the asset. The target is --base-exposure (100%) of equity, overnight and at the open too.
  2. At every completed 1-minute close t compute a residual z-score (below) from data up to t only.
     z >= +2 (asset rich versus the reference) starts a "rich" episode: the target becomes
     --rich-exposure (0%, i.e. cash). The episode ends, and the target returns to 100%, when z is
     back at or below --z-exit (0), after --max-hold (90) bars, or --eod-exit-min (2) minutes before
     the scheduled close. No new episode starts in the last --last-entry-min (30) minutes, or before
     --min-bars (10) residuals exist in the session. A side re-arms only after z has left its entry
     zone, so a time-stopped episode is not re-entered at once.
  3. On a change of target exposure the order for (target - held) shares is sent. It fills at the
     NEXT minute's open with adverse slippage, commission and fee, or expires if that minute is
     missing or in the next session. Buys are cash-limited (a partial fill is accepted, not chased).
     Sells are market orders that may realize losses (FIFO lots), so nothing is ever bag-held.
     No shorting, no leverage, integer shares, cash never negative.
  Everything above was chosen on TRAIN (2025-05-09..06-13) only and confirmed on VALIDATE
  (06-16..07-03). Buying "cheap" dips (z <= -2) lost money on TRAIN for every construction tried,
  so the cheap side is off by default (--cheap-exposure equals --base-exposure).

RESIDUAL SIGNAL (--signal resid, default)
  r_A, r_V  1-minute log returns inside the session; a session's first bar uses its own open->close,
            so overnight gaps never enter the signal. A missing minute does not restart anything.
  beta      OLS slope of r_A on r_V over the last --beta-window (1950, ~5 sessions) valid minutes,
            carried across sessions, shrunk toward --beta-prior (1.0) with --beta-prior-weight (390)
            pseudo-minutes. The residual at t uses the estimate made through t-1.
  e_t       r_A - beta * r_V: the part of the asset's move the reference does not explain.
  f(m)      expected e^2 at minute-of-session m: mean over the previous --tod-sessions (20) sessions,
            lightly smoothed; needs --tod-min-sessions (2) earlier sessions. Removes the intraday
            U-shape (09:30 residuals are ~10x the midday ones).
  L_t       recent volatility level: mean of e^2/f over the previous --vol-window (390) minutes,
            shrunk toward 1 with --vol-prior-weight (30) pseudo-minutes.
  S_t       this session's residuals summed over the last --k (30) bars (--kernel box), or an
            exponentially weighted sum with half-life --k minutes (--kernel exp).
  z_t       S_t / sqrt(L_t * sum of f over the same bars): calibrated, |z| >= 2 is ~3% of bars.
  arb.py's price-level gap (--signal gap) is kept for comparison only. Its 20-minute OLS slope of
  price on price is unstable (implied return beta outside [0, 3] on 13-25% of bars) and its z-score
  mixes 20 different fits; --gap-beta-min/--gap-beta-max mark bars outside a sane implied return
  beta (slope x reference price / asset price) as not ready.

EXECUTION MODES
  --exec-mode target (default)  the state machine above.
  --exec-mode legacy            arb.py's machinery: --chunk share buys on buy signals, and one
                                profitable --chunk of the oldest lot per sell signal at a net
                                breakeven limit; losing lots are never sold. For comparison only.

CALENDAR
  Only continuous-trading bars are used: 09:30 up to (not including) the scheduled close. The
  closing-auction bar (16:00, or 13:00 on early closes) is dropped, so the 15:59 (12:59) bar is the
  last tradable price; the auction itself is not modeled. NYSE 13:00 early closes for 2023-2026 are
  built in (--early-close adds dates, --no-early-close-calendar restores arb.py's session filter).
  Holidays are whatever days the data lacks. All "minutes before close" rules use the scheduled
  close of that day, so an episode can never be carried overnight by an early close.

FLAGS (default in brackets; every default lives in Config and the CLI is derived from it)
  data      --data-root DIR (with <asset>/ and <reference>/ CSV folders) or --asset-data and
            --reference-data; --asset [AAPL] --reference [VOO] --input-timezone [UTC]
            --market-timezone [America/New_York] --start/--end YYYY-MM-DD (evaluation dates; earlier
            data warms up the signal; later bars are dropped before computing)
  signal    --signal [resid] --beta-window [1950] --beta-prior [1.0] --beta-prior-weight [390]
            --kernel [box] --k [30] --min-bars [10] --vol-window [390] --vol-prior-weight [30]
            --tod-sessions [20] --tod-min-sessions [2]
  gap       --window [20] --mode [zscore] --threshold-pct [0.2] --z-window [=window]
            --gap-beta-min/--gap-beta-max [none]
  trading   --z-threshold [2] --z-exit [0] --max-hold [90 bars] --exec-mode [target]
            --base-exposure [1.0] --cheap-exposure [1.0 = off] --rich-exposure [0.0]
            --last-entry-min [30, 'none' disables] --eod-exit-min [2, 'none' disables]
            --chunk [25, legacy only] --max-shares [3000]
  costs     --initial-cash [10000] --slippage-bps [1 per fill] --commission-bps [0] --fee [0 $/fill]
            --execution [next_open | next_close]
  calendar  --early-close-calendar/--no-early-close-calendar [on] --early-close [extra dates, comma]
  output    --output [results] --plot/--no-plot [plot on] --split-dates [none: comma-separated
            dates that split one continuous run into reported sub-periods] --self-test
  Pure buy-and-hold with the same costs: --rich-exposure 1 (no side differs from the base).

OUTPUTS (one folder): results.csv (every bar: prices, signal columns, state, target, cash, shares,
  equity, benchmarks, base_only_equity), trades.csv, summary.json, equity.png. summary.json adds:
  attribution (exposure / timing / execution-and-cost P&L with the scoring harness's definitions;
  overlay_pnl = strategy minus the same rules with the overlay off), an ex-post exposure-matched
  buy-and-hold benchmark (reporting only: it holds the strategy's average share count), daily
  Sharpe, episode counts and --split-dates segments. Benchmarks start on the same executable bar
  with the same cash and costs. Price returns only: dividends, splits and taxes are not simulated.

RESULTS ON THE DEVELOPMENT DATA (AAPL vs VOO, $10,000; 1 bp per fill; 2 bp in brackets)
                                this script       arb.py (CLI defaults)  AAPL buy-and-hold
  TRAIN    2025-05-09..06-13      -0.00% [-0.30%]     -3.45% [-3.66%]           -1.64%
  VALIDATE 2025-06-16..07-03      +8.38% [+8.12%]     +7.89% [+7.68%]           +7.98%
  DEV, one continuous run        +8.99% [+8.43%]     +4.97% [+4.72%]           +6.93%
  VALIDATE slice of that run     +8.99% [+8.76%]     +8.72% [+8.69%]           +8.71%
  Max drawdown TRAIN -7.66% (arb.py -8.40%); daily Sharpe VALIDATE 8.91 (arb.py 8.99, hold 8.60).
  Other tickers vs VOO (6 days, 1 bp): beats arb.py on 4 of 5 (not GOOG), buy-and-hold on 1 (META).
  Read this honestly. On DEV the gain over arb.py is about half staying invested (no bag-holding,
  no 40-minute wait on day 1: +$196 [+$220]) and half the overlay (+$206 [+$151]). The overlay
  itself (strategy minus the same rules with the overlay off) made TRAIN +$164 [+$135], which is
  in-sample, and VALIDATE +$40 [+$16] over 12 episodes, -$11 at 3 bp, and -$250 in total on the
  other five tickers (negative on 4 of 5): statistically indistinguishable from zero out of
  sample. Expect roughly the asset's own return, with slightly lower exposure (~93-95%).

  HOLDOUT 2025-07-07..08-04 (20 sessions never used for any choice). One continuous 58-session run
  per script, sliced at 07-07, so both are measured over the same bars with the state they carried in.
                                this script       arb.py (CLI defaults)  AAPL buy-and-hold
  HOLDOUT                       -3.08% [-3.35%]     -3.87% [-4.05%]           -4.79%
  All 58 sessions               +5.63% [+4.79%]     +0.91% [+0.48%]           +1.81%
  The overlay made +$177 [+$149] over 15 holdout episodes (per-episode t 1.6; random-timing
  placebo p ~0.05). With VALIDATE near zero and the other tickers negative, it is still unproven.
  Over the 58 sessions the gain over arb.py is +$472 at 1 bp: +$89 from simply holding and +$382
  from the overlay. The same overlay driven by arb.py's gap signal loses (-3.22% [-11.41%]);
  it trades about 11x as often.

  LIMITS: 58 days of data (38 development + 20 holdout); next-open fills with 1-2 bp slippage and
  no market impact; no closing auction; positions are held overnight (gap risk); the residual
  signal uses ~26 sessions of history, so results for a date depend on where the data starts: a
  fresh start holds without signals for --tod-min-sessions sessions; the end-of-day exit needs a
  bar at close minus --eod-exit-min; the early-close list must be extended beyond 2026.

RECOVERY NOTE: This source was transcribed from 62 supplied photographs. Editor overlays and
OCR artifacts were removed and formatting normalized. Historical results and audit claims above
are the original author's comments; they were not independently rerun during recovery.
"""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
from dataclasses import asdict, dataclass, field, replace
from datetime import date
import io
import json
import math
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MINUTE_NS = 60_000_000_000
SESSION_OPEN = 570  # 09:30, minutes after midnight in the market time zone.
SESSION_MINUTES = 390  # Longest regular session, 09:30–16:00.
REGULAR_CLOSE = 960  # 16:00
EARLY_CLOSE = 780  # 13:00
# NYSE scheduled 13:00 ET early closes (published in advance; extend with --early-close).
EARLY_CLOSES = ("2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
                "2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24")


class BacktestError(RuntimeError):
    """A broken accounting or execution invariant. Raised explicitly, so python -O keeps the check."""


@dataclass(frozen=True)
class Config:
    asset: str = "MSFT"
    reference: str = "VOO"
    signal: str = "gap"
    # Residual signal.
    beta_window: int = 20
    beta_prior: float = 1.0
    beta_prior_weight: float = 20.0
    kernel: str = "box"
    k: int = 20
    min_bars: int = 10
    vol_window: int = 20
    vol_prior_weight: float = 20.0
    tod_sessions: int = 20
    tod_min_sessions: int = 2
    # arb.py's price-level gap signal (--signal gap), kept for comparison.
    window: int = 20
    mode: str = "zscore"
    threshold_pct: float = 0.2
    z_window: int | None = None
    gap_beta_min: float | None = None
    gap_beta_max: float | None = None
    # Entries, exits and sizing. Chosen on TRAIN: long by default, step aside while rich.
    z_threshold: float = 2.0
    z_exit: float = 0.0
    max_hold: int = 90
    exec_mode: str = "target"
    base_exposure: float = 1.0
    cheap_exposure: float = 1.0
    rich_exposure: float = 0.0
    last_entry_min: int | None = 30
    eod_exit_min: int | None = 2
    chunk: int = 25
    max_shares: int = 3000
    # Account and costs.
    initial_cash: float = 10_000.0
    commission_bps: float = 0.0
    fee: float = 0.0
    slippage_bps: float = 0.0
    execution: str = "next_open"
    # Calendar, dates and reporting.
    input_timezone: str = "UTC"
    market_timezone: str = "America/New_York"
    early_close_calendar: bool = True
    early_close: str = ""
    start: str | None = None
    end: str | None = None
    split_dates: str = ""

    def validate(self):
        if self.asset == self.reference:
            raise ValueError("Asset and reference must be different tickers.")
        if self.signal not in ("resid", "gap"):
            raise ValueError("signal must be resid or gap.")
        if self.exec_mode not in ("target", "legacy"):
            raise ValueError("exec-mode must be target or legacy.")
        if self.kernel not in ("box", "exp"):
            raise ValueError("kernel must be box or exp.")
        if self.mode not in ("percent", "zscore"):
            raise ValueError("mode must be percent or zscore.")
        if self.execution not in ("next_open", "next_close"):
            raise ValueError("execution must be next_open or next_close.")
        if self.signal == "gap" and self.exec_mode == "target" and self.mode != "zscore":
            raise ValueError("--signal gap with --exec-mode target needs --mode zscore (exits use z).")
        if self.window < 3 or (self.z_window is not None and self.z_window < 2):
            raise ValueError("window must be >= 3 and z-window >= 2.")
        for name in ("beta_window", "k", "min_bars", "vol_window", "tod_sessions", "tod_min_sessions",
                     "max_hold", "chunk", "max_shares"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.beta_window < 2:
            raise ValueError("beta-window must be >= 2.")
        for name in ("initial_cash", "threshold_pct", "z_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("beta_prior_weight", "vol_prior_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not math.isfinite(self.beta_prior) or not math.isfinite(self.z_exit):
            raise ValueError("beta-prior and z-exit must be finite.")
        if self.z_exit >= self.z_threshold:
            raise ValueError("z-exit must be below z-threshold (otherwise an episode exits at once).")
        for name in ("base_exposure", "cheap_exposure", "rich_exposure"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1] (long-only, no leverage).")
        for name in ("last_entry_min", "eod_exit_min"):
            value = getattr(self, name)
            if value is not None and not 0 <= value < SESSION_MINUTES:
                raise ValueError(f"{name} must be 'none' or in [0, {SESSION_MINUTES}) minutes.")
        for name in ("gap_beta_min", "gap_beta_max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or 'none'.")
        if (self.gap_beta_min is not None and self.gap_beta_max is not None
                and self.gap_beta_min >= self.gap_beta_max):
            raise ValueError("gap-beta-min must be below gap-beta-max.")
        if not math.isfinite(self.fee) or self.fee < 0:
            raise ValueError("fee must be finite and nonnegative.")
        for name in ("commission_bps", "slippage_bps"):
            if not 0 <= getattr(self, name) < 10_000:
                raise ValueError(f"{name} must be in [0, 10000).")
        ZoneInfo(self.input_timezone)
        ZoneInfo(self.market_timezone)
        for value in (self.start, self.end):
            # Dates are compared as YYYY-MM-DD strings, so 20250616 would silently match everything.
            if value is not None and date.fromisoformat(value).isoformat() != value:
                raise ValueError(f"dates must be YYYY-MM-DD, got {value!r}.")
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end.")
        date_list(self.early_close)
        date_list(self.split_dates)


def date_list(text: str) -> list[str]:
    """Comma-separated ISO dates, normalised to YYYY-MM-DD (raises ValueError on bad input)."""
    return [date.fromisoformat(part.strip()).isoformat() for part in (text or "").split(",") if part.strip()]


def early_close_days(cfg: Config) -> list[str]:
    days = set(date_list(cfg.early_close))
    if cfg.early_close_calendar:
        days |= set(EARLY_CLOSES)
    return sorted(days)


def close_minutes(day: np.ndarray, cfg: Config) -> np.ndarray:
    """Scheduled close (minutes after midnight, market time) for each market-local date label."""
    return np.where(np.isin(day, early_close_days(cfg)), EARLY_CLOSE, REGULAR_CLOSE).astype(int)


def market_days(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    return np.asarray(index.tz_convert(tz).strftime("%Y-%m-%d"))


# ----------------------------------------------------------------------------- data
def read_asset(path: Path, cfg: Config) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No CSV files found in {path}")
    pieces = []
    required = ["close", "open"] if cfg.execution == "next_open" else ["close"]
    for file in files:
        raw = pd.read_csv(file)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        ts_col = next((c for c in ("timestamp", "datetime", "date", "time") if c in raw), None)
        if ts_col is None or any(c not in raw for c in required):
            raise ValueError(f"{file}: need timestamp and {', '.join(required)} columns")
        if pd.api.types.is_numeric_dtype(raw[ts_col]):
            raise ValueError(f"{file}: timestamp must be an ISO date/time, not an epoch number")
        text = raw[ts_col].astype(str).str.strip()
        aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        if aware.any() and not aware.all():
            raise ValueError(f"{file}: do not mix aware and naive timestamps")
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
    minute = np.asarray(local.hour * 60 + local.minute)
    day = np.asarray(local.strftime("%Y-%m-%d"))
    # Continuous trading only; drop the scheduled closing-auction bar.
    keep = (np.asarray(local.dayofweek) < 5) & (minute >= SESSION_OPEN) & (minute < close_minutes(day, cfg))
    data = data[keep]
    if data.empty:
        raise ValueError("No aligned regular-session bars. Check timestamps/timezones.")
    return data


def session_layout(index: pd.DatetimeIndex, tz: str):
    """Market-local day labels, new-session flags, session numbers, first bar of each bar's session,
    and clock minutes."""
    local = index.tz_convert(tz)
    day = np.asarray(local.strftime("%Y-%m-%d"))
    new = np.r_[True, day[1:] != day[:-1]] if len(day) else np.zeros(0, dtype=bool)
    session = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    start_of = starts[session] if len(day) else np.zeros(0, dtype=int)
    clock = np.asarray(local.hour * 60 + local.minute, dtype=int)
    return day, new, session, start_of, clock


# -------------------------------------------------------------------------- signals
def make_gap_signals(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """arb.py's signal: OLS of asset PRICE on reference PRICE over the previous `window` minutes."""
    out = data.copy()
    names = ("slope", "intercept", "implied_beta", "estimate", "gap", "gap_pct", "gap_mean", "gap_std",
             "zscore")
    values = np.full((len(out), len(names)), np.nan)
    times = out.index.as_unit("ns").asi8  # Unit-safe: pandas 3 may store s/ms/us.
    starts = np.r_[0, np.flatnonzero(np.diff(times) != MINUTE_NS) + 1]
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
        slope = covariance / variance.where(variance > 1e-14)
        intercept = (y[0] + my) - slope * (x[0] + mx)
        implied = slope * (x[0] + mx) / (y[0] + my)  # Return beta implied by the price-level fit.
        estimate = slope * x + intercept
        estimate = estimate.where(estimate > 0)
        gap = pd.Series(y) - estimate
        history = gap.shift(1).rolling(zw)
        mean, std = history.mean(), history.std(ddof=1)
        z = (gap - mean) / std.where(std > 1e-12)
        values[begin:stop] = np.column_stack(
            [slope, intercept, implied, estimate, gap, 100 * gap / estimate, mean, std, z])
    for col, val in zip(names, values.T):
        out[col] = val
    metric = out["gap_pct"] if cfg.mode == "percent" else out["zscore"]
    threshold = cfg.threshold_pct if cfg.mode == "percent" else cfg.z_threshold
    ready = np.isfinite(metric) & np.isfinite(out["estimate"])
    if cfg.gap_beta_min is not None:
        ready &= out["implied_beta"] >= cfg.gap_beta_min
    if cfg.gap_beta_max is not None:
        ready &= out["implied_beta"] <= cfg.gap_beta_max
    out["ready"] = ready
    out["buy_signal"] = out["ready"] & (out["gap"] < 0) & (metric < -threshold)
    out["sell_signal"] = out["ready"] & (out["gap"] > 0) & (metric > threshold)
    return out


def intraday_log_returns(close: np.ndarray, open_: np.ndarray | None, new_session: np.ndarray):
    """Close-to-close log returns inside a session; a session's first bar uses its own open->close."""
    r = np.full(len(close), np.nan)
    if len(close) > 1:
        r[1:] = np.log(close[1:] / close[:-1])
    if open_ is not None:
        r[new_session] = np.log(close[new_session] / open_[new_session])
    else:
        r[new_session] = np.nan
    return r


def rolling_beta(ra: np.ndarray, rv: np.ndarray, window: int, prior: float, prior_weight: float):
    """Slope of ra on rv over the last `window` valid pairs up to and including each bar, shrunk
    toward `prior`: (n * b_ols + w0 * prior) / (n + w0). Returns (through_t, used_at_t), where
    used_at_t is the estimate through t-1 (so the residual at t is out of sample)."""
    n = len(ra)
    ok = np.isfinite(ra) & np.isfinite(rv)
    idx = np.flatnonzero(ok)
    through = np.full(n, np.nan)
    if len(idx):
        x, y = rv[idx], ra[idx]
        cx, cy = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(y)]
        cxx, cxy = np.r_[0.0, np.cumsum(x * x)], np.r_[0.0, np.cumsum(x * y)]
        j = np.arange(1, len(idx) + 1)
        lo = np.maximum(0, j - window)
        m = (j - lo).astype(float)
        sx, sy = cx[j] - cx[lo], cy[j] - cy[lo]
        sxx = cxx[j] - cxx[lo] - sx * sx / m
        sxy = cxy[j] - cxy[lo] - sx * sy / m
        good = (m >= 2) & (sxx > 1e-12 * m)
        ols = np.where(good, sxy / np.where(good, sxx, 1.0), prior)
        weight = np.where(good, m, 0.0)
        through[idx] = (weight * ols + prior_weight * prior) / np.maximum(weight + prior_weight, 1e-300)
    through = pd.Series(through).ffill().fillna(prior).to_numpy()
    used = np.r_[prior, through[:-1]] if n else through
    return through, used


def tod_variance(e2: np.ndarray, session: np.ndarray, mod: np.ndarray, lookback: int,
                 min_sessions: int, max_halfwidth: int = 5) -> np.ndarray:
    """Expected squared residual at each bar's minute-of-session, from PREVIOUS sessions only.

    Mean of e^2 at the same minute over the last `lookback` sessions, smoothed over
    +/- min(max_halfwidth, m // 3) neighbouring minutes (no smoothing for the first minutes, whose
    volatility changes fastest). NaN until `min_sessions` earlier sessions exist."""
    n = len(e2)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    n_s = int(session.max()) + 1
    width = SESSION_MINUTES
    inside = (mod >= 0) & (mod < width)
    ok = np.isfinite(e2) & inside
    sums = np.zeros((n_s, width))
    counts = np.zeros((n_s, width))
    np.add.at(sums, (session[ok], mod[ok]), e2[ok])
    np.add.at(counts, (session[ok], mod[ok]), 1.0)
    csum = np.vstack([np.zeros((1, width)), np.cumsum(sums, axis=0)])
    ccnt = np.vstack([np.zeros((1, width)), np.cumsum(counts, axis=0)])
    s = np.arange(n_s)
    lo = np.maximum(0, s - lookback)
    past_sum, past_cnt = csum[s] - csum[lo], ccnt[s] - ccnt[lo]
    m = np.arange(width)
    half = np.minimum(max_halfwidth, m // 3)
    a, b = np.maximum(0, m - half), np.minimum(width, m + half + 1)
    qs = np.hstack([np.zeros((n_s, 1)), np.cumsum(past_sum, axis=1)])
    qc = np.hstack([np.zeros((n_s, 1)), np.cumsum(past_cnt, axis=1)])
    num, den = qs[:, b] - qs[:, a], qc[:, b] - qc[:, a]
    profile = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    profile[(s - lo) < min_sessions] = np.nan
    profile = np.where(profile > 0, profile, np.nan)
    out[inside] = profile[session[inside], mod[inside]]
    return out


def make_resid_signals(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = data.copy()
    n = len(out)
    _, new, session, start_of, clock = session_layout(out.index, cfg.market_timezone)
    mod = clock - SESSION_OPEN
    has_open = "asset_open" in out and "reference_open" in out
    ra = intraday_log_returns(out["asset_close"].to_numpy(float),
                             out["asset_open"].to_numpy(float) if has_open else None, new)
    rv = intraday_log_returns(out["reference_close"].to_numpy(float),
                             out["reference_open"].to_numpy(float) if has_open else None, new)
    beta_through, beta = rolling_beta(ra, rv, cfg.beta_window, cfg.beta_prior, cfg.beta_prior_weight)
    e = ra - beta * rv
    tod = tod_variance(e * e, session, mod, cfg.tod_sessions, cfg.tod_min_sessions)
    # Volatility level relative to the time-of-day profile, from residuals strictly before t.
    u = e * e / tod
    valid_u = np.flatnonzero(np.isfinite(u))
    cu = np.r_[0.0, np.cumsum(u[valid_u])]
    before = np.searchsorted(valid_u, np.arange(n), side="left")
    lo = np.maximum(0, before - cfg.vol_window)
    level = (cu[before] - cu[lo] + cfg.vol_prior_weight) / (before - lo + cfg.vol_prior_weight)
    use = np.isfinite(e) & np.isfinite(tod)
    ev, fv, cnt = np.where(use, e, 0.0), np.where(use, tod, 0.0), use.astype(float)
    if cfg.kernel == "box":
        ce, cf, cn = (np.r_[0.0, np.cumsum(v)] for v in (ev, fv, cnt))
        t = np.arange(n)
        first = np.maximum(t - cfg.k + 1, start_of)
        cum, var, count = ce[t + 1] - ce[first], cf[t + 1] - cf[first], cn[t + 1] - cn[first]
    else:
        lam = 0.5 ** (1.0 / cfg.k)
        cum, var, count = np.zeros(n), np.zeros(n), np.zeros(n)
        c_e = c_f = c_n = 0.0
        for i in range(n):
            if new[i]:
                c_e = c_f = c_n = 0.0
            c_e = ev[i] + lam * c_e
            c_f = fv[i] + lam * lam * c_f
            c_n = cnt[i] + c_n
            cum[i], var[i], count[i] = c_e, c_f, c_n
    scale = np.sqrt(level * var)
    ready = (count >= cfg.min_bars) & (var > 0) & np.isfinite(scale) & (scale > 0)
    z = np.where(ready, cum / np.where(ready, scale, 1.0), np.nan)
    out["ret_asset"], out["ret_reference"] = ra, rv
    out["beta"], out["beta_through_t"] = beta, beta_through
    out["resid"], out["tod_var"], out["vol_level"] = e, tod, level
    out["cum_resid"], out["cum_var"], out["n_resid"] = cum, level * var, count
    out["zscore"] = z
    out["ready"] = ready
    out["buy_signal"] = ready & (z <= -cfg.z_threshold)
    out["sell_signal"] = ready & (z >= cfg.z_threshold)
    return out


def make_signals(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return make_resid_signals(data, cfg) if cfg.signal == "resid" else make_gap_signals(data, cfg)


# ----------------------------------------------------------------------- accounting
@dataclass
class Lot:
    lot_id: int
    qty: int
    unit_cost: float


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

    def unit_buy_cost(self, raw_price: float):
        return raw_price * (1 + self.cfg.slippage_bps / 10_000) * (1 + self.cfg.commission_bps / 10_000)

    def available(self, raw_price: float, requested: int):
        affordable = max(0, int(math.floor((self.cash - self.cfg.fee) / self.unit_buy_cost(raw_price))))
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
        """Legacy: one chunk of the oldest lot that a net-breakeven limit can sell at a profit."""
        multiplier = (1 - self.cfg.slippage_bps / 10_000) * (1 - self.cfg.commission_bps / 10_000)
        for lot in self.lots:
            qty = min(self.cfg.chunk, lot.qty)
            limit = (qty * lot.unit_cost + self.cfg.fee) / (qty * multiplier)
            limit *= 1 + 1e-12  # Strictly profitable, rather than exactly breakeven.
            if close >= limit:
                return dict(side="sell", lot_id=lot.lot_id, qty=qty, limit=limit)
        return None

    def sell(self, order: dict, raw_price: float):
        """Legacy limit sell of one lot chunk; None if the quote is below the limit."""
        if raw_price < order["limit"]:
            return None
        lot = next(lot for lot in self.lots if lot.lot_id == order["lot_id"])
        qty = order["qty"]
        if not 0 < qty <= lot.qty:
            raise BacktestError("Sell quantity exceeds the lot")
        price = raw_price * (1 - self.cfg.slippage_bps / 10_000)
        fee = qty * price * self.cfg.commission_bps / 10_000 + self.cfg.fee
        proceeds = qty * price - fee
        pnl = proceeds - qty * lot.unit_cost
        if pnl <= 0:
            raise BacktestError("Sell limit failed to preserve positive net profit")
        self.cash += proceeds
        self.realized += pnl
        lot.qty -= qty
        if lot.qty == 0:
            self.lots.remove(lot)
        return dict(side="sell", lot_id=lot.lot_id, qty=qty, price=price, fee=fee,
                    realized_pnl=pnl, cash_after=self.cash, shares_after=self.shares)

    def sell_market(self, raw_price: float, requested: int):
        """Sell up to `requested` shares at the quote (first-in, first-out lots), gains or losses."""
        qty = min(int(requested), self.shares)
        if qty <= 0:
            return None
        price = raw_price * (1 - self.cfg.slippage_bps / 10_000)
        fee = qty * price * self.cfg.commission_bps / 10_000 + self.cfg.fee
        proceeds = qty * price - fee
        left, basis, first_id = qty, 0.0, self.lots[0].lot_id
        while left:
            lot = self.lots[0]
            take = min(left, lot.qty)
            basis += take * lot.unit_cost
            lot.qty -= take
            left -= take
            if lot.qty == 0:
                self.lots.pop(0)
        pnl = proceeds - basis
        self.cash += proceeds
        self.realized += pnl
        return dict(side="sell", lot_id=first_id, qty=qty, price=price, fee=fee,
                    realized_pnl=pnl, cash_after=self.cash, shares_after=self.shares)


def check_book(book: Account, cfg: Config, price: float):
    """Per-bar invariants: no negative cash, share cap, and cash + inventory reconcile to P&L."""
    equity = book.cash + book.shares * price
    unrealized = book.shares * price - book.basis
    if book.cash < -1e-7 or book.shares > cfg.max_shares or book.shares < 0:
        raise BacktestError("Cash or share constraint violated")
    if not math.isclose(equity - cfg.initial_cash, book.realized + unrealized, abs_tol=1e-6):
        raise BacktestError("Cash, realized P&L and inventory basis do not reconcile")
    return equity, unrealized


def start_benchmarks(cfg, asset_quote, reference_quote):
    benchmarks = [Account(cfg, None), Account(cfg, None), Account(cfg, cfg.max_shares)]
    for b, quote in zip(benchmarks, (asset_quote, reference_quote, asset_quote)):
        b.buy(quote, 2**60)
    return benchmarks


def evaluation_window(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    local_dates = market_days(features.index, cfg.market_timezone)
    mask = np.ones(len(features), dtype=bool)
    if cfg.start:
        mask &= local_dates >= cfg.start
    if cfg.end:
        mask &= local_dates <= cfg.end
    data = features.loc[mask].copy()
    if len(data) < 2:
        raise ValueError("Fewer than two aligned bars within the requested evaluation dates.")
    return data


# ----------------------------------------------------------------------- simulation
BALANCE_COLUMNS = ["cash", "shares", "cost_basis", "realized_pnl", "unrealized_pnl", "equity",
                   "buy_hold_asset", "buy_hold_reference", "buy_hold_asset_capped"]


@dataclass
class Run:
    """Raw output of one simulation pass (finished into results/trades/summary by finish())."""
    data: pd.DataFrame
    columns: dict
    trades: list
    counts: Counter
    book: Account
    quote1: float
    notes: list
    episodes: Counter = field(default_factory=Counter)


def fill_pending(pending, book, quote, contiguous_now, ts, trades, legacy):
    """Execute an order decided at the previous close on this bar's quote, or let it expire."""
    if not contiguous_now:
        return "expired_missing_minute_or_session", None
    if pending["side"] == "buy":
        event = book.buy(quote, pending["qty"])
        status = "filled_buy" if event else "blocked_cash_at_fill"
    elif legacy:
        event = book.sell(pending, quote)
        status = "filled_sell" if event else "expired_profit_limit"
    else:
        event = book.sell_market(quote, pending["qty"])
        status = "filled_sell" if event else "nothing_to_sell"
    if event is not None:
        event.update(signal_timestamp=pending["signal_timestamp"].isoformat(),
                     fill_timestamp=ts.isoformat(), limit=pending.get("limit", None))
        trades.append(event)
    return status, event


def run_legacy(features: pd.DataFrame, cfg: Config) -> Run:
    """arb.py's machinery: chunk buys on buy signals, one profitable lot chunk per sell signal."""
    data = evaluation_window(features, cfg)
    next_ok = np.r_[np.diff(data.index.as_unit("ns").asi8) == MINUTE_NS, False]
    first = np.flatnonzero(data["ready"].to_numpy() & next_ok)
    if len(first) == 0:
        raise ValueError("No executable evaluation period after indicator warm-up. "
                         "Need consecutive one-minute bars after the signal warm-up.")
    data = data.iloc[first[0]:].copy()
    n = len(data)
    book = Account(cfg, cfg.max_shares)
    benchmarks = None
    pending = None
    trades, counts = [], Counter()
    balances = np.zeros((n, len(BALANCE_COLUMNS)))
    decisions, order_states = [], []
    limits = np.full(n, np.nan)
    close = data["asset_close"].to_numpy(float)
    reference_close = data["reference_close"].to_numpy(float)
    fill_col = "open" if cfg.execution == "next_open" else "close"
    asset_quotes = data[f"asset_{fill_col}"].to_numpy(float)
    reference_quotes = data[f"reference_{fill_col}"].to_numpy(float)
    buy, sell = data["buy_signal"].to_numpy(bool), data["sell_signal"].to_numpy(bool)
    ready = data["ready"].to_numpy(bool)
    times = data.index
    contiguous = np.r_[False, np.diff(times.as_unit("ns").asi8) == MINUTE_NS]
    for i, ts in enumerate(times):
        status = "none"
        if i == 1:
            if not contiguous[1]:
                raise BacktestError("first executable bar is not the next minute")
            benchmarks = start_benchmarks(cfg, asset_quotes[i], reference_quotes[i])
        if pending is not None:
            status, _ = fill_pending(pending, book, asset_quotes[i], contiguous[i], ts, trades, legacy=True)
            counts[status] += 1
            pending = None
        decision = "no_signal" if ready[i] else "warmup"
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
        equity, unrealized = check_book(book, cfg, close[i])
        bh = ([cfg.initial_cash] * 3 if benchmarks is None else
              [b.cash + b.shares * p for b, p in zip(benchmarks, (close[i], reference_close[i], close[i]))])
        balances[i] = [book.cash, book.shares, book.basis, book.realized, unrealized, equity, *bh]
        decisions.append(decision)
        order_states.append(status)
    if pending is not None:
        counts["cancelled_at_end"] += 1
        decisions[-1] += "_cancelled_at_end"
    columns = dict(zip(BALANCE_COLUMNS, balances.T))
    columns.update(decision=decisions, order_status=order_states, submitted_sell_limit=limits)
    notes = ["Legacy execution (arb.py): profit-only exits can leave losing lots open; equity includes them."]
    return Run(data, columns, trades, counts, book, float(asset_quotes[1]), notes)


def active_sides(cfg: Config) -> dict:
    """A side only exists if its exposure differs from the base: an identical target is a no-op,
    and entering it would only re-target the position from current equity (spurious odd lots)."""
    return {1: cfg.cheap_exposure != cfg.base_exposure, -1: cfg.rich_exposure != cfg.base_exposure}


def target_shares(exposure: float, equity: float, price: float, cfg: Config) -> int:
    if exposure <= 0:
        return 0
    unit = price * (1 + cfg.slippage_bps / 10_000) * (1 + cfg.commission_bps / 10_000)
    shares = int(math.floor(max(0.0, exposure * equity - cfg.fee) / unit))
    return min(shares, cfg.max_shares)


def run_target(features: pd.DataFrame, cfg: Config, overlay: bool = True) -> Run:
    """Signal state machine -> target exposure -> next-open market orders for the difference.
    overlay=False runs the same rules with every side disabled (the base position only)."""
    data = evaluation_window(features, cfg)
    t_ns = data.index.as_unit("ns").asi8
    next_ok = np.r_[np.diff(t_ns) == MINUTE_NS, False]
    first = np.flatnonzero(next_ok)
    if len(first) == 0:
        raise ValueError("No executable evaluation period: need two consecutive one-minute bars.")
    data = data.iloc[first[0]:].copy()
    times = data.index
    n = len(data)
    contiguous = np.r_[False, np.diff(times.as_unit("ns").asi8) == MINUTE_NS]
    day, _, _, _, clock = session_layout(times, cfg.market_timezone)
    close_min = close_minutes(day, cfg)
    close = data["asset_close"].to_numpy(float)
    reference_close = data["reference_close"].to_numpy(float)
    fill_col = "open" if cfg.execution == "next_open" else "close"
    asset_quotes = data[f"asset_{fill_col}"].to_numpy(float)
    reference_quotes = data[f"reference_{fill_col}"].to_numpy(float)
    z = data["zscore"].to_numpy(float)
    ready = data["ready"].to_numpy(bool)
    sides = active_sides(cfg) if overlay else {1: False, -1: False}
    if cfg.signal == "gap":  # The gap signal's entries also require the gap's sign.
        cheap_now = data["buy_signal"].to_numpy(bool) & sides[1]
        rich_now = data["sell_signal"].to_numpy(bool) & sides[-1]
    else:
        cheap_now = ready & (z <= -cfg.z_threshold) & sides[1]
        rich_now = ready & (z >= cfg.z_threshold) & sides[-1]
    exposure = {0: cfg.base_exposure, 1: cfg.cheap_exposure, -1: cfg.rich_exposure}
    book = Account(cfg, cfg.max_shares)
    benchmarks = None
    pending = None
    trades, counts, episodes = [], Counter(), Counter()
    balances = np.zeros((n, len(BALANCE_COLUMNS)))
    decisions, order_states = [], []
    states = np.zeros(n, dtype=int)
    desired_log = np.zeros(n, dtype=int)
    state, held, desired = 0, 0, None
    armed = {1: True, -1: True}
    for i in range(n):
        ts = times[i]
        status = "none"
        if i == 1:
            if not contiguous[1]:
                raise BacktestError("first executable bar is not the next minute")
            benchmarks = start_benchmarks(cfg, asset_quotes[i], reference_quotes[i])
        if pending is not None:
            status, event = fill_pending(pending, book, asset_quotes[i], contiguous[i], ts, trades,
                                         legacy=False)
            if pending["side"] == "buy" and contiguous[i] and (event is None or event["qty"] < pending["qty"]):
                desired = book.shares  # Accept a cash-limited fill; do not chase it.
            counts[status] += 1
            pending = None

        # ---- Signal state machine at close i.
        previous = state
        decision = "hold" if ready[i] else "warmup"
        eod = cfg.eod_exit_min is not None and clock[i] >= close_min[i] - cfg.eod_exit_min
        entry_window = cfg.last_entry_min is None or clock[i] < close_min[i] - cfg.last_entry_min
        if not cheap_now[i]:
            armed[1] = True
        if not rich_now[i]:
            armed[-1] = True
        if state != 0:
            held += 1
            reverted = ready[i] and ((state == 1 and z[i] >= -cfg.z_exit) or
                                     (state == -1 and z[i] <= cfg.z_exit))
            if reverted or held >= cfg.max_hold or eod:
                decision = "exit_revert" if reverted else "exit_time" if held >= cfg.max_hold else "exit_eod"
                if not reverted:
                    armed[state] = False
                state, held = 0, 0
        if state == 0 and ready[i] and not eod and entry_window:
            if cheap_now[i] and armed[1]:
                state, held, decision = 1, 0, "enter_cheap"
                armed[1] = False
                episodes["cheap"] += 1
            elif rich_now[i] and armed[-1]:
                state, held, decision = -1, 0, "enter_rich"
                armed[-1] = False
                episodes["rich"] += 1
        if desired is None or exposure[state] != exposure[previous]:
            desired = target_shares(exposure[state], book.cash + book.shares * close[i], close[i], cfg)
        if desired != book.shares:
            side = "buy" if desired > book.shares else "sell"
            pending = dict(side=side, qty=abs(desired - book.shares), signal_timestamp=ts)
            decision += "_submit_" + side
        counts[decision] += 1
        equity, unrealized = check_book(book, cfg, close[i])
        bh = ([cfg.initial_cash] * 3 if benchmarks is None else
              [b.cash + b.shares * p for b, p in zip(benchmarks, (close[i], reference_close[i], close[i]))])
        balances[i] = [book.cash, book.shares, book.basis, book.realized, unrealized, equity, *bh]
        decisions.append(decision)
        order_states.append(status)
        states[i] = state
        desired_log[i] = desired
    if pending is not None:
        counts["cancelled_at_end"] += 1
        decisions[-1] += "_cancelled_at_end"
    columns = dict(zip(BALANCE_COLUMNS, balances.T))
    columns.update(signal_state=states, target_shares=desired_log, decision=decisions,
                   order_status=order_states)
    notes = ["Target execution: market orders for (target - held) shares at the next open; exits may "
             "realize losses, so no lot is bag-held."]
    return Run(data, columns, trades, counts, book, float(asset_quotes[1]), notes, episodes)


def pnl_attribution(shares, close, equity, start_value: float) -> dict:
    """Scoring-harness definitions: holding P&L = sum(shares[t] * dP[t+1]); exposure = mean(shares) x
    total price change; timing = holding - exposure (> 0 means WHEN it held paid); the rest of the
    P&L (fills away from the close, slippage, commissions, fees) is execution_and_cost."""
    sh, px, eq = (np.asarray(v, float) for v in (shares, close, equity))
    total = float(eq[-1] - start_value)
    if len(sh) < 2:
        return dict(total_pnl=total, holding_pnl=0.0, exposure_pnl=0.0, timing_pnl=0.0,
                    execution_and_cost_pnl=total)
    held, dpx = sh[:-1], np.diff(px)
    holding = float(held @ dpx)
    exposure = float(held.mean() * dpx.sum())
    return dict(total_pnl=total, holding_pnl=holding, exposure_pnl=exposure, timing_pnl=holding - exposure,
                execution_and_cost_pnl=total - holding)


def performance(values, times: pd.DatetimeIndex, cfg: Config, start_value: float | None = None) -> dict:
    values = np.asarray(values, float)
    init = cfg.initial_cash if start_value is None else float(start_value)
    days = (times[-1] - times[0]).total_seconds() / 86400
    day = market_days(times, cfg.market_timezone)
    daily = values[np.r_[day[1:] != day[:-1], True]]
    rets = np.diff(np.r_[init, daily]) / np.r_[init, daily[:-1]]
    sd = float(rets.std(ddof=1)) if len(rets) > 2 else 0.0
    path = np.r_[init, values]
    return dict(final_equity=float(values[-1]), pnl=float(values[-1] - init),
                return_pct=float(100 * (values[-1] / init - 1)),
                max_drawdown_pct=float(100 * np.min(path / np.maximum.accumulate(path) - 1)),
                daily_sharpe=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else None,
                cagr_pct=float(100 * ((values[-1] / init) ** (365.25 / days) - 1)) if days >= 365.25 else None)


def fixed_position_curve(cfg: Config, quote1: float, close: np.ndarray, shares: int):
    """Equity of buying `shares` (cash-limited) at bar 1's quote with the usual costs, then holding."""
    account = Account(cfg, None)
    if shares > 0:
        account.buy(quote1, shares)
    curve = account.cash + account.shares * np.asarray(close, float)
    curve[0] = cfg.initial_cash
    return curve, account.shares


def segment_report(data: pd.DataFrame, trades: pd.DataFrame, cfg: Config, curves: dict) -> list:
    """Sub-period results of ONE continuous run, split at --split-dates (each date starts a segment)."""
    splits = date_list(cfg.split_dates)
    if not splits:
        return []
    day = market_days(data.index, cfg.market_timezone)
    bounds = [None, *sorted(set(splits)), None]
    equity = data["equity"].to_numpy(float)
    fills = pd.to_datetime(trades["fill_timestamp"], utc=True) if len(trades) else pd.Series([], dtype=object)
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = np.ones(len(day), dtype=bool)
        if lo:
            mask &= day >= lo
        if hi:
            mask &= day < hi
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        a, b = int(idx[0]), int(idx[-1])
        p = max(a - 1, 0)  # Previous bar: the segment starts from its equity (bar 0 = initial cash).
        row = dict(first_date=str(day[a]), last_date=str(day[b]), bars=int(len(idx)))
        stats = performance(equity[a:b + 1], data.index[a:b + 1], cfg, start_value=equity[p])
        row.update(return_pct=stats["return_pct"], max_drawdown_pct=stats["max_drawdown_pct"],
                   daily_sharpe=stats["daily_sharpe"])
        for name, curve in curves.items():
            row[f"{name}_return_pct"] = float(100 * (curve[b] / curve[p] - 1))
        if len(trades):
            inside = (fills > data.index[p]) & (fills <= data.index[b])
            row["fills"] = int(inside.sum())
        else:
            row["fills"] = 0
        row.update(pnl_attribution(data["shares"].to_numpy()[p:b + 1], data["asset_close"].to_numpy()[p:b + 1],
                                   equity[p:b + 1], equity[p]))
        rows.append(row)
    return rows


def finish(run: Run, cfg: Config):
    data, columns = run.data, run.columns
    for name, values in columns.items():
        data[name] = values
    times = data.index
    trade_columns = ["signal_timestamp", "fill_timestamp", "side", "lot_id", "qty", "price",
                     "fee", "realized_pnl", "cash_after", "shares_after", "limit"]
    trades = pd.DataFrame(run.trades, columns=trade_columns)
    close = data["asset_close"].to_numpy(float)
    shares = data["shares"].to_numpy(float)
    equity = data["equity"].to_numpy(float)
    avg_shares = float(shares.mean())
    # Ex-post, reporting only (not in results.csv): passive position at the strategy's average size.
    matched, matched_shares = fixed_position_curve(cfg, run.quote1, close, int(round(avg_shares)))
    curves = {"buy_hold_asset": data["buy_hold_asset"].to_numpy(float),
              "buy_hold_reference": data["buy_hold_reference"].to_numpy(float),
              "buy_hold_asset_capped": data["buy_hold_asset_capped"].to_numpy(float),
              "buy_hold_asset_exposure_matched": matched}
    if "base_only_equity" in data:
        curves["base_only"] = data["base_only_equity"].to_numpy(float)
    attribution = pnl_attribution(shares, close, equity, cfg.initial_cash)
    attribution.update(avg_shares=avg_shares,
                       avg_exposure_pct=float(np.mean(shares * close / equity) * 100),
                       time_in_market_pct=float(100 * np.mean(shares > 0)),
                       vs_buy_hold_asset_pnl=float(equity[-1] - curves["buy_hold_asset"][-1]),
                       exposure_matched_shares=int(matched_shares))
    if "base_only" in curves:
        attribution["base_only_pnl"] = float(curves["base_only"][-1] - cfg.initial_cash)
        attribution["overlay_pnl"] = float(equity[-1] - curves["base_only"][-1])
    if "signal_state" in data:
        state = data["signal_state"].to_numpy()
        attribution.update(rich_episodes=int(run.episodes["rich"]), cheap_episodes=int(run.episodes["cheap"]),
                           bars_in_episode_pct=float(100 * np.mean(state != 0)))
    notes = run.notes + [
        "Signals at close t use data up to t; orders fill at the next minute's quote or expire.",
        "Continuous-trading bars only (09:30 to the scheduled close; auction bars dropped).",
        "Price returns only: dividends, corporate actions and taxes are not modeled.",
        "Benchmarks start on the same next-minute quote with the same costs; the exposure-matched "
        "benchmark holds the strategy's average share count and is ex post (reporting only).",
        "overlay_pnl = strategy minus the same rules with the overlay off (base position only)."]
    summary = dict(parameters=asdict(cfg), start=times[0].isoformat(), end=times[-1].isoformat(),
                   evaluated_bars=len(data), strategy=performance(equity, times, cfg),
                   benchmarks={name: performance(curve, times, cfg) for name, curve in curves.items()},
                   attribution=attribution,
                   final_cash=run.book.cash, final_shares=run.book.shares,
                   realized_pnl=run.book.realized, unrealized_pnl=float(data["unrealized_pnl"].iloc[-1]),
                   buy_fills=int((trades["side"] == "buy").sum()),
                   sell_fills=int((trades["side"] == "sell").sum()), order_counts=dict(run.counts),
                   avg_shares=avg_shares, time_in_market_pct=attribution["time_in_market_pct"],
                   early_closes_in_data=sorted(set(early_close_days(cfg)) & set(market_days(times, cfg.market_timezone))),
                   segments=segment_report(data, trades, cfg, curves), notes=notes)
    yearly = []
    previous = {name: cfg.initial_cash for name in ["equity", *curves]}
    years = data.index.tz_convert(cfg.market_timezone).year
    for year in np.unique(years):
        rows = np.flatnonzero(years == year)
        row = {"year": int(year), "first_timestamp": times[rows[0]].isoformat(),
               "last_timestamp": times[rows[-1]].isoformat()}
        for name, curve in [("equity", equity), *curves.items()]:
            final = float(curve[rows[-1]])
            row[name + "_return_pct"] = 100 * (final / previous[name] - 1)
            previous[name] = final
        yearly.append(row)
    summary["yearly_returns"] = yearly
    return data, trades, summary


def simulate(features: pd.DataFrame, cfg: Config):
    if cfg.exec_mode == "legacy":
        return finish(run_legacy(features, cfg), cfg)
    run = run_target(features, cfg, overlay=True)
    if any(active_sides(cfg).values()):
        base = run_target(features, cfg, overlay=False)
        run.columns["base_only_equity"] = base.columns["equity"]
        run.columns["base_only_shares"] = base.columns["shares"]
    return finish(run, cfg)


def save_results(data, trades, summary, folder: Path, plot: bool = True):
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
        lines = [("equity", "Strategy"), ("buy_hold_asset", cfg["asset"] + " buy & hold"),
                 ("buy_hold_reference", cfg["reference"] + " buy & hold")]
        if "base_only_equity" in shown:
            lines.append(("base_only_equity", "Same rules, overlay off"))
        for name, label in lines:
            ax.plot(shown.index, shown[name], label=label, linewidth=1.2,
                    linestyle="--" if name == "base_only_equity" else "-")
        ax.set(title=f"{cfg['asset']} / {cfg['reference']}: marked portfolio equity",
               ylabel="Equity ($)", xlabel="Time (UTC)")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(folder / "equity.png", dpi=160)
        plt.close(fig)


# ------------------------------------------------------------------------ self-test
def _synthetic_sessions(days, seed: int = 5, beta: float = 1.3, early=()):
    """Regular-session minute bars with AAPL = beta * VOO + idiosyncratic noise. Days listed in
    `early` stop at 12:59 ET, like a 13:00 early close."""
    rng = np.random.default_rng(seed)
    pieces, mods = [], []
    for day in days:
        count = 210 if day in early else 390
        pieces.append(pd.date_range(pd.Timestamp(day + " 09:30", tz="America/New_York"), periods=count,
                                    freq="min").tz_convert("UTC"))
        mods.append(np.arange(count))
    idx = pieces[0].append(pieces[1:])
    m = np.concatenate(mods)
    n = len(idx)
    vol = 1 + 3 * np.exp(-m / 20)  # U-shape-like: busier opening minutes.
    rv = rng.normal(0, 3e-4, n) * vol
    ra = beta * rv + rng.normal(0, 4e-4, n) * vol
    overnight = rng.normal(0, 3e-3, (n, 2))
    prices = np.zeros((n, 4))  # asset_open, asset_close, reference_open, reference_close
    last_a, last_v = 200.0, 500.0
    for i in range(n):
        # Inside a session the open is the previous close; a session's first open gaps overnight.
        open_a = last_a * (math.exp(overnight[i, 0]) if m[i] == 0 else 1.0)
        open_v = last_v * (math.exp(overnight[i, 1]) if m[i] == 0 else 1.0)
        last_a, last_v = open_a * math.exp(ra[i]), open_v * math.exp(rv[i])
        prices[i] = open_a, last_a, open_v, last_v
    return pd.DataFrame(prices, index=idx,
                        columns=["asset_open", "asset_close", "reference_open", "reference_close"])


def _write_csvs(frame: pd.DataFrame, root: Path, tickers=("AAPL", "VOO")):
    """Write a pair frame as naive-UTC daily CSVs under root/<ticker>/."""
    for ticker, prefix in zip(tickers, ("asset", "reference")):
        folder = root / ticker
        folder.mkdir(parents=True, exist_ok=True)
        raw = frame[[prefix + "_open", prefix + "_close"]].copy()
        raw.columns = ["open", "close"]
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
        for day, part in raw.groupby(raw.index.date):
            part.to_csv(folder / f"{ticker}_{day}.csv", index_label="timestamp")


def _check_gap_and_legacy():
    """arb.py's gap signal and lot machinery, as in the original self-test, plus the beta guard."""
    cfg = Config(signal="gap", exec_mode="legacy", window=5, z_window=4, mode="percent", chunk=5,
                 max_shares=30, slippage_bps=0)
    cfg.validate()
    idx = pd.date_range("2025-01-02 14:30", periods=90, freq="min", tz="UTC")
    x = 500 + np.arange(90) * 0.02 + np.sin(np.arange(90)) * 0.03
    y = 2 * x + 10
    y[15] -= 5
    frame = pd.DataFrame(dict(asset_close=y, asset_open=y, reference_close=x, reference_open=x), index=idx)
    signals = make_signals(frame, cfg)
    assert abs(signals.iloc[15]["slope"] - 2) < 1e-8
    assert abs(signals.iloc[15]["intercept"] - 10) < 1e-5
    assert abs(signals.iloc[15]["gap"] + 5) < 1e-8
    assert signals.iloc[15]["buy_signal"]
    rng = np.random.default_rng(17)
    noisy = frame.copy()
    noisy["asset_close"] += rng.normal(0, 0.6, len(frame))
    for mode in ("percent", "zscore"):
        c = replace(cfg, mode=mode)
        full = make_signals(noisy, c)
        part = make_signals(noisy.iloc[:60], c)
        pd.testing.assert_frame_equal(full.iloc[:60], part)
        j = 40
        fit = np.linalg.lstsq(np.c_[x[j - c.window:j], np.ones(c.window)],
                              noisy["asset_close"].iloc[j - c.window:j], rcond=None)[0]
        assert np.allclose(full.iloc[j][["slope", "intercept"]].astype(float), fit, atol=1e-6)
        implied = fit[0] * x[j - c.window:j].mean() / noisy["asset_close"].iloc[j - c.window:j].mean()
        assert abs(full.iloc[j]["implied_beta"] - implied) < 1e-6
        hist = full["gap"].iloc[j - c.z_window:j]
        assert abs(full["zscore"].iloc[j] - (full["gap"].iloc[j] - hist.mean()) / hist.std()) < 1e-8
        a, ta, _ = simulate(full, c)
        b, tb, _ = simulate(part, c)
        pd.testing.assert_frame_equal(a.loc[b.index, ["cash", "shares", "equity"]], b[["cash", "shares", "equity"]])
        pd.testing.assert_frame_equal(ta[ta["fill_timestamp"] <= b.index[-1].isoformat()].reset_index(drop=True),
                                      tb.reset_index(drop=True))
        # The implied-beta guard marks out-of-range bars as not ready and leaves the rest alone.
        g = make_signals(noisy, replace(c, gap_beta_min=0.0, gap_beta_max=1.5))
        bad = (full["implied_beta"] < 0) | (full["implied_beta"] > 1.5)
        assert bad.any() and not g["ready"][bad].any()
        assert (g["ready"][~bad] == full["ready"][~bad]).all()
    # Force: buy, profitable sell signal, unprofitable next-open expiration, profitable later fill,
    # then an unsold loss held across a market night.
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
    # Missing minutes restart the gap fits (arb.py behaviour); empty evaluation dates are rejected.
    missing = make_signals(frame.drop(idx[20]), cfg)
    assert not missing.loc[idx[21:26], "ready"].any()
    try:
        simulate(signals, replace(cfg, start="2026-01-01"))
    except ValueError as exc:
        assert "evaluation dates" in str(exc)
    else:
        raise AssertionError("Empty evaluation dates were accepted")
    # Runtime invariants raise BacktestError explicitly (they survive python -O).
    broken = Account(cfg, cfg.max_shares)
    broken.cash = -1.0
    try:
        check_book(broken, cfg, 100.0)
    except BacktestError:
        pass
    else:
        raise AssertionError("negative cash was not caught")
    broken = Account(cfg, cfg.max_shares)
    broken.buy(100, 5)
    broken.cash += 1.0  # Money from nowhere: the reconciliation must fail.
    try:
        check_book(broken, cfg, 100.0)
    except BacktestError:
        pass
    else:
        raise AssertionError("an unreconciled book was not caught")
    return frame, result, trades, summary


def _check_residual_signal():
    days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    frame = _synthetic_sessions(days)
    cfg = Config(beta_window=300, beta_prior_weight=60, k=15, min_bars=5, vol_window=120,
                 vol_prior_weight=10, tod_sessions=2, tod_min_sessions=1, slippage_bps=0)
    cfg.validate()
    out = make_resid_signals(frame, cfg)
    n = len(frame)
    o_a, c_a = frame.asset_open.to_numpy(), frame.asset_close.to_numpy()
    o_v, c_v = frame.reference_open.to_numpy(), frame.reference_close.to_numpy()
    first = np.arange(n) % 390 == 0
    ra = np.where(first, np.log(c_a / o_a), np.log(c_a / np.r_[np.nan, c_a[:-1]]))
    rv = np.where(first, np.log(c_v / o_v), np.log(c_v / np.r_[np.nan, c_v[:-1]]))
    assert np.allclose(out.ret_asset, ra, atol=1e-14) and np.allclose(out.ret_reference, rv, atol=1e-14)
    j = 2 * 390 + 100  # Third session, minute 100.
    # Beta used at j: shrunk OLS slope over the last beta_window pairs through j-1 (direct lstsq).
    x, y = rv[j - cfg.beta_window:j], ra[j - cfg.beta_window:j]
    b_ols = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0][0]
    b = (cfg.beta_window * b_ols + cfg.beta_prior_weight * cfg.beta_prior) / (cfg.beta_window + cfg.beta_prior_weight)
    assert abs(out.beta.iloc[j] - b) < 1e-9, (out.beta.iloc[j], b)
    assert abs(out.beta_through_t.iloc[j - 1] - b) < 1e-9
    assert 1.15 < out.beta.iloc[-1] < 1.45, "beta should recover the true 1.3"
    e = out.resid.to_numpy()
    assert abs(e[j] - (ra[j] - b * rv[j])) < 1e-12
    # Time-of-day variance at j: mean e^2 at minutes 95..105 of previous tod_sessions sessions.
    sess, mod = np.arange(n) // 390, np.arange(n) % 390
    past = (sess >= 2 - cfg.tod_sessions) & (sess < 2) & (np.abs(mod - 100) <= 5) & np.isfinite(e)
    assert abs(out.tod_var.iloc[j] - np.mean(e[past] ** 2)) < 1e-15
    m0 = 390 + 2  # Second session, minute 2: no smoothing for opening minutes.
    assert abs(out.tod_var.iloc[m0] - e[2] ** 2) < 1e-18
    assert out.tod_var.iloc[:390].isna().all() and not out.ready.iloc[:390].any()
    # Level: mean e^2/tod over last vol_window valid residuals strictly before j, shrunk to 1.
    u = e ** 2 / out.tod_var.to_numpy()
    prior_u = u[:j][np.isfinite(u[:j])][-cfg.vol_window:]
    level = (prior_u.sum() + cfg.vol_prior_weight) / (len(prior_u) + cfg.vol_prior_weight)
    assert abs(out.vol_level.iloc[j] - level) < 1e-12
    window = np.arange(j - cfg.k + 1, j + 1)
    z = e[window].sum() / np.sqrt(level * out.tod_var.to_numpy()[window].sum())
    assert abs(out.zscore.iloc[j] - z) < 1e-9
    # Partial window at start of session, and the min-bars rule.
    s3 = 2 * 390
    assert not out.ready.iloc[s3:s3 + cfg.min_bars - 1].any() and out.ready.iloc[s3 + cfg.min_bars - 1]
    w3 = np.arange(s3, s3 + 7)
    z3 = e[w3].sum() / np.sqrt(out.vol_level.iloc[s3 + 6] * out.tod_var.to_numpy()[w3].sum())
    assert abs(out.zscore.iloc[s3 + 6] - z3) < 1e-9
    # Exponential kernel.
    ce = replace(cfg, kernel="exp", k=10)
    oe = make_resid_signals(frame, ce)
    lam = 0.5 ** (1 / ce.k)
    w = lam ** (j - np.arange(s3, j + 1))
    tv = oe.tod_var.to_numpy()[s3:j + 1]
    ze = (w * e[s3:j + 1]).sum() / np.sqrt(oe.vol_level.iloc[j] * (w * w * tv).sum())
    assert abs(oe.zscore.iloc[j] - ze) < 1e-9
    # Calibration on pure noise: z is roughly standard normal.
    zz = out.zscore.dropna()
    assert 0.75 < zz.std() < 1.25, zz.std()
    # A planted idiosyncratic drop makes AAPL cheap at that close; earlier signals stay unchanged.
    shock = frame.copy()
    jj = 3 * 390 + 200
    shock.iloc[jj:, shock.columns.get_loc("asset_close")] *= 0.985
    shock.iloc[jj + 1:, shock.columns.get_loc("asset_open")] *= 0.985
    so = make_resid_signals(shock, cfg)
    assert so.zscore.iloc[jj] < -5 and so.buy_signal.iloc[jj] and not so.buy_signal.iloc[jj - 1]
    pd.testing.assert_frame_equal(so.iloc[:jj], out.iloc[:jj].assign(
        asset_close=shock.asset_close.iloc[:jj], asset_open=shock.asset_open.iloc[:jj]), check_exact=True)
    # A missing minute inside a session does not break the residual chain or restart the signal.
    gap = make_resid_signals(frame.drop(frame.index[j]), cfg)
    assert gap.ready.iloc[j:j + 3].all()


def _forced(times, z, opens, closes, ready=None):
    z = np.asarray(z, dtype=float)
    ready = np.isfinite(z) if ready is None else np.asarray(ready)
    return pd.DataFrame(dict(asset_open=np.asarray(opens, float), asset_close=np.asarray(closes, float),
                             reference_open=400.0, reference_close=400.0, zscore=z, ready=ready,
                             buy_signal=ready & (z <= -2), sell_signal=ready & (z >= 2)), index=times)


def _check_target_machinery():
    cfg = Config(slippage_bps=0, max_hold=3, z_exit=0.0, z_threshold=2.0, eod_exit_min=None,
                 last_entry_min=None, base_exposure=0.0, cheap_exposure=1.0, rich_exposure=0.0)
    day1 = pd.date_range("2025-01-02 14:30", periods=13, freq="min", tz="UTC")
    day2 = pd.date_range("2025-01-03 14:30", periods=6, freq="min", tz="UTC")
    times = day1.append(day2)
    z = [0, -2.5, -2.2, -1.0, 0.5, -2.5, -2.4, -1.5, -2.3, -2.6, -1.0, -2.1, -3.0, np.nan, -2.2, 0.1, 0, 0, 0]
    opens = [100] * 19
    closes = [100] * 19
    opens[2], opens[5], opens[6], opens[9] = 100, 102, 99, 97
    res, trades, summary = simulate(_forced(times, z, opens, closes), cfg)
    sides = trades["side"].tolist()
    fills = [pd.Timestamp(t) for t in trades.fill_timestamp]
    # Enter at 1 -> buy at 2; revert at 4 -> sell at 5; re-enter at 5 -> buy at 6.
    # Time stop at 8 (held 3 bars) -> sell at 9; no re-entry at 9; re-arm at 10.
    # Enter at 11 -> buy at 12; held overnight; 13 not ready; 14 time stop -> sell at 15.
    assert sides == ["buy", "sell", "buy", "sell", "buy", "sell"], sides
    assert fills == [times[2], times[5], times[6], times[9], times[12], times[15]], fills
    assert (pd.to_datetime(trades.fill_timestamp) > pd.to_datetime(trades.signal_timestamp)).all()
    assert trades.qty.tolist()[:2] == [100, 100] and trades.price.iloc[1] == 102 and trades.price.iloc[2] == 99
    assert trades.qty.iloc[2] == 102, "target from equity at close (10,200 / 100), filled at the next open"
    assert math.isclose(trades.realized_pnl.iloc[1], 200) and math.isclose(trades.realized_pnl.iloc[3], -204)
    dec = res["decision"].tolist()
    assert dec[1].startswith("enter_cheap") and dec[4].startswith("exit_revert") and dec[8].startswith("exit_time")
    assert dec[9] == "hold" and dec[11] == "enter_cheap_submit_buy" and dec[13] == "warmup"
    assert dec[14].startswith("exit_time") and res.signal_state.iloc[12] == 1 and res.signal_state.iloc[13] == 1
    assert summary["final_shares"] == 0 and res.shares.min() >= 0 and (res.cash >= -1e-9).all()
    assert res.buy_hold_asset.iloc[0] == cfg.initial_cash and res.buy_hold_asset.iloc[1] == cfg.initial_cash
    assert summary["attribution"]["cheap_episodes"] == 3 and summary["attribution"]["rich_episodes"] == 0
    # A last-bar order expires overnight and is re-sent at the next session.
    z2 = [0] * 12 + [-2.5] + [np.nan, -2.5, -2.5, 0, 0, 0]
    res2, trades2, s2 = simulate(_forced(times, z2, [100] * 19, [100] * 19), cfg)
    assert s2["order_counts"]["expired_missing_minute_or_session"] == 1
    assert pd.Timestamp(trades2.fill_timestamp.iloc[0]) == times[14]
    # Rich side from a long base, end-of-day exit, entry cutoff, costs and cash-limited fills.
    c3 = replace(cfg, base_exposure=1.0, rich_exposure=0.0, cheap_exposure=1.0, max_hold=60,
                 eod_exit_min=380, last_entry_min=382, slippage_bps=10, commission_bps=5, fee=1)
    z3 = [0, 0, 2.5, 2.6, 2.4, 0.5, 0, 2.2, 2.1, 2.3, 2.5, 2.5, 0] + [0] * 6
    o3 = [100, 101] + [100] * 17
    res3, trades3, s3 = simulate(_forced(times, z3, o3, [100] * 19), c3)
    unit = 101 * (1 + 10 / 10_000) * (1 + 5 / 10_000)
    assert trades3.qty.iloc[0] == int((10_000 - 1) // unit) < 99 and trades3.side.iloc[0] == "buy"
    assert res3.target_shares.iloc[1] == trades3.qty.iloc[0], "cash-limited fill is accepted, not chased"
    assert res3.decision.iloc[2].startswith("enter_rich") and res3.decision.iloc[5] == "hold"
    assert res3.decision.iloc[6].startswith("exit_revert_submit_buy")
    assert trades3.side.to_list() == ["buy", "sell", "buy", "sell", "buy"], trades3.side.to_list()
    assert res3.decision.iloc[7].startswith("enter_rich")
    assert res3.decision.iloc[10] == "exit_eod_submit_buy"
    assert pd.Timestamp(trades3.fill_timestamp.iloc[-1]) == times[11]
    assert not any(d.startswith("enter") for d in res3.decision.iloc[8:].tolist())
    assert (res3.cash >= -1e-9).all()
    sells = trades3[trades3.side == "sell"]
    assert np.allclose(sells.price, 100 * (1 - 10 / 10_000))
    assert np.allclose(trades3.fee.to_numpy(), trades3.qty * trades3.price * 5 / 10_000 + 1)
    equity = res3.cash.iloc[-1] + res3.shares.iloc[-1] * 100
    assert math.isclose(equity - c3.initial_cash, s3["realized_pnl"] + s3["unrealized_pnl"], abs_tol=1e-6)
    # FIFO market sells realize losses as well as gains.
    book = Account(cfg, 1000)
    book.buy(100, 10)
    book.buy(90, 10)
    sold = book.sell_market(95, 15)
    assert sold["qty"] == 15 and math.isclose(sold["realized_pnl"], 10 * -5 + 5 * 5)
    assert book.shares == 5 and math.isclose(book.basis, 450)
    assert book.sell_market(95, 100)["qty"] == 5 and book.sell_market(95, 1) is None


def _check_overlay_and_reports():
    """No-op sides, re-targeting only on exposure changes, attribution, base-only counterfactual,
    exposure-matched benchmark and split-date segments."""
    cfg = Config(slippage_bps=1, max_hold=50, z_threshold=2.0, eod_exit_min=None, last_entry_min=None,
                 split_dates="2025-01-03")  # Base 1.0, cheap 1.0 (no-op), rich 0.0.
    times = pd.date_range("2025-01-02 14:30", periods=20, freq="min", tz="UTC").append(
        pd.date_range("2025-01-03 14:30", periods=20, freq="min", tz="UTC"))
    z = [0.0] * 40
    z[3:7] = [-2.5, -2.6, -2.2, -1.0]  # Cheap: must do nothing (cheap exposure = base).
    z[10:13] = [2.4, 2.1, 1.0]  # Rich at 10 -> sell at 11; reverts at 13.
    z[25:28] = [2.2, 2.3, 0.5]  # Rich again on day 2 -> sell at 26; reverts at 28.
    rng = np.random.default_rng(3)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 40)))
    opens = np.r_[100.0, closes[:-1]] * np.exp(rng.normal(0, 0.001, 40))
    res, trades, s = simulate(_forced(times, z, opens, closes), cfg)
    dec = res["decision"].tolist()
    assert not any("cheap" in d for d in dec) and dec[3] == "hold", dec[:8]
    assert trades.side.tolist() == ["buy", "sell", "buy", "sell", "buy"], trades.side.tolist()
    fills = [pd.Timestamp(t) for t in trades.fill_timestamp]
    assert fills == [times[1], times[11], times[14], times[26], times[29]], fills
    assert s["attribution"]["rich_episodes"] == 2 and s["attribution"]["cheap_episodes"] == 0
    # No re-target while exposure is unchanged: shares only move on fills.
    moved = np.flatnonzero(np.diff(res.shares.to_numpy()) != 0) + 1
    assert [times[i] for i in moved] == fills
    # Flipping between two episodes with the SAME exposure sends no order, even after a price move.
    zf = [0.0] * 40
    zf[5], zf[6], zf[7], zf[8] = -2.5, -1.0, 2.5, 1.0
    pf = [100.0] * 7 + [90.0] * 33  # Same-exposure target from current equity would be 52, not 49.
    rf, tf, _ = simulate(_forced(times, zf, pf, pf), replace(cfg, cheap_exposure=0.5, rich_exposure=0.5))
    assert rf.decision.iloc[5] == "enter_cheap_submit_sell" and rf.decision.iloc[7] == "enter_rich"
    assert tf.side.tolist() == ["buy", "sell", "buy"] and pd.Timestamp(tf.fill_timestamp.iloc[2]) == times[10]
    # Attribution equals the scoring harness's formulas.
    sh, px, eq = res.shares.to_numpy(float), res.asset_close.to_numpy(float), res.equity.to_numpy(float)
    held, dpx = sh[:-1], np.diff(px)
    att = s["attribution"]
    assert math.isclose(att["holding_pnl"], held @ dpx, abs_tol=1e-9)
    assert math.isclose(att["timing_pnl"], held @ dpx - held.mean() * dpx.sum(), abs_tol=1e-9)
    assert math.isclose(att["execution_and_cost_pnl"], eq[-1] - 10_000 - held @ dpx, abs_tol=1e-9)
    # Base-only counterfactual = the same run with no signal; overlay P&L is the difference.
    quiet, _, sq = simulate(_forced(times, [0.0] * 40, opens, closes), cfg)
    assert np.allclose(res.base_only_equity, quiet.equity) and np.allclose(res.base_only_shares, quiet.shares)
    assert math.isclose(att["overlay_pnl"], eq[-1] - quiet.equity.iloc[-1], abs_tol=1e-9)
    assert "base_only" in s["benchmarks"] and sq["attribution"]["overlay_pnl"] == 0
    # Pure buy-and-hold when no side differs from the base (no base-only rerun needed).
    hold, thold, _ = simulate(_forced(times, z, opens, closes), replace(cfg, rich_exposure=1.0))
    assert len(thold) == 1 and np.allclose(hold.equity, quiet.equity) and "base_only_equity" not in hold
    # Exposure-matched benchmark: average share count bought at bar 1's open.
    k = int(round(sh.mean()))
    unit = opens[1] * (1 + 1 / 10_000)
    curve = 10_000 - k * unit + k * closes
    curve[0] = 10_000
    assert att["exposure_matched_shares"] == k
    assert math.isclose(s["benchmarks"]["buy_hold_asset_exposure_matched"]["final_equity"], curve[-1], abs_tol=1e-9)
    # Segments of one continuous run chain back to the whole-run return.
    seg = s["segments"]
    assert [r["first_date"] for r in seg] == ["2025-01-02", "2025-01-03"] and sum(r["bars"] for r in seg) == 40
    for key, whole in (("return_pct", s["strategy"]["return_pct"]),
                       ("buy_hold_asset_return_pct", s["benchmarks"]["buy_hold_asset"]["return_pct"])):
        chained = 100 * (np.prod([1 + r[key] / 100 for r in seg]) - 1)
        assert math.isclose(chained, whole, abs_tol=1e-9), (key, chained, whole)
    assert sum(r["fills"] for r in seg) == len(trades)
    assert math.isclose(sum(r["total_pnl"] for r in seg), eq[-1] - 10_000, abs_tol=1e-9)
    assert math.isclose(sum(r["holding_pnl"] for r in seg), held @ dpx, abs_tol=1e-9)


def _check_calendar():
    """Loader drops auction bars and knows early closes; rules use the scheduled close."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Vendor-like data: pre/post-market on 07-02, auction at 13:00 and after-hours at 16:00 on 07-03.
        idx = pd.date_range("2025-07-02 09:29", "2025-07-02 16:01", freq="min").append(
            pd.date_range("2025-07-03 09:29", "2025-07-03 13:00", freq="min")).append(
            pd.DatetimeIndex([pd.Timestamp("2025-07-03 16:00")]))
        price = pd.DataFrame(dict(open=100.0 + np.arange(len(idx)) * 0.01,
                                  close=100.0 + np.arange(len(idx)) * 0.01), index=idx)
        for ticker in ("AAPL", "VOO"):
            (root / ticker).mkdir()
            price.to_csv(root / ticker / "bars.csv", index_label="timestamp")
        cfg = Config(input_timezone="America/New_York")
        counts = {}
        for name, c in (("nyse", cfg), ("off", replace(cfg, early_close_calendar=False)),
                        ("extra", replace(cfg, early_close="2025-07-02"))):
            pair = load_pair(root / "AAPL", root / "VOO", c)
            local = pair.index.tz_convert("America/New_York")
            counts[name] = {d: (int((local.strftime("%Y-%m-%d") == d).sum()),
                                local[local.strftime("%Y-%m-%d") == d][-1].strftime("%H:%M"))
                            for d in ("2025-07-02", "2025-07-03")}
            assert local[0].strftime("%H:%M") == "09:30"
        assert counts["nyse"] == {"2025-07-02": (390, "15:59"), "2025-07-03": (210, "12:59")}, counts
        assert counts["off"] == {"2025-07-02": (390, "15:59"), "2025-07-03": (211, "13:00")}, counts
        assert counts["extra"] == {"2025-07-02": (210, "12:59"), "2025-07-03": (210, "12:59")}, counts
    # On an early close, exit the episode at 12:58 (fill 12:59); no entry after 12:30.
    early = pd.date_range(pd.Timestamp("2025-07-03 09:30", tz="America/New_York"), periods=210, freq="min")
    nxt = pd.date_range(pd.Timestamp("2025-07-07 09:30", tz="America/New_York"), periods=15, freq="min")
    times = early.append(nxt).tz_convert("UTC")
    clock = np.r_[np.arange(210), np.arange(15)] + SESSION_OPEN
    z = np.zeros(len(times))
    z[(clock >= 12 * 60 + 20) & (np.arange(len(times)) < 210)] = 2.5
    z[210:220] = np.nan
    c = Config(slippage_bps=0, max_hold=300)
    res, trades, s = simulate(_forced(times, z, [100.0] * len(times), [100.0] * len(times)), c)
    local = pd.DatetimeIndex(trades.fill_timestamp.map(pd.Timestamp)).tz_convert("America/New_York")
    assert [t.strftime("%m-%d %H:%M") for t in local] == ["07-03 09:31", "07-03 12:21", "07-03 12:59"], local
    assert res.decision.iloc[210 - 2].startswith("exit_eod")
    assert res.signal_state.iloc[209] == 0 and res.shares.iloc[209] > 0
    z2 = np.zeros(len(times))
    z2[clock == 12 * 60 + 35] = 2.5
    z2[210:220] = np.nan
    res2, trades2, _ = simulate(_forced(times, z2, [100.0] * len(times), [100.0] * len(times)), c)
    assert len(trades2) == 1 and not any(d.startswith("enter") for d in res2.decision)
    # The same 12:35 signal on a regular day does start one (cutoff 15:30).
    reg = pd.date_range(pd.Timestamp("2025-07-02 09:30", tz="America/New_York"), periods=390, freq="min")
    times3 = reg.tz_convert("UTC")
    z3 = np.zeros(390)
    z3[12 * 60 + 35 - SESSION_OPEN] = 2.5
    res3, trades3, _ = simulate(_forced(times3, z3, [100.0] * 390, [100.0] * 390), c)
    assert len(trades3) == 3 and res3.decision.iloc[12 * 60 + 35 - SESSION_OPEN].startswith("enter_rich")
    # Without the calendar, the 12:58 exit never happens and the episode is carried overnight.
    res4, _, _ = simulate(_forced(times, z, [100.0] * len(times), [100.0] * len(times)),
                          replace(c, early_close_calendar=False))
    assert res4.signal_state.iloc[209] == -1 and res4.shares.iloc[209] == 0


def _check_invariance_and_units():
    """Future-data invariance of the whole default pipeline (signals, state machine, fills, base-only
    run), by truncation and by perturbing every later price; and timestamp-unit independence."""
    days = ["2025-06-26", "2025-06-27", "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-07"]
    frame = _synthetic_sessions(days, seed=11, early=("2025-07-03",))
    cfg = Config(beta_window=390, beta_prior_weight=60, k=15, min_bars=5, vol_window=120,
                 vol_prior_weight=10, tod_sessions=3, tod_min_sessions=1, z_threshold=1.5, max_hold=20)
    full_sig = make_signals(frame, cfg)
    full, full_trades, _ = simulate(full_sig, cfg)
    assert len(full_trades) >= 8 and (full_trades.side == "sell").sum() >= 3, len(full_trades)
    local = frame.index.tz_convert("America/New_York")
    cuts = [int(np.flatnonzero((local.strftime("%Y-%m-%d") == d) & (local.hour * 60 + local.minute == m))[0])
            for d, m in (("2025-06-30", 11 * 60 + 3), ("2025-07-02", 15 * 60 + 58), ("2025-07-03", 12 * 60 + 40),
                         ("2025-07-07", 10 * 60 + 15))]
    rng = np.random.default_rng(99)
    checked = 0
    for cut in cuts:
        variants = {"truncated": frame.iloc[:cut + 1]}
        shocked = frame.copy()
        walk = np.exp(np.cumsum(rng.normal(0, 0.004, (len(frame) - cut - 1, 4)), axis=0))
        shocked.iloc[cut + 1:] = shocked.iloc[cut + 1:].to_numpy() * walk
        variants["perturbed"] = shocked
        for name, variant in variants.items():
            sig = make_signals(variant, cfg)
            pd.testing.assert_frame_equal(sig.iloc[:cut + 1], full_sig.iloc[:cut + 1], check_exact=True)
            res, tr, _ = simulate(sig, cfg)
            # Every past column; the cut bar can carry _cancelled_at_end in its decision text.
            pd.testing.assert_frame_equal(res.iloc[:cut], full.iloc[:cut], check_exact=True)
            pd.testing.assert_frame_equal(res.iloc[[cut]].drop(columns="decision"),
                                          full.iloc[[cut]].drop(columns="decision"), check_exact=True)
            assert res.decision.iloc[cut].startswith(full.decision.iloc[cut])
            end = frame.index[cut].isoformat()
            a = full_trades[full_trades.fill_timestamp <= end].reset_index(drop=True)
            b = tr[tr.fill_timestamp <= end].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            checked += len(a)
        if name == "perturbed":
            assert not np.allclose(res.equity.iloc[-50:], full.equity.iloc[-50:]), "perturbation had no effect"
    assert checked > 10
    # Every fill is exactly one minute after its signal, at that minute's raw open +/- slippage.
    sig_t, fill_t = pd.to_datetime(full_trades.signal_timestamp), pd.to_datetime(full_trades.fill_timestamp)
    assert ((fill_t - sig_t) == pd.Timedelta(minutes=1)).all()
    raw = frame.loc[pd.DatetimeIndex(fill_t), "asset_open"].to_numpy()
    sign = np.where(full_trades.side == "buy", 1, -1)
    assert np.allclose(full_trades.price, raw * (1 + sign * cfg.slippage_bps / 10_000), rtol=0, atol=1e-9)
    # No episode is open at any session's last bar, including the early close.
    last_bar = np.r_[local.strftime("%Y-%m-%d")[1:] != local.strftime("%Y-%m-%d")[:-1], True]
    assert (full.signal_state.to_numpy()[last_bar[-len(full):]] == 0).all()
    # Timestamps stored in s / ms / us give identical results (pandas 3 unit safety), both signals.
    for c in (cfg, replace(cfg, signal="gap", window=20, max_hold=20)):
        base_res, base_tr, _ = simulate(make_signals(frame, c), c)
        for unit in ("s", "ms", "us"):
            other = frame.copy()
            other.index = frame.index.as_unit(unit)
            res, tr, _ = simulate(make_signals(other, c), c)
            assert np.array_equal(res.equity.to_numpy(), base_res.equity.to_numpy()), unit
            assert tr.fill_timestamp.tolist() == base_tr.fill_timestamp.tolist(), unit


def _check_cli(frame_gap, result, trades, summary):
    parser = build_parser()
    defaults = config_from_args(parser.parse_args([]))
    assert defaults == Config(), "CLI defaults must equal the Config defaults"
    assert defaults.asset == "AAPL" and defaults.slippage_bps >= 1 and parser.parse_args([]).plot is True
    assert parser.parse_args(["--no-plot"]).plot is False and parser.parse_args(["--plot"]).plot is True
    assert config_from_args(parser.parse_args(["--eod-exit-min", "none"])).eod_exit_min is None
    assert config_from_args(parser.parse_args(["--last-entry-min", "none"])).last_entry_min is None
    assert config_from_args(parser.parse_args(["--no-early-close-calendar"])).early_close_calendar is False
    assert config_from_args(parser.parse_args(["--gap-beta-min", "0"])).gap_beta_min == 0.0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Loading respects time zones, overlapping files and conflicting duplicates.
        for ticker, prefix in (("AAPL", "asset"), ("VOO", "reference")):
            folder = root / "tz" / ticker / "2025"
            folder.mkdir(parents=True)
            raw = frame_gap[[prefix + "_open", prefix + "_close"]].copy()
            raw.columns = ["open", "close"]
            raw.index = raw.index.tz_convert("America/New_York").tz_localize(None)
            raw.to_csv(folder / "day.csv", index_label="timestamp")
            raw.iloc[:3].to_csv(folder / "overlap.csv", index_label="timestamp")
        c = Config(input_timezone="America/New_York")
        loaded = load_pair(root / "tz" / "AAPL", root / "tz" / "VOO", c)
        pd.testing.assert_frame_equal(loaded, frame_gap, check_freq=False, check_names=False)
        raw.iloc[:3].assign(close=1).to_csv(folder / "conflict.csv", index_label="timestamp")
        try:
            read_asset(root / "tz" / "VOO", c)
        except ValueError as exc:
            assert "conflicting" in str(exc)
        else:
            raise AssertionError("Conflicting duplicates were accepted")
        save_results(result, trades, summary, root / "legacy_out", plot=True)
        assert (root / "legacy_out" / "equity.png").stat().st_size > 0
        json.loads((root / "legacy_out" / "summary.json").read_text())
        days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        _write_csvs(_synthetic_sessions(days, seed=21), root / "data")
        out = root / "cli_out"
        quiet = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
        with quiet[0], quiet[1]:
            _check_cli_runs(root, out)


def _check_cli_runs(root: Path, out: Path):
    """End-to-end CLI runs (called with stdout/stderr captured)."""
    main(["--data-root", str(root / "data"), "--output", str(out), "--split-dates", "2025-01-06"])
    s = json.loads((out / "summary.json").read_text())
    expected = asdict(replace(Config(), split_dates="2025-01-06"))
    assert s["parameters"] == expected, "the CLI must run exactly the documented defaults"
    assert (out / "equity.png").stat().st_size > 0 and (out / "trades.csv").exists()
    r = pd.read_csv(out / "results.csv")
    assert {"equity", "asset_close", "shares", "base_only_equity", "signal_state"} <= set(r.columns)
    chained = 100 * (np.prod([1 + seg["return_pct"] / 100 for seg in s["segments"]]) - 1)
    assert len(s["segments"]) == 2 and math.isclose(chained, s["strategy"]["return_pct"], abs_tol=1e-9)
    out2 = root / "cli_end"
    main(["--data-root", str(root / "data"), "--output", str(out2), "--end", "2025-01-07", "--no-plot"])
    r2 = pd.read_csv(out2 / "results.csv")
    assert not (out2 / "equity.png").exists()
    pd.testing.assert_frame_equal(r2[["equity", "shares"]], r[["equity", "shares"]].iloc[:len(r2)])
    # Bad inputs exit with code 2.
    for argv in (["--data-root", str(root / "data"), "--start", "2025-02-01", "--end", "2025-01-01"],
                 ["--data-root", str(root / "data"), "--z-exit", "3"],
                 ["--data-root", str(root / "data"), "--end", "20250107"],
                 ["--data-root", str(root / "missing")]):
        try:
            main(argv + ["--output", str(root / "bad")])
        except SystemExit as exc:
            assert exc.code == 2, argv
        else:
            raise AssertionError(f"bad input accepted: {argv}")


def self_test():
    """Offline behavioral checks, including execution gaps and future-data invariance."""
    if not __debug__:
        raise SystemExit("Run --self-test without python -O: the checks use assert statements.")
    frame_gap, result, trades, summary = _check_gap_and_legacy()
    _check_residual_signal()
    _check_target_machinery()
    _check_overlay_and_reports()
    _check_calendar()
    _check_invariance_and_units()
    _check_cli(frame_gap, result, trades, summary)
    print("PASS: arb.py gap OLS + legacy lots + implied-beta guard; residual signal vs direct")
    print("recomputation (beta, time-of-day variance, vol level, box/exp z, min bars, missing minute);")
    print("target state machine (entries, reversion/time/end-of-day exits, re-arming, overnight expiry,")
    print("cash-limited fills, costs, FIFO losses, no-op sides); attribution, base-only and")
    print("exposure-matched benchmarks, split segments; early-close calendar and auction bars;")
    print("future-data invariance by truncation and perturbation; s/ms/us/ns timestamps; explicit")
    print("invariant errors; CLI defaults end to end, plot flag, --end truncation, parser errors.")


def _none_or(kind):
    def parse(text):
        return None if str(text).strip().lower() in ("none", "") else kind(text)
    parse.__name__ = kind.__name__
    return parse


def build_parser():
    d = Config()
    # set the asset and refrence up there in the config
    # d.asset ='AMZN'
    # d.reference ='VOO'
    parser = argparse.ArgumentParser(prog="arb_improved.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default= '/home/fariborz/Downloads/live/main_codes/arbitrage/data_', type=Path, help="folder with <asset>/ and <reference>/ CSV subfolders")
    parser.add_argument("--asset-data", type=Path, help="alternative: asset CSV file or folder")
    parser.add_argument("--reference-data", type=Path, help="alternative: reference CSV file or folder")
    parser.add_argument("--asset", default=d.asset)
    parser.add_argument("--reference", default=d.reference)
    parser.add_argument("--signal", choices=["resid", "gap"], default=d.signal,
                        help="resid: calibrated beta-hedged residual (default); gap: arb.py's price-level gap")
    g = parser.add_argument_group("residual signal")
    g.add_argument("--beta-window", type=int, default=d.beta_window, help="1-minute return pairs for beta")
    g.add_argument("--beta-prior", type=float, default=d.beta_prior)
    g.add_argument("--beta-prior-weight", type=float, default=d.beta_prior_weight, help="pseudo-minutes")
    g.add_argument("--kernel", choices=["box", "exp"], default=d.kernel)
    g.add_argument("--k", type=int, default=d.k, help="box length or exp half-life, bars")
    g.add_argument("--min-bars", type=int, default=d.min_bars, help="residuals needed in the session")
    g.add_argument("--vol-window", type=int, default=d.vol_window, help="minutes for the volatility level")
    g.add_argument("--vol-prior-weight", type=float, default=d.vol_prior_weight)
    g.add_argument("--tod-sessions", type=int, default=d.tod_sessions, help="sessions in the time-of-day profile")
    g.add_argument("--tod-min-sessions", type=int, default=d.tod_min_sessions)
    g = parser.add_argument_group("arb.py price-gap signal (--signal gap)")
    g.add_argument("--window", type=int, default=d.window, help="preceding consecutive minutes for the OLS")
    g.add_argument("--mode", choices=["percent", "zscore"], default=d.mode)
    g.add_argument("--threshold-pct", type=float, default=d.threshold_pct, help="0.2 means 0.2%%, percent mode")
    g.add_argument("--z-window", type=_none_or(int), default=d.z_window, help="preceding gaps; default = window")
    g.add_argument("--gap-beta-min", type=_none_or(float), default=d.gap_beta_min,
                   help="not ready below this implied return beta ('none' = off)")
    g.add_argument("--gap-beta-max", type=_none_or(float), default=d.gap_beta_max,
                   help="not ready above this implied return beta ('none' = off)")
    g = parser.add_argument_group("entries, exits and sizing")
    g.add_argument("--z-threshold", type=float, default=d.z_threshold, help="episode entry |z|")
    g.add_argument("--z-exit", type=float, default=d.z_exit, help="episode ends once z is back inside +/- this")
    g.add_argument("--max-hold", type=int, default=d.max_hold, help="time stop, bars (trading minutes)")
    g.add_argument("--exec-mode", choices=["target", "legacy"], default=d.exec_mode)
    g.add_argument("--base-exposure", type=float, default=d.base_exposure, help="fraction of equity, neutral")
    g.add_argument("--cheap-exposure", type=float, default=d.cheap_exposure,
                   help="fraction of equity while cheap (equal to base = side off)")
    g.add_argument("--rich-exposure", type=float, default=d.rich_exposure,
                   help="fraction of equity while rich (equal to base = side off)")
    g.add_argument("--last-entry-min", type=_none_or(int), default=d.last_entry_min,
                   help="no new episode in the last N minutes before the scheduled close")
    g.add_argument("--eod-exit-min", type=_none_or(int), default=d.eod_exit_min,
                   help="end every episode N minutes before the scheduled close")
    g.add_argument("--chunk", type=int, default=d.chunk, help="legacy execution: shares per buy/sell")
    g.add_argument("--max-shares", type=int, default=d.max_shares)
    g = parser.add_argument_group("account and costs")
    g.add_argument("--initial-cash", type=float, default=d.initial_cash)
    g.add_argument("--commission-bps", type=float, default=d.commission_bps)
    g.add_argument("--fee", type=float, default=d.fee, help="fixed dollars per filled order")
    g.add_argument("--slippage-bps", type=float, default=d.slippage_bps, help="adverse slippage per fill")
    g.add_argument("--execution", choices=["next_open", "next_close"], default=d.execution)
    g = parser.add_argument_group("calendar, dates and output")
    g.add_argument("--input-timezone", default=d.input_timezone, help="timezone of naive CSV timestamps")
    g.add_argument("--market-timezone", default=d.market_timezone)
    g.add_argument("--early-close-calendar", action=argparse.BooleanOptionalAction,
                   default=d.early_close_calendar, help="use the built-in NYSE 13:00 early closes")
    g.add_argument("--early-close", default=d.early_close, help="extra 13:00 early-close dates, comma separated")
    g.add_argument("--start", help="first evaluation date YYYY-MM-DD; earlier data warms up the signal")
    g.add_argument("--end", help="last evaluation date YYYY-MM-DD, inclusive; later bars are dropped first")
    g.add_argument("--split-dates", default=d.split_dates,
                   help="report sub-periods of this run starting at these dates (comma separated)")
    g.add_argument("--output", type=Path, default=Path("results"))
    g.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                   help="write equity.png (--no-plot skips it)")
    parser.add_argument("--self-test", action="store_true", help="run the offline behavioral checks")
    return parser


def config_from_args(args) -> Config:
    return Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    try:
        cfg = config_from_args(args)
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
        if cfg.end:  # Bars after evaluation never influence results.
            data = data[market_days(data.index, cfg.market_timezone) <= cfg.end]
        results, trades, summary = simulate(make_signals(data, cfg), cfg)
        save_results(results, trades, summary, args.output, plot=args.plot)
    except (ValueError, OSError, KeyError, BacktestError) as exc:
        parser.error(str(exc))
    p, att = summary["strategy"], summary["attribution"]
    print(f"{summary['start']} to {summary['end']} ({len(results):,} aligned evaluation bars)")
    print(f"Strategy equity ${p['final_equity']:,.2f}; P&L ${p['pnl']:,.2f}; return {p['return_pct']:.2f}%; "
          f"max drawdown {p['max_drawdown_pct']:.2f}%")
    print(f"Realized ${summary['realized_pnl']:,.2f}; unrealized ${summary['unrealized_pnl']:,.2f}; "
          f"open shares {summary['final_shares']}; fills {summary['buy_fills'] + summary['sell_fills']}")
    print(f"Average exposure {att['avg_exposure_pct']:.1f}%; timing P&L ${att['timing_pnl']:,.2f}; "
          f"execution and costs ${att['execution_and_cost_pnl']:,.2f}"
          + (f"; overlay vs base-only ${att['overlay_pnl']:,.2f}" if "overlay_pnl" in att else ""))
    for name, stats in summary["benchmarks"].items():
        print(f"{name}: {stats['return_pct']:.2f}%")
    for seg in summary["segments"]:
        print(f"segment {seg['first_date']}..{seg['last_date']}: {seg['return_pct']:.2f}% "
              f"(buy & hold {seg['buy_hold_asset_return_pct']:.2f}%), fills {seg['fills']}")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
