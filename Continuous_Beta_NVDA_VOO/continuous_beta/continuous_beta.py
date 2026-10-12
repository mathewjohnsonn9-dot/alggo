#!/usr/bin/env python3
"""Plot continuous causal return beta for two folders of daily minute CSV files.

Example:
    python continuous_beta.py --x_dir /path/VOO --y_dir /path/NVDA \
        --ticker_x VOO --ticker_y NVDA --out_dir beta_results --show

Y is the dependent asset (NVDA); X is the driver (VOO):
    log_return_Y[t] = alpha[t] + beta[t] * log_return_X[t] + noise[t]

Three estimators run recursively from the first usable return to the last:
  kalman:    time-varying alpha/beta, robust update, past-only noise adaptation.
  ewma:      exponentially weighted covariance(X,Y) / variance(X).
  expanding: ordinary least-squares beta using every earlier usable observation.

No full-sample fitting, backward smoothing, future calibration, or daily reset.
Default gap handling skips returns spanning more than one minute; filter state
survives overnight, missing minutes, and skipped days. No missing prices are filled.

Timestamp-aware strings retain their timezone. Naive/numeric timestamps use
--input_tz (default UTC), and plots/session filters use --market_tz.

Outputs: beta_minutes.csv, beta_changes.csv, paired_days.csv, summary.json,
and beta_timeline.png. --html also writes a Plotly chart if plotly is installed.
The script estimates/plots relationships; it submits no trades.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
TIME_ALIASES = ("timestamp", "datetime", "date_time", "time", "date", "ts")
CLOSE_ALIASES = ("close", "adj close", "adj_close", "close_price", "price", "last", "c")
EPS = 1e-12


def locate_column(columns: list[str], explicit: str | None, aliases: tuple[str, ...], path: Path) -> str:
    mapping = {str(c).strip().casefold(): c for c in columns}
    if explicit:
        key = explicit.casefold()
        if key not in mapping:
            raise ValueError(f"{path.name}: column {explicit!r} missing; columns={columns}")
        return mapping[key]
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    raise ValueError(f"{path.name}: cannot locate {aliases[0]} column; columns={columns}")


def parse_timestamps(values: pd.Series, input_tz: str) -> pd.Series:
    """Preserve offsets; localize naive times explicitly; support Unix epochs."""
    text = values.astype(str).str.strip()
    nonempty = values.notna() & text.ne("")
    numeric = pd.to_numeric(values, errors="coerce")
    valid_numeric = numeric.dropna()
    # Recognize epochs, but do not mistake YYYYMMDD strings for epoch seconds.
    if len(valid_numeric) == int(nonempty.sum()) and len(valid_numeric) and valid_numeric.abs().median() >= 1e8:
        mag = float(valid_numeric.abs().median())
        unit = "ns" if mag >= 1e17 else "us" if mag >= 1e14 else "ms" if mag >= 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if bool(aware[nonempty].all()) and bool(nonempty.any()):
        return pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
    if bool(aware.any()):
        raise ValueError("Mixed timezone-aware and naive timestamps in one CSV; make them consistent.")
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    return parsed.dt.tz_localize(input_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def index_day_files(folder: Path, pattern: str, start: str | None, end: str | None) -> tuple[dict[str, Path], list[str]]:
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")
    indexed: dict[str, Path] = {}
    undated: list[str] = []
    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        matches = DATE_PATTERN.findall(path.stem)
        candidates = set()
        for y, m, d in matches:
            try:
                candidates.add(date(int(y), int(m), int(d)).isoformat())
            except ValueError:
                pass
        if len(candidates) != 1:
            undated.append(str(path))
            continue
        day = candidates.pop()
        if (start and day < start) or (end and day > end):
            continue
        if day in indexed:
            raise ValueError(f"Multiple files for {day} in {folder}: {indexed[day].name}, {path.name}. "
                             "Restrict --x_glob / --y_glob to the desired interval.")
        indexed[day] = path
    return indexed, undated


def read_day(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        headers = list(pd.read_csv(path, nrows=0).columns)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["timestamp", "close"]), {"raw": 0, "invalid": 0, "duplicates": 0, "session_filtered": 0}
    tc = locate_column(headers, args.ts_col, TIME_ALIASES, path)
    pc = locate_column(headers, args.close_col, CLOSE_ALIASES, path)
    frame = pd.read_csv(path, usecols=[tc, pc])
    raw = len(frame)
    ts = parse_timestamps(frame[tc], args.input_tz)
    px = pd.to_numeric(frame[pc], errors="coerce")
    good = ts.notna() & np.isfinite(px) & px.gt(0)
    clean = pd.DataFrame({"timestamp": ts[good], "close": px[good]}).sort_values("timestamp", kind="stable")
    before_dupes = len(clean)
    clean = clean.drop_duplicates("timestamp", keep="last")
    duplicates = before_dupes - len(clean)
    before_session = len(clean)
    if args.session == "rth" and len(clean):
        local = clean["timestamp"].dt.tz_convert(args.market_tz)
        minute = local.dt.hour * 60 + local.dt.minute
        clean = clean.loc[(minute >= 9 * 60 + 30) & (minute < 16 * 60)]
    return clean.reset_index(drop=True), {"raw": raw, "invalid": raw - int(good.sum()),
                                         "duplicates": duplicates, "session_filtered": before_session - len(clean)}


@dataclass
class Moments:
    """Stable online centered moments; decay=1 yields expanding OLS."""
    decay: float = 1.0
    count: int = 0
    weight: float = 0.0
    weight2: float = 0.0
    mean_x: float = 0.0
    mean_y: float = 0.0
    c_xx: float = 0.0
    c_yy: float = 0.0
    c_xy: float = 0.0

    def update(self, x: float, y: float) -> None:
        old_weight = self.decay * self.weight
        new_weight = old_weight + 1.0
        dx, dy = x - self.mean_x, y - self.mean_y
        adjustment = old_weight / new_weight
        self.c_xx = self.decay * self.c_xx + adjustment * dx * dx
        self.c_yy = self.decay * self.c_yy + adjustment * dy * dy
        self.c_xy = self.decay * self.c_xy + adjustment * dx * dy
        self.mean_x += dx / new_weight
        self.mean_y += dy / new_weight
        self.weight, self.weight2 = new_weight, self.decay ** 2 * self.weight2 + 1.0
        self.count += 1

    def beta(self, min_std_bps: float) -> float:
        if self.count < 2 or self.var_x < min_std_bps ** 2:
            return math.nan
        return self.c_xy / max(self.c_xx, EPS)

    @property
    def var_x(self) -> float:
        return max(0.0, self.c_xx / max(self.weight, EPS))

    @property
    def corr(self) -> float:
        if self.count < 2 or self.c_xx <= EPS or self.c_yy <= EPS:
            return math.nan
        return max(-1.0, min(1.0, self.c_xy / math.sqrt(self.c_xx * self.c_yy)))

    @property
    def effective_n(self) -> float:
        return self.weight ** 2 / max(self.weight2, EPS)


class KalmanBeta:
    """Random-walk regression; scalar Joseph covariance update for numerical stability."""

    def __init__(self, args: argparse.Namespace):
        self.alpha, self.beta = 0.0, args.beta_prior
        self.p00, self.p01, self.p11 = 100.0, 0.0, args.beta_prior_sd ** 2
        self.q_alpha, self.q_beta = args.kalman_q_alpha, args.kalman_q_beta
        self.noise = args.initial_noise_bps ** 2
        self.noise_floor = args.noise_floor_bps ** 2
        self.noise_decay = math.exp(-math.log(2) / args.noise_half_life)
        self.huber_z = args.huber_z
        self.last_prior = self.beta
        self.last_innovation = math.nan
        self.last_weight = 1.0

    def update(self, x: float, y: float) -> None:
        self.last_prior = self.beta
        a, b, c = self.p00 + self.q_alpha, self.p01, self.p11 + self.q_beta
        innovation = y - self.alpha - self.beta * x
        self.last_innovation = innovation
        predicted_variance = max(a + 2 * b * x + c * x * x + self.noise, EPS)
        z = abs(innovation) / math.sqrt(predicted_variance)
        weight = min(1.0, self.huber_z / max(z, EPS)) if self.huber_z > 0 else 1.0
        self.last_weight = weight
        effective_noise = self.noise / max(weight, EPS)
        s = max(a + 2 * b * x + c * x * x + effective_noise, EPS)
        k0, k1 = (a + b * x) / s, (b + c * x) / s
        self.alpha += k0 * innovation
        self.beta += k1 * innovation
        # (I-KH) P (I-KH)' + K R K': preserves positive covariance.
        u, v, w, z2 = 1 - k0, -k0 * x, -k1, 1 - k1 * x
        self.p00 = max(u * u * a + 2 * u * v * b + v * v * c + k0 * k0 * effective_noise, EPS)
        self.p01 = u * w * a + (u * z2 + v * w) * b + v * z2 * c + k0 * k1 * effective_noise
        self.p11 = max(w * w * a + 2 * w * z2 * b + z2 * z2 * c + k1 * k1 * effective_noise, EPS)
        bound = math.sqrt(self.p00 * self.p11)
        self.p01 = max(-bound, min(bound, self.p01))
        # Current innovation affects the noise used at the NEXT observation only.
        squared = innovation * innovation
        if self.huber_z > 0:
            squared = min(squared, self.huber_z ** 2 * predicted_variance)
        self.noise = max(self.noise_floor, self.noise_decay * self.noise + (1 - self.noise_decay) * squared)


@dataclass
class ChangeTracker:
    threshold: float
    confirmation: int
    reference: float | None = None
    direction: int = 0
    run: int = 0
    events: int = 0

    def update(self, beta: float, ready: bool, used: bool) -> tuple[bool, int, float, float]:
        before = self.reference if self.reference is not None else math.nan
        if not ready or not used:
            self.run, self.direction = 0, 0
            return False, 0, before, math.nan
        if self.reference is None:
            self.reference = beta
            return False, 0, math.nan, 0.0
        delta = beta - self.reference
        direction = (1 if delta > 0 else -1) if abs(delta) >= self.threshold else 0
        if direction == 0:
            self.run, self.direction = 0, 0
        elif direction == self.direction:
            self.run += 1
        else:
            self.direction, self.run = direction, 1
        changed = self.run >= self.confirmation
        if changed:
            self.events += 1
            self.reference, self.run, self.direction = beta, 0, 0
        return changed, direction if changed else 0, before, delta


class BetaEngine:
    """State persists across every matched CSV; no loader controls filter state."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.kalman = KalmanBeta(args)
        self.ewma = Moments(math.exp(-math.log(2) / args.ewma_half_life))
        self.expanding = Moments()
        self.changes = ChangeTracker(args.change_threshold, args.confirm_bars)
        self.previous: tuple[pd.Timestamp, float, float] | None = None
        self.previous_beta = math.nan
        self.rows, self.used, self.skipped = 0, 0, 0

    def step(self, timestamp: pd.Timestamp, px_x: float, px_y: float, paired_day: str) -> dict[str, Any]:
        gap = math.nan
        reason = "first_price"
        rx = ry = math.nan
        used = False
        if self.previous:
            old_ts, old_x, old_y = self.previous
            gap = (timestamp - old_ts).total_seconds() / 60
            if gap <= 0:
                raise ValueError("Timestamps must increase strictly across paired files.")
            if self.args.gap_policy == "skip" and abs(gap - 1.0) > self.args.minute_tolerance:
                reason = "gap_not_one_minute"
            else:
                rx, ry = math.log(px_x / old_x), math.log(px_y / old_y)
                self.kalman.update(rx * 1e4, ry * 1e4)
                self.ewma.update(rx * 1e4, ry * 1e4)
                self.expanding.update(rx * 1e4, ry * 1e4)
                self.used += 1
                used, reason = True, ""
        self.previous = timestamp, px_x, px_y
        if not used:
            self.skipped += 1
        beta_k = self.kalman.beta if self.used else math.nan
        beta_e = self.ewma.beta(self.args.min_driver_std_bps)
        beta_a = self.expanding.beta(self.args.min_driver_std_bps)
        selected = {"kalman": beta_k, "ewma": beta_e, "expanding": beta_a}[self.args.method]
        ready = (self.used >= self.args.min_observations and self.ewma.var_x >= self.args.min_driver_std_bps ** 2
                 and math.isfinite(selected))
        delta = selected - self.previous_beta if math.isfinite(selected) and math.isfinite(self.previous_beta) else math.nan
        changed, sign, reference, reference_delta = self.changes.update(selected, ready, used)
        minute_moved = bool(ready and used and math.isfinite(delta) and abs(delta) >= self.args.minute_change_threshold)
        row = {
            "timestamp": timestamp.isoformat(), "paired_day": paired_day, "bar_index": self.rows,
            "close_x": px_x, "close_y": px_y, "gap_minutes": gap,
            "observation_used": int(used), "skip_reason": reason, "observations_so_far": self.used,
            "return_x_log": rx, "return_y_log": ry,
            "beta_kalman": beta_k, "beta_kalman_before_update": self.kalman.last_prior if used else beta_k,
            "alpha_kalman_bps": self.kalman.alpha, "beta_kalman_sd": math.sqrt(self.kalman.p11),
            "kalman_innovation_bps": self.kalman.last_innovation if used else math.nan,
            "kalman_update_weight": self.kalman.last_weight if used else math.nan,
            "beta_ewma": beta_e, "beta_expanding": beta_a, "ewma_correlation": self.ewma.corr,
            "driver_std_bps": math.sqrt(self.ewma.var_x), "ewma_effective_observations": self.ewma.effective_n,
            "beta": selected, "beta_ready": int(ready), "beta_delta": delta,
            "beta_moved_this_minute": int(minute_moved), "beta_changed": int(changed),
            "beta_change_direction": sign, "beta_reference_before": reference,
            "beta_vs_reference": reference_delta,
            "hedge_x_shares_per_y_share": selected * px_y / px_x if math.isfinite(selected) else math.nan,
        }
        self.rows += 1
        self.previous_beta = selected
        return row


PLOT_FIELDS = ("timestamp", "bar_index", "close_x", "close_y", "beta", "beta_ready", "beta_delta",
               "beta_kalman", "beta_kalman_sd", "beta_ewma", "beta_expanding", "beta_change_direction")


class PlotBuffer:
    """Bound plotting memory without downsampling estimator inputs or CSV rows."""
    def __init__(self, limit: int):
        self.limit, self.stride = limit, 1
        self.rows: list[dict] = []
        self.last: dict | None = None
        self.events: list[dict] = []
        self.event_stride, self.event_count = 1, 0

    def add(self, row: dict) -> None:
        item = {k: row[k] for k in PLOT_FIELDS}
        self.last = item
        if row["bar_index"] % self.stride == 0:
            self.rows.append(item)
            if self.limit and len(self.rows) > self.limit:
                self.rows = self.rows[::2]
                self.stride *= 2
        if row["beta_changed"]:
            if self.event_count % self.event_stride == 0:
                self.events.append(item)
                if len(self.events) > 2000:
                    self.events = self.events[::2]
                    self.event_stride *= 2
            self.event_count += 1

    def finish(self) -> pd.DataFrame:
        if self.last and (not self.rows or self.rows[-1]["bar_index"] != self.last["bar_index"]):
            self.rows.append(self.last)
        return pd.DataFrame(self.rows)


def x_coordinates(frame: pd.DataFrame, args: argparse.Namespace):
    if args.x_axis == "trading":
        return frame["bar_index"].to_numpy()
    return pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(args.market_tz).to_numpy()


def make_plot(frame: pd.DataFrame, events: list[dict], args: argparse.Namespace, target: Path, first_prices: tuple[float, float]) -> None:
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from zoneinfo import ZoneInfo

    x = x_coordinates(frame, args)
    ready = frame["beta_ready"].astype(bool).to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1.5, .8]})
    axes[0].plot(x, frame["close_x"] / first_prices[0] * 100, color="#7c8292", lw=1.2, label=args.ticker_x)
    axes[0].plot(x, frame["close_y"] / first_prices[1] * 100, color="#4263a6", lw=1.2, label=args.ticker_y)
    axes[0].set_ylabel("Price (first bar = 100)")
    axes[0].legend(loc="upper left", ncol=2)
    primary = frame["beta"].to_numpy(dtype=float)
    axes[1].plot(x, np.where(~ready, primary, np.nan), color="#149393", ls=":", lw=1.2, label="Initial / insufficient-data estimate")
    axes[1].plot(x, np.where(ready, primary, np.nan), color="#078080", lw=1.9, label=f"{args.method.capitalize()} beta (selected)")
    if args.method == "kalman":
        sd = frame["beta_kalman_sd"].to_numpy(dtype=float)
        axes[1].fill_between(x, np.where(ready, primary - 1.96 * sd, np.nan),
                             np.where(ready, primary + 1.96 * sd, np.nan), color="#078080", alpha=.12,
                             label="Approx. model band (±1.96 SD)")
    if args.compare:
        for name, column, color in [("Kalman", "beta_kalman", "#078080"), ("EWMA", "beta_ewma", "#de8b27"),
                                    ("Since-start OLS", "beta_expanding", "#858590")]:
            if column != "beta_" + args.method:
                axes[1].plot(x, np.where(ready, frame[column].to_numpy(dtype=float), np.nan), color=color, lw=1.0, alpha=.8, label=name)
    if events:
        event_frame = pd.DataFrame(events)
        xe = x_coordinates(event_frame, args)
        for sign, marker, color, label in [(1, "^", "#259457", "Confirmed increase"), (-1, "v", "#bf4f4f", "Confirmed decrease")]:
            mask = event_frame["beta_change_direction"].to_numpy() == sign
            axes[1].scatter(np.asarray(xe)[mask], event_frame.loc[mask, "beta"], marker=marker, color=color, s=25, zorder=4, label=label)
    axes[1].axhline(0, color="#555555", lw=.6, alpha=.5)
    axes[1].set_ylabel(f"Return beta: {args.ticker_y} on {args.ticker_x}")
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)
    axes[2].plot(x, np.where(ready, frame["beta_delta"], np.nan), color="#078080", lw=.9)
    axes[2].axhline(args.minute_change_threshold, color="#999999", ls="--", lw=.7)
    axes[2].axhline(-args.minute_change_threshold, color="#999999", ls="--", lw=.7)
    axes[2].axhline(0, color="#555555", lw=.6)
    axes[2].set_ylabel("Beta change / bar")
    if args.x_axis == "calendar":
        tz = ZoneInfo(args.market_tz)
        locator = mdates.AutoDateLocator(tz=tz, minticks=4, maxticks=10)
        axes[2].xaxis.set_major_locator(locator)
        axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=tz))
        axes[2].set_xlabel(f"Time ({args.market_tz}); missing periods have no updates")
    else:
        idx = frame["bar_index"].to_numpy()
        times = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(args.market_tz)
        def label_tick(value, _position):
            j = int(np.searchsorted(idx, value))
            return times.iloc[min(j, len(times) - 1)].strftime("%Y-%m-%d\n%H:%M")
        axes[2].xaxis.set_major_locator(mticker.MaxNLocator(8, integer=True))
        axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(label_tick))
        axes[2].set_xlabel(f"Matched bars; nights/gaps compressed ({args.market_tz})")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.ticker_y} / {args.ticker_x} · continuous causal beta · no daily reset", fontsize=15)
    fig.text(.01, .005, f"Persistent change: |beta − last confirmed level| ≥ {args.change_threshold:g}, "
             f"for {args.confirm_bars} usable bars. Plot sampling only; CSV retains every matched bar.", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, .02, 1, .96])
    fig.savefig(target, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)


def make_html(frame: pd.DataFrame, events: list[dict], args: argparse.Namespace, target: Path, first_prices: tuple[float, float]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ValueError("--html requires plotly. Install it with: python -m pip install plotly") from exc
    x = x_coordinates(frame, args)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.04, row_heights=[.25, .5, .25])
    for column, price, name in [("close_x", first_prices[0], args.ticker_x), ("close_y", first_prices[1], args.ticker_y)]:
        fig.add_trace(go.Scattergl(x=x, y=frame[column] / price * 100, name=name), row=1, col=1)
    columns = [("beta_" + args.method, args.method.capitalize())]
    if args.compare:
        columns += [(c, n) for c, n in [("beta_kalman", "Kalman"), ("beta_ewma", "EWMA"), ("beta_expanding", "Since-start OLS")] if c != columns[0][0]]
    for column, name in columns:
        fig.add_trace(go.Scattergl(x=x, y=frame[column], name=name + " beta"), row=2, col=1)
    if events:
        ev = pd.DataFrame(events)
        fig.add_trace(go.Scattergl(x=x_coordinates(ev, args), y=ev.beta, mode="markers", name="Confirmed changes"), row=2, col=1)
    fig.add_trace(go.Scattergl(x=x, y=frame.beta_delta, name="Minute beta change"), row=3, col=1)
    fig.update_layout(height=950, title=f"{args.ticker_y} on {args.ticker_x}: continuous causal beta", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
    fig.write_html(target, include_plotlyjs=True, auto_open=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x_dir", "--x-dir", type=Path, help="Driver asset CSV folder, e.g. VOO")
    p.add_argument("--y_dir", "--y-dir", type=Path, help="Target asset CSV folder, e.g. NVDA")
    p.add_argument("--data_root", "--data-root", type=Path, help="Alternative: parent containing ticker subfolders")
    p.add_argument("--ticker_x", "--ticker-x", default="VOO")
    p.add_argument("--ticker_y", "--ticker-y", default="NVDA")
    p.add_argument("--x_glob", default="*.csv")
    p.add_argument("--y_glob", default="*.csv")
    p.add_argument("--ts_col", default=None, help="Timestamp column; otherwise inferred")
    p.add_argument("--close_col", default=None, help="Price column; otherwise inferred (close first)")
    p.add_argument("--input_tz", "--input-tz", default="UTC", help="Timezone of naive CSV timestamps")
    p.add_argument("--market_tz", default="America/New_York")
    p.add_argument("--session", choices=["all", "rth"], default="all", help="all=all supplied bars; rth=09:30 ≤ time < 16:00 in market_tz")
    p.add_argument("--start", default=None, help="First file date, inclusive, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Last file date, inclusive, YYYY-MM-DD")
    p.add_argument("--method", choices=["kalman", "ewma", "expanding"], default="kalman")
    p.add_argument("--compare", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--beta_prior", type=float, default=1.0, help="Initial Kalman beta guess, not a fitted first-bar beta")
    p.add_argument("--beta_prior_sd", type=float, default=2.0)
    p.add_argument("--kalman_q_beta", "--kalman-q-beta", type=float, default=1e-5, help="Beta random-walk variance per usable return; smaller is steadier")
    p.add_argument("--kalman_q_alpha", type=float, default=.001, help="Intercept random-walk variance in squared bps per usable return")
    p.add_argument("--initial_noise_bps", type=float, default=10.0)
    p.add_argument("--noise_floor_bps", type=float, default=.5)
    p.add_argument("--noise_half_life", type=float, default=120.0, help="Past-only Kalman measurement-noise adaptation half-life")
    p.add_argument("--huber_z", type=float, default=4.0, help="Inflate measurement noise for large innovations; 0 disables")
    p.add_argument("--ewma_half_life", "--ewma-half-life", type=float, default=240.0, help="EWMA half-life in usable observations, not calendar minutes")
    p.add_argument("--min_observations", type=int, default=120, help="Initial estimates remain available but beta_ready=0 until this count")
    p.add_argument("--min_driver_std_bps", type=float, default=.1, help="Low-X-variation guard for beta availability/readiness")
    p.add_argument("--gap_policy", choices=["skip", "include"], default="skip", help="skip=use only one-minute returns; include=also model multi-minute/overnight returns")
    p.add_argument("--minute_tolerance", type=float, default=.05, help="Tolerance around a one-minute timestamp gap")
    p.add_argument("--minute_change_threshold", type=float, default=.01, help="Flag |beta[t]−beta[t−1]| at or above this value")
    p.add_argument("--change_threshold", "--change-threshold", type=float, default=.10, help="Deviation from the last confirmed beta level")
    p.add_argument("--confirm_bars", "--confirm-bars", type=int, default=3, help="Consecutive usable bars required to confirm that deviation")
    p.add_argument("--out_dir", "--out-dir", type=Path, default=Path("beta_results"))
    p.add_argument("--x_axis", choices=["calendar", "trading"], default="calendar")
    p.add_argument("--max_plot_points", type=int, default=100_000, help="Plot-memory bound; 0 stores every plotted point; CSV always has every row")
    p.add_argument("--gzip", action="store_true", help="Write compressed beta_minutes.csv.gz")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--show", action="store_true", help="Display Matplotlib's zoomable figure")
    p.add_argument("--html", action="store_true", help="Also save a self-contained zoomable HTML (requires plotly)")
    p.add_argument("--print_every", type=int, default=0, help="Print every N matched bars; 1 prints every minute")
    p.add_argument("--quiet", action="store_true")
    return p


def validate(args: argparse.Namespace) -> None:
    from zoneinfo import ZoneInfo
    ZoneInfo(args.input_tz)
    ZoneInfo(args.market_tz)
    if args.data_root:
        args.x_dir = args.x_dir or args.data_root / args.ticker_x
        args.y_dir = args.y_dir or args.data_root / args.ticker_y
    if args.x_dir is None or args.y_dir is None:
        raise ValueError("Supply --x_dir and --y_dir, or --data_root with ticker subfolders.")
    args.x_dir, args.y_dir = args.x_dir.expanduser().resolve(), args.y_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    for name in ["beta_prior_sd", "initial_noise_bps", "noise_floor_bps", "noise_half_life", "ewma_half_life",
                 "min_driver_std_bps", "minute_change_threshold", "change_threshold"]:
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"--{name} must be finite and positive.")
    for name in ["kalman_q_alpha", "kalman_q_beta", "huber_z", "minute_tolerance"]:
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"--{name} must be finite and nonnegative.")
    if not math.isfinite(args.beta_prior):
        raise ValueError("--beta_prior must be finite.")
    if args.min_observations < 2 or args.confirm_bars < 1 or args.max_plot_points < 0 or args.print_every < 0:
        raise ValueError("min_observations ≥ 2, confirm_bars ≥ 1, max_plot_points ≥ 0, print_every ≥ 0 required.")
    for value in [args.start, args.end]:
        if value:
            date.fromisoformat(value)
    if args.start and args.end and args.start > args.end:
        raise ValueError("--start must not be later than --end.")
    if args.show and args.no_plot:
        raise ValueError("--show and --no_plot cannot be used together.")
    if args.html:
        import importlib.util
        if importlib.util.find_spec("plotly") is None:
            raise ValueError("--html requires plotly. Install it with: python -m pip install plotly")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate(args)
    x_files, undated_x = index_day_files(args.x_dir, args.x_glob, args.start, args.end)
    y_files, undated_y = index_day_files(args.y_dir, args.y_glob, args.start, args.end)
    paired = sorted(set(x_files) & set(y_files))
    if not paired:
        raise ValueError("No paired daily files. Filenames need YYYY-MM-DD or YYYYMMDD dates. "
                         f"Dated files found: {args.ticker_x}={len(x_files)}, {args.ticker_y}={len(y_files)}.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.out_dir / "paired_days.csv"
    minute_path = args.out_dir / ("beta_minutes.csv.gz" if args.gzip else "beta_minutes.csv")
    event_path = args.out_dir / "beta_changes.csv"
    engine = BetaEngine(args)
    plot = PlotBuffer(args.max_plot_points)
    days: list[dict] = []
    union = sorted(set(x_files) | set(y_files))
    for day in union:
        if day not in x_files or day not in y_files:
            days.append({"day": day, "status": "unpaired", "x_file": str(x_files.get(day, "")),
                         "y_file": str(y_files.get(day, "")), "aligned_bars": 0, "usable_returns": 0})
    first_prices = None
    first_timestamp = last_timestamp = None
    last_row = None
    opener = gzip.open if args.gzip else open
    with opener(minute_path, "wt", newline="", encoding="utf-8") as stream, event_path.open("w", newline="", encoding="utf-8") as event_stream:
        writer = event_writer = None
        for day in paired:
            dx, stats_x = read_day(x_files[day], args)
            dy, stats_y = read_day(y_files[day], args)
            panel = dx.merge(dy, on="timestamp", how="inner", suffixes=("_x", "_y")).sort_values("timestamp")
            before = engine.used
            day_row = {"day": day, "status": "paired" if len(panel) else "no_common_timestamps",
                       "x_file": str(x_files[day]), "y_file": str(y_files[day]),
                       "aligned_bars": len(panel), "x_clean_bars": len(dx), "y_clean_bars": len(dy),
                       "x_unmatched_bars": len(dx) - len(panel), "y_unmatched_bars": len(dy) - len(panel),
                       **{f"x_{k}": v for k, v in stats_x.items()}, **{f"y_{k}": v for k, v in stats_y.items()}}
            for ts, px_x, px_y in panel[["timestamp", "close_x", "close_y"]].itertuples(index=False, name=None):
                if first_prices is None:
                    first_prices, first_timestamp = (float(px_x), float(px_y)), ts.isoformat()
                row = engine.step(ts, float(px_x), float(px_y), day)
                last_row, last_timestamp = row, row["timestamp"]
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    event_writer = csv.DictWriter(event_stream, fieldnames=list(row))
                    writer.writeheader()
                    event_writer.writeheader()
                writer.writerow(row)
                if row["beta_changed"]:
                    event_writer.writerow(row)
                plot.add(row)
                if args.print_every and engine.rows % args.print_every == 0 and not args.quiet:
                    marker = "CHANGED" if row["beta_changed"] else "no confirmed change"
                    print(f"{row['timestamp']} beta={row['beta']:.5f} Δ={row['beta_delta']:+.5f} "
                          f"ready={row['beta_ready']} used={row['observation_used']} {marker}")
            day_row["usable_returns"] = engine.used - before
            days.append(day_row)
            if not args.quiet:
                print(f"[DAY {day}] common bars={len(panel):,}, usable returns={engine.used-before:,}, cumulative={engine.used:,}")
    pd.DataFrame(days).sort_values("day").to_csv(daily_path, index=False)
    if not engine.rows:
        raise ValueError("Paired files have no shared timestamps after validation/session filtering. See paired_days.csv.")
    if not engine.used:
        raise ValueError("No usable returns. Check timezones and one-minute timestamps; see paired_days.csv. "
                         "--gap_policy include permits multi-minute returns explicitly.")
    frame = plot.finish()
    if not args.no_plot:
        make_plot(frame, plot.events, args, args.out_dir / "beta_timeline.png", first_prices)
    if args.html:
        make_html(frame, plot.events, args, args.out_dir / "beta_timeline.html", first_prices)
    summary = {"ticker_x": args.ticker_x, "ticker_y": args.ticker_y, "selected_method": args.method,
               "paired_file_days": len(paired), "days_with_aligned_bars": sum(d["aligned_bars"] > 0 for d in days),
               "unpaired_days_skipped": sum(d["status"] == "unpaired" for d in days),
               "undated_files_ignored_x": undated_x, "undated_files_ignored_y": undated_y,
               "first_timestamp": first_timestamp, "last_timestamp": last_timestamp,
               "matched_bars": engine.rows, "usable_returns": engine.used,
               "rows_without_filter_update": engine.skipped, "confirmed_beta_changes": engine.changes.events,
               "final_beta": last_row["beta"], "final_beta_ready": bool(last_row["beta_ready"]),
               "final_beta_kalman": last_row["beta_kalman"], "final_beta_ewma": last_row["beta_ewma"],
               "final_beta_expanding": last_row["beta_expanding"], "plot_stride": plot.stride,
               "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "notes": ["Beta is a return coefficient, not the price ratio.",
                         "All three estimators retain state across paired days.",
                         "EWMA/Kalman adapt over time; expanding OLS weights all usable history equally.",
                         "beta_ready=0 marks initial or low-driver-variation estimates.",
                         "beta_changed is a persistence/threshold heuristic, not a statistical hypothesis test.",
                         "Kalman uncertainty bands are approximate model bands.",
                         "No price filling, no full-day variance calibration, no backward smoothing."]}
    # Standard JSON: unavailable estimates are null, not NaN.
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        return value
    summary = json_safe(summary)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"[DONE] {engine.rows:,} matched bars, {engine.used:,} returns, {len(paired):,} paired days, "
              f"{summary['unpaired_days_skipped']:,} unpaired days skipped.")
        print(f"Selected beta: {last_row['beta']:.5f}; confirmed changes: {engine.changes.events:,}; ready={last_row['beta_ready']}")
        print(f"Outputs: {args.out_dir}")
        if not last_row["beta_ready"]:
            print("[NOTE] Final estimate is still initial or has insufficient driver variation.")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
