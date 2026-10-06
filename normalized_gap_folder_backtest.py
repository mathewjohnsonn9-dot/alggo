#!/usr/bin/env python3
"""Continuous long-only folder backtest driven by normalized price/filter gaps.

Run:
  python normalized_gap_folder_backtest.py --data_dir /path/to/TSLA --show

Also plots percentage deviations from the raw Kalman estimate and the
Kalman+EMA baseline: 100 * (actual close - estimate) / estimate.
Saved as price_minus_kalman_normalized.png in the output folder.
Positive percentages mean price is above the estimate; negative means below.

Trading gap = 100 * (actual close - baseline) / baseline.
Default baseline: Kalman+EMA (the smoothed green curve); --signal_baseline kalman
uses the raw Kalman estimate instead.
BUY one chunk on EVERY completed close with gap < -0.15 percent.
SELL ALL eligible profitable lots on EVERY completed close with gap > +0.15 percent.
Equality at either threshold produces no order. Thresholds are configurable
through --buy_threshold_pct and --sell_threshold_pct; 0.15 means 0.15%, not 15%.
Cash, total-share and inventory-cost limits apply to every buy; a final chunk
may be smaller than chunk_shares when the remaining capacity is smaller.

Signals use completed closes; fills use the NEXT observed bar's open. Sell
lots must be profitable after costs at both the decision close and fill open.
Unsold lots, cash and filter state carry across days.
Realized profits increase the trading limit by default. Equity includes every
unsold share marked at the close; there is no forced final liquidation.

Kalman/EMA and the previous regime diagnostics are included below; regime
diagnostics are informational and do not gate this threshold policy.
Kalman q/r and the diagnostics' volatility reference use warm-up-only
calibration. Both strategy and buy-and-hold start at the first open AFTER warm-up.
The displayed warm-up history is calibration history, not evaluated trading.

Requires Python 3.10+, numpy, pandas 2+, matplotlib. Input prices must be on a
consistent split-adjusted basis. Dividends, taxes and cash interest are not modeled.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Config:
    initial_capital: float = 10000.0
    buy_limit: int = 10000
    chunk_shares: int = 15
    warmup_bars: int = 200
    slippage: float = 0.0
    fee_per_share: float = 0.0
    use_realized_profit_limit: bool = True
    print_trades: bool = False
    buy_threshold_pct: float = -0.15
    sell_threshold_pct: float = 0.15
    signal_baseline: str = "kalman_ema"
    # Model defaults match the attached detector's command-line defaults.
    alpha: float = 0.45
    q_scale: float = 0.20
    r_scale: float = 0.40
    w: int = 10
    enter_th: float = 2.0
    exit_th: float = 0.5
    eps_slope: float = 0.0
    huber_delta_mult: float = 1.5
    huber_max_iter: int = 25
    huber_tol: float = 1e-6
    vol_w: int = 60
    vol_gamma: float = 0.5
    vol_clip_lo: float = 0.7
    vol_clip_hi: float = 1.8
    vol_ref_mode: str = "median"
    vol_ref_ewm_alpha: float = 0.02
    cusum_k: float = 0.5
    cusum_h: float = 6.0
    cusum_scale_with_vol: bool = False
    cusum_reset_on_switch: bool = False
    resid_w: int = 60
    resid_z_enter: float = 0.3
    resid_z_exit: float = 0.1

    def validate(self):
        float_names = (
            "initial_capital", "slippage", "fee_per_share", "alpha", "q_scale", "r_scale",
            "enter_th", "exit_th", "eps_slope", "huber_delta_mult", "huber_tol",
            "vol_gamma", "vol_clip_lo", "vol_clip_hi", "vol_ref_ewm_alpha",
            "cusum_k", "cusum_h", "resid_z_enter", "resid_z_exit",
            "buy_threshold_pct", "sell_threshold_pct",
        )
        if not np.isfinite([getattr(self, name) for name in float_names]).all():
            raise ValueError("Numeric settings must be finite")
        for name in ("buy_limit", "chunk_shares", "warmup_bars", "w", "huber_max_iter", "vol_w", "resid_w"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.buy_threshold_pct >= self.sell_threshold_pct:
            raise ValueError("buy_threshold_pct must be below sell_threshold_pct")
        if self.signal_baseline not in ("kalman", "kalman_ema"):
            raise ValueError("signal_baseline must be kalman or kalman_ema")
        if not 0 <= self.slippage < 1 or self.fee_per_share < 0:
            raise ValueError("Use 0 <= slippage < 1 and fee_per_share >= 0")
        if not 0 < self.alpha <= 1 or self.q_scale <= 0 or self.r_scale <= 0:
            raise ValueError("Use 0 < alpha <= 1 and positive q_scale/r_scale")
        if self.w < 3 or self.vol_w < 10 or self.resid_w < 10:
            raise ValueError("Use w >= 3, vol_w >= 10, resid_w >= 10")
        needed = max(self.w, max(10, self.vol_w // 5) + 1, max(10, self.resid_w // 5))
        if self.warmup_bars < needed:
            raise ValueError(f"warmup_bars must be at least {needed} for these model windows")
        if not 0 <= self.exit_th <= self.enter_th or self.enter_th <= 0:
            raise ValueError("Use 0 <= exit_th <= enter_th and enter_th > 0")
        if self.eps_slope < 0 or self.huber_delta_mult <= 0 or self.huber_tol <= 0:
            raise ValueError("Use eps_slope >= 0 and positive Huber delta/tolerance")
        if self.vol_gamma < 0 or not 0 < self.vol_clip_lo <= self.vol_clip_hi:
            raise ValueError("Use vol_gamma >= 0 and 0 < vol_clip_lo <= vol_clip_hi")
        if self.vol_ref_mode not in ("median", "ewm") or not 0 < self.vol_ref_ewm_alpha <= 1:
            raise ValueError("Use vol_ref_mode median/ewm and 0 < vol_ref_ewm_alpha <= 1")
        if self.cusum_k < 0 or self.cusum_h <= 0 or self.resid_z_enter < 0 or self.resid_z_exit < 0:
            raise ValueError("Use cusum_k/residual thresholds >= 0 and cusum_h > 0")


def kalman_rw_causal(y: np.ndarray, q: float, r: float, x0=None, p0: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    xhat = np.zeros(n, dtype=float)

    P = float(p0)
    x = float(y[0] if x0 is None else x0)

    for i in range(n):
        x_pred = x
        P_pred = P + q

        yi = y[i]
        if np.isfinite(yi):
            S = P_pred + r
            K = P_pred / S
            x = x_pred + K * (yi - x_pred)
            P = (1.0 - K) * P_pred
        else:
            x, P = x_pred, P_pred

        xhat[i] = x

    return xhat


def ema_causal(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    out[0] = x[0]
    a = float(alpha)
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def _wls_fit_slope_intercept(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    x = x.astype(float)
    y = y.astype(float)
    w = w.astype(float)

    Sw = np.sum(w)
    Sx = np.sum(w * x)
    Sxx = np.sum(w * x * x)
    Sy = np.sum(w * y)
    Sxy = np.sum(w * x * y)

    det = Sxx * Sw - Sx * Sx
    if det <= 1e-18 or not np.isfinite(det):
        return np.nan, np.nan, None, None, None

    inv = (1.0 / det) * np.array([[Sw, -Sx],
                                  [-Sx, Sxx]], dtype=float)

    a = inv[0, 0] * Sxy + inv[0, 1] * Sy
    b = inv[1, 0] * Sxy + inv[1, 1] * Sy

    yhat = a * x + b
    resid = y - yhat
    return float(a), float(b), inv, yhat, resid


def _mad_scale(resid: np.ndarray) -> float:
    r = np.asarray(resid, float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return float(1.4826 * max(mad, 1e-12))


def rolling_huber_slope_tstat(
    y: np.ndarray,
    w: int,
    huber_delta_mult: float = 1.5,
    huber_max_iter: int = 25,
    huber_tol: float = 1e-6,
    eps_slope: float = 0.0,
):
    y = np.asarray(y, dtype=float)
    n = len(y)

    slope = np.full(n, np.nan, dtype=float)
    resid_scale = np.full(n, np.nan, dtype=float)
    slope_se = np.full(n, np.nan, dtype=float)
    tstat = np.full(n, np.nan, dtype=float)

    if w < 3:
        return slope, resid_scale, slope_se, tstat

    x = np.arange(w, dtype=float)

    for t in range(w - 1, n):
        yw = y[t - w + 1: t + 1]
        if not np.all(np.isfinite(yw)):
            continue

        wgt = np.ones(w, dtype=float)
        a, b, inv_xtwx, yhat, resid = _wls_fit_slope_intercept(x, yw, wgt)
        if not np.isfinite(a):
            continue

        s0 = _mad_scale(resid)
        delta = float(huber_delta_mult) * s0

        prev_a = a
        for _ in range(int(huber_max_iter)):
            r = resid
            ar = np.abs(r)
            wgt = np.ones_like(ar)
            mask = ar > delta
            wgt[mask] = delta / np.maximum(ar[mask], 1e-12)

            a, b, inv_xtwx, yhat, resid = _wls_fit_slope_intercept(x, yw, wgt)
            if not np.isfinite(a):
                break
            if abs(a - prev_a) <= float(huber_tol) * (1.0 + abs(prev_a)):
                break
            prev_a = a

        if not np.isfinite(a) or inv_xtwx is None:
            continue

        s = _mad_scale(resid)

        weff = float(np.sum(wgt))
        df = max(weff - 2.0, 1.0)
        sigma2 = float(np.sum(wgt * (resid ** 2)) / df)
        sigma2 = max(sigma2, 1e-12)

        var_a = float(sigma2 * inv_xtwx[0, 0])
        se_a = float(np.sqrt(max(var_a, 1e-12)))
        ts = float(a / max(se_a, 1e-12))

        slope[t] = a
        resid_scale[t] = s
        slope_se[t] = se_a
        tstat[t] = ts

    if eps_slope > 0:
        mask = np.isfinite(slope) & (np.abs(slope) <= float(eps_slope))
        tstat[mask] = 0.0

    return slope, resid_scale, slope_se, tstat


def rolling_std_causal(x: np.ndarray, w: int, min_count: int = 10) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    out = np.full(n, np.nan, float)
    w = int(w)
    min_count = int(min_count)

    for t in range(n):
        s = max(0, t - w + 1)
        win = x[s:t + 1]
        win = win[np.isfinite(win)]
        if win.size < min_count:
            continue
        out[t] = float(np.std(win, ddof=0))
    return out


def ewm_mean_causal(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan, dtype=float)
    a = float(alpha)

    m = np.nan
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            out[i] = m
            continue
        if not np.isfinite(m):
            m = xi
        else:
            m = a * xi + (1.0 - a) * m
        out[i] = m
    return out


def cusum_regime_with_neutral_resid_veto(
    z: np.ndarray,
    enter_t: np.ndarray,
    exit_t: np.ndarray,
    k_t: np.ndarray,
    h_t: np.ndarray,
    z_res: np.ndarray,
    veto_enter_res_z: float,
    veto_exit_res_z: float,
    reset_on_switch: bool = True,
):
    """
    States: +1 UP, 0 NEUTRAL, -1 DOWN

    Residual veto logic:
      - To ENTER/REVERSE into DOWN, require z_res <= -veto_enter_res_z
      - To ENTER/REVERSE into UP,   require z_res >= +veto_enter_res_z

      - To EXIT from UP -> NEUTRAL, require z_res <= -veto_exit_res_z
      - To EXIT from DOWN -> NEUTRAL, require z_res >= +veto_exit_res_z

    Intuition:
      - Don't call it "DOWN" while price is still above baseline (z_res positive)
      - Don't exit UP on a tiny pullback above baseline
    """
    z = np.asarray(z, float)
    enter_t = np.asarray(enter_t, float)
    exit_t = np.asarray(exit_t, float)
    k_t = np.asarray(k_t, float)
    h_t = np.asarray(h_t, float)
    z_res = np.asarray(z_res, float)

    n = len(z)
    state = np.zeros(n, dtype=int)

    Spos = 0.0
    Sneg = 0.0
    s = 0

    ve = float(veto_enter_res_z)
    vx = float(veto_exit_res_z)

    for t in range(n):
        if not (np.isfinite(z[t]) and np.isfinite(enter_t[t]) and np.isfinite(exit_t[t]) and
                np.isfinite(k_t[t]) and np.isfinite(h_t[t]) and np.isfinite(z_res[t])):
            state[t] = s
            continue

        kt = float(k_t[t])
        ht = float(h_t[t])
        ent = float(enter_t[t])
        ex = float(exit_t[t])
        zr = float(z_res[t])

        # Update CUSUM accumulators
        Spos = max(0.0, Spos + float(z[t]) - kt)
        Sneg = max(0.0, Sneg - float(z[t]) - kt)

        ev_up = (Spos > ht)
        ev_dn = (Sneg > ht)

        # helper: residual gates
        allow_up = (zr >= +ve)
        allow_dn = (zr <= -ve)

        allow_exit_up = (zr <= -vx)  # only exit UP if price really dipped below baseline
        allow_exit_dn = (zr >= +vx)  # only exit DOWN if price really popped above baseline

        if s == 0:
            if ev_up and (z[t] >= ent) and allow_up:
                s = 1
                if reset_on_switch:
                    Spos, Sneg = 0.0, 0.0
            elif ev_dn and (z[t] <= -ent) and allow_dn:
                s = -1
                if reset_on_switch:
                    Spos, Sneg = 0.0, 0.0

        elif s == 1:
            # exit to neutral ONLY if both (signal fades) and (structure breaks)
            # if (z[t] <= ex) and allow_exit_up:
            #     s = 0
            # reverse to down ONLY if strong persistent opposite evidence AND structural break
            if ev_dn and (z[t] <= -ent) and allow_dn:
                s = -1
                if reset_on_switch:
                    Spos, Sneg = 0.0, 0.0

        elif s == -1:
            if (z[t] >= -ex) and allow_exit_dn:
                s = 0
            elif ev_up and (z[t] >= ent) and allow_up:
                s = 1
                if reset_on_switch:
                    Spos, Sneg = 0.0, 0.0

        state[t] = s

    return state


def parse_times(values, timezone):
    """Respect supplied offsets; treat naive timestamps as local exchange time."""
    text = values.astype(str).str.strip()
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    times = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if aware.any():
        times.loc[aware] = pd.to_datetime(text.loc[aware], format="mixed", utc=True)
    if (~aware).any():
        naive = pd.to_datetime(text.loc[~aware], format="mixed")
        times.loc[~aware] = naive.dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    if times.isna().any():
        raise ValueError("Missing or invalid timestamps")
    return times.dt.tz_convert(timezone)


def load_folder(folder, pattern="*.csv", timezone="America/New_York",
                recursive=False, output_dir=None):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"CSV folder does not exist: {folder}")
    output_dir = Path(output_dir).resolve() if output_dir else None
    candidates = folder.rglob(pattern) if recursive else folder.glob(pattern)
    frames, used = [], []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if output_dir and path.resolve().is_relative_to(output_dir):
            continue
        # Ignore derived copies produced by the standalone indicator scripts.
        if any(tag in path.stem.lower() for tag in ("_kalman_ema", "_regime_cusum_residveto")):
            continue
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        time_col = next((c for c in ("timestamp", "datetime", "date", "time",
                                    "unnamed: 0") if c in frame.columns), None)
        if time_col is None or not {"open", "close"}.issubset(frame.columns):
            raise ValueError(f"{path.name}: requires timestamp, open and close columns")
        if frame.empty:
            continue
        for c in ("symbol", "ticker"):
            if c in frame.columns:
                symbols = frame[c].dropna().astype(str).str.upper().unique()
                if len(symbols) != 1:
                    raise ValueError(f"{path.name}: use one instrument per run")
                frame["instrument"] = symbols[0]
        frame["timestamp"] = parse_times(frame[time_col], timezone)
        for c in ("open", "close"):
            frame[c] = pd.to_numeric(frame[c], errors="raise")
        prices = frame[["open", "close"]].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f"{path.name}: open/close must be finite positive prices")
        columns = ["timestamp", "open", "close"]
        if "instrument" in frame:
            columns.append("instrument")
        frames.append(frame[columns])
        used.append(path)
    if not frames:
        raise ValueError(f"No price CSV files match {pattern!r} in {folder}")
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    if "instrument" in data and data["instrument"].dropna().nunique() > 1:
        raise ValueError("Folder contains multiple instruments; select one with --pattern")
    duplicated = data.duplicated("timestamp", keep=False)
    if duplicated.any():
        conflicts = data.loc[duplicated].groupby("timestamp")[["open", "close"]].nunique()
        if (conflicts > 1).any().any():
            raise ValueError("Overlapping CSVs have conflicting prices at the same timestamp")
    removed = int(data.duplicated("timestamp").sum())
    data = data.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    return data, {"files": len(used), "duplicates_removed": removed, "rows": len(data)}


def normalized_flags(gap_pct, buy_threshold_pct, sell_threshold_pct):
    """Strict threshold conditions on every bar; invalid gaps never signal."""
    gap_pct = np.asarray(gap_pct, dtype=float)
    if gap_pct.ndim != 1:
        raise ValueError("gap_pct must be a 1D array")
    valid = np.isfinite(gap_pct)
    return valid & (gap_pct < buy_threshold_pct), valid & (gap_pct > sell_threshold_pct)


def make_signals(data, cfg):
    """Compute causal filters, threshold signals and informational diagnostics."""
    cfg.validate()
    if len(data) <= cfg.warmup_bars:
        raise ValueError("Need more rows than warmup_bars")
    close = data["close"].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("close must contain finite positive prices")

    # These two calibration references are frozen before the evaluation starts.
    noise_std = max(float(np.std(np.diff(close[:cfg.warmup_bars]))), 1e-8)
    q = (cfg.q_scale * noise_std) ** 2
    r = (cfg.r_scale * noise_std) ** 2
    kalman = kalman_rw_causal(close, q=q, r=r, p0=1.0)
    smooth = ema_causal(kalman, alpha=cfg.alpha)
    slope, resid_scale, slope_se, tstat = rolling_huber_slope_tstat(
        smooth, w=cfg.w, huber_delta_mult=cfg.huber_delta_mult,
        huber_max_iter=cfg.huber_max_iter, huber_tol=cfg.huber_tol,
        eps_slope=cfg.eps_slope,
    )
    ret = np.full_like(close, np.nan)
    ret[1:] = np.log(close[1:] / close[:-1])
    sig = rolling_std_causal(ret, w=cfg.vol_w, min_count=max(10, cfg.vol_w // 5))
    warm_vol = sig[:cfg.warmup_bars]
    finite_vol = warm_vol[np.isfinite(warm_vol)]
    reference = float(np.median(finite_vol)) if finite_vol.size else np.nan
    if not np.isfinite(reference) or reference <= 0:
        reference = max(float(np.std(ret[1:cfg.warmup_bars])), 1e-6)
    if cfg.vol_ref_mode == "median":
        sig_ref = np.full_like(sig, reference)
    else:
        sig_ref = ewm_mean_causal(sig, alpha=cfg.vol_ref_ewm_alpha)
        sig_ref = np.where(np.isfinite(sig_ref), sig_ref, reference)

    eps = 1e-12
    ratio = sig / np.maximum(sig_ref, eps)
    g = np.clip(np.power(np.maximum(ratio, eps), cfg.vol_gamma),
                cfg.vol_clip_lo, cfg.vol_clip_hi)
    enter_t = cfg.enter_th * g
    exit_t = np.minimum(cfg.exit_th * g, enter_t)
    k_t = cfg.cusum_k * g if cfg.cusum_scale_with_vol else np.full_like(g, cfg.cusum_k)
    h_t = cfg.cusum_h * g if cfg.cusum_scale_with_vol else np.full_like(g, cfg.cusum_h)
    resid = close - smooth
    resid_sig = rolling_std_causal(resid, w=cfg.resid_w, min_count=max(10, cfg.resid_w // 5))
    z_res = resid / np.maximum(resid_sig, eps)
    regime = cusum_regime_with_neutral_resid_veto(
        z=tstat, enter_t=enter_t, exit_t=exit_t, k_t=k_t, h_t=h_t,
        z_res=z_res, veto_enter_res_z=cfg.resid_z_enter,
        veto_exit_res_z=cfg.resid_z_exit, reset_on_switch=cfg.cusum_reset_on_switch,
    )
    out = data.copy()
    for name, values in (
        ("kalman", kalman), ("smooth", smooth), ("huber_slope", slope),
        ("huber_resid_scale", resid_scale), ("huber_slope_se", slope_se),
        ("huber_tstat", tstat), ("log_return", ret), ("volatility", sig),
        ("vol_reference", sig_ref), ("thr_scale_g", g), ("enter_t", enter_t),
        ("exit_t", exit_t), ("cusum_k_t", k_t), ("cusum_h_t", h_t),
        ("resid", resid), ("resid_std", resid_sig), ("z_res", z_res), ("regime", regime),
    ):
        out[name] = values
    out["price_deviation_kalman_pct"] = 100 * (close - kalman) / kalman
    out["price_deviation_ema_pct"] = 100 * (close - smooth) / smooth
    baseline_column = ("price_deviation_kalman_pct" if cfg.signal_baseline == "kalman"
                       else "price_deviation_ema_pct")
    out["normalized_deviation_pct"] = out[baseline_column]
    out["buy_signal"], out["sell_signal"] = normalized_flags(
        out["normalized_deviation_pct"], cfg.buy_threshold_pct, cfg.sell_threshold_pct,
    )
    out["threshold_zone"] = np.where(out["buy_signal"], -1,
                                      np.where(out["sell_signal"], 1, 0))
    # The last warm-up close may signal a fill at the first evaluation open.
    out.loc[:cfg.warmup_bars - 2, ["buy_signal", "sell_signal"]] = False
    return out, {"noise_std": noise_std, "q": q, "r": r, "warmup_vol_reference": reference}


TRADE_COLUMNS = ['timestamp', 'signal_timestamp', 'action', 'shares', 'fill_price', 'fee', 'cash_flow', 'realized_profit', 'shares_after', 'cash_after', 'signal_deviation_pct']


def backtest(signals, cfg):
    """Execute prior-close signals, keeping individual purchase lots."""
    start, n = cfg.warmup_bars, len(signals)
    if n <= start:
        raise ValueError("No evaluation bars remain after warm-up")
    opens = signals["open"].to_numpy(dtype=float)
    closes = signals["close"].to_numpy(dtype=float)
    times = signals["timestamp"].array
    buys = signals["buy_signal"].to_numpy(dtype=bool)
    sells = signals["sell_signal"].to_numpy(dtype=bool)
    gaps = signals["normalized_deviation_pct"].to_numpy(dtype=float)
    cash, realized, held_cost = cfg.initial_capital, 0.0, 0.0
    shares, next_id = 0, 0
    spent, peak_cash_used, peak_held_cost, fees = 0.0, 0.0, 0.0, 0.0
    lots, trades = [], []
    m = n - start
    records = {name: np.empty(m) for name in
               ("cash", "held_cost", "realized_profit", "unrealized_profit",
                "account_value", "buy_hold_value", "trading_limit")}
    records["shares"] = np.empty(m, dtype=np.int64)

    bh_entry = opens[start] * (1 + cfg.slippage) + cfg.fee_per_share
    bh_shares = int(np.floor(cfg.initial_capital / bh_entry))
    bh_cash = cfg.initial_capital - bh_shares * bh_entry
    pending = "BUY" if buys[start - 1] else None
    eligible_ids = set()

    for i in range(start, n):
        qty, pnl, fee, cash_flow, fill = 0, 0.0, 0.0, 0.0, 0.0
        action = pending
        if pending == "SELL":
            fill = opens[i] * (1 - cfg.slippage)
            net_unit = fill - cfg.fee_per_share
            keep = []
            for lot in lots:
                if lot["id"] in eligible_ids and net_unit > lot["cost_per_share"]:
                    qlot = lot["shares"]
                    qty += qlot
                    pnl += qlot * (net_unit - lot["cost_per_share"])
                    held_cost -= qlot * lot["cost_per_share"]
                else:
                    keep.append(lot)
            lots = keep
            if qty:
                cash_flow = qty * net_unit
                cash += cash_flow
                realized += pnl
                shares -= qty
                fee = qty * cfg.fee_per_share
                if not lots:
                    held_cost = 0.0
        elif pending == "BUY":
            fill = opens[i] * (1 + cfg.slippage)
            unit_cost = fill + cfg.fee_per_share
            limit = cfg.initial_capital + realized if cfg.use_realized_profit_limit else cfg.initial_capital
            budget = max(0.0, min(cash, limit - held_cost))
            qty = min(cfg.chunk_shares, cfg.buy_limit - shares,
                      int(np.floor(budget / unit_cost)))
            if qty > 0:
                cost = qty * unit_cost
                cash_flow = -cost
                cash -= cost
                held_cost += cost
                spent += cost
                shares += qty
                fee = qty * cfg.fee_per_share
                lots.append({"id": next_id, "timestamp": times[i], "shares": qty,
                             "entry_price": fill, "cost_per_share": unit_cost})
                next_id += 1
        if qty > 0:
            fees += fee
            trades.append([times[i], times[i - 1], action, qty, fill, fee,
                           cash_flow, pnl, shares, cash, gaps[i - 1]])
            if cfg.print_trades:
                print(f"{times[i]} {action:4s} {qty:5d} shares @ {fill:.4f} "
                      f"| signal_gap={gaps[i - 1]:+.4f}% | realized={pnl:.4f} "
                      f"| held={shares} | cash={cash:.2f}")

        peak_cash_used = max(peak_cash_used, cfg.initial_capital - cash)
        peak_held_cost = max(peak_held_cost, held_cost)
        unrealized = shares * closes[i] - held_cost
        j = i - start
        for name, value in (
            ("cash", cash), ("shares", shares), ("held_cost", held_cost),
            ("realized_profit", realized), ("unrealized_profit", unrealized),
            ("account_value", cash + shares * closes[i]),
            ("buy_hold_value", bh_cash + bh_shares * closes[i]),
            ("trading_limit", cfg.initial_capital + realized if cfg.use_realized_profit_limit else cfg.initial_capital),
        ):
            records[name][j] = value

        # Current completed close creates an order for the NEXT observed open.
        pending, eligible_ids = None, set()
        if sells[i]:
            expected_net = closes[i] * (1 - cfg.slippage) - cfg.fee_per_share
            eligible_ids = {lot["id"] for lot in lots if expected_net > lot["cost_per_share"]}
            if eligible_ids:
                pending = "SELL"
        elif buys[i]:
            pending = "BUY"

    curve = signals.iloc[start:].copy().reset_index(drop=True)
    for name, values in records.items():
        curve[name] = values
    curve["account_return_pct"] = (curve["account_value"] / cfg.initial_capital - 1) * 100
    curve["buy_hold_return_pct"] = (curve["buy_hold_value"] / cfg.initial_capital - 1) * 100
    ledger = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    last = curve.iloc[-1]
    report = {
        "realized_profit": realized, "unsold_shares": shares, "total_spent": spent,
        "held_cost": held_cost, "peak_cash_used": peak_cash_used,
        "peak_held_cost": peak_held_cost, "unrealized_profit": float(last.unrealized_profit),
        "ending_cash": cash, "account_value": float(last.account_value),
        "account_return_pct": float(last.account_return_pct),
        "buy_hold_shares": bh_shares, "buy_hold_entry_cost": bh_entry,
        "buy_hold_cash": bh_cash, "buy_hold_value": float(last.buy_hold_value),
        "buy_hold_return_pct": float(last.buy_hold_return_pct),
        "fees": fees, "buys": int((ledger.action == "BUY").sum()),
        "sells": int((ledger.action == "SELL").sum()),
        "pending_final_order": pending or "none",
    }
    # Financial identity, share cap, and cash constraint on EVERY bar.
    if not np.allclose(curve.account_value,
                       cfg.initial_capital + curve.realized_profit + curve.unrealized_profit,
                       rtol=1e-10, atol=1e-6):
        raise RuntimeError("Account reconciliation failed")
    if (curve.shares > cfg.buy_limit).any() or (curve.cash < -1e-7).any():
        raise RuntimeError("Share/cash constraint failed")
    return curve, ledger, pd.DataFrame(lots, columns=["id", "timestamp", "shares",
                                                    "entry_price", "cost_per_share"]), report


def summary_text(curve, report, cfg, load_info, calibration):
    p = report
    lines = [
        f"Files loaded: {load_info['files']} | Bars loaded: {load_info['rows']} | Duplicate bars removed: {load_info['duplicates_removed']}",
        f"Evaluation: {curve.timestamp.iloc[0]} through {curve.timestamp.iloc[-1]}",
        f"Warm-up excluded from BOTH accounts: {cfg.warmup_bars} bars",
        f"Initial capital: ${cfg.initial_capital:.2f} | Maximum held shares: {cfg.buy_limit} | Buy chunk: {cfg.chunk_shares}",
        f"Trading baseline: {'raw Kalman' if cfg.signal_baseline == 'kalman' else 'Kalman+EMA'}",
        "Normalized gap (%) = 100 * (close - baseline) / baseline",
        f"BUY one chunk: gap < {cfg.buy_threshold_pct:+g}% | SELL profitable lots: gap > {cfg.sell_threshold_pct:+g}%",
        "Signals checked on EVERY completed close; equality at the thresholds does not trade. Execution: next bar open.",
        "Sell eligibility checked after costs at both the decision close and execution open; other lots remain held.",
        f"Informational diagnostics: Huber slope w={cfg.w} + volatility adaptation + CUSUM + residual veto",
        f"Volatility reference: {cfg.vol_ref_mode} | Warm-up reference: {calibration['warmup_vol_reference']:.10g}",
        f"Kalman q={calibration['q']:.10g}, r={calibration['r']:.10g} | EMA alpha={cfg.alpha:g}",
        f"Slippage: {cfg.slippage:g} per side | Fee/share: ${cfg.fee_per_share:g}",
        "",
    ]
    if cfg.use_realized_profit_limit:
        lines.append(f"Realized-profit trading limit: ${cfg.initial_capital + p['realized_profit']:.2f} (initial capital + net realized profit)")
    else:
        lines.append(f"Fixed trading limit: ${cfg.initial_capital:.2f}")
    lines += [
        f"Shares bought but not sold: {p['unsold_shares']}",
        f"Total spent on all buys: ${p['total_spent']:.2f}",
        f"Cost of shares still held: ${p['held_cost']:.2f}",
        f"Peak cash used: ${p['peak_cash_used']:.2f}",
        f"Profit from sold shares: ${p['realized_profit']:.2f} ({p['realized_profit'] / cfg.initial_capital * 100:.2f}%)",
        f"Gain/loss on unsold shares: ${p['unrealized_profit']:.2f}",
        f"Ending account value: ${p['account_value']:.2f}",
        f"Account return: {p['account_return_pct']:.4f}%",
        f"Buy-and-hold shares: {p['buy_hold_shares']} | Entry cost/share: ${p['buy_hold_entry_cost']:.4f}",
        f"Buy-and-hold leftover cash: ${p['buy_hold_cash']:.2f}",
        f"Buy-and-hold ending account value: ${p['buy_hold_value']:.2f}",
        f"Buy-and-hold same period: {p['buy_hold_return_pct']:+.4f}%",
        f"Strategy minus buy-and-hold: {p['account_return_pct'] - p['buy_hold_return_pct']:+.4f} percentage points",
        "",
        f"Ending cash: ${p['ending_cash']:.2f} | Peak cost of shares held: ${p['peak_held_cost']:.2f}",
        f"Buy orders filled: {p['buys']} | Sell orders filled: {p['sells']} | Strategy fees paid: ${p['fees']:.2f}",
        f"Last-bar order without a subsequent open: {p['pending_final_order']} (not executed)",
        "Peak cash used = maximum initial capital minus available cash; realized profits can fund additional inventory.",
        "Realized profit percentage uses initial capital as its denominator. Total spent includes entry fees.",
        "Every-bar equity includes ALL unsold shares marked at the raw close. No forced final sale.",
        "Buy-and-hold invests full capital in whole shares at the first evaluation open with the same entry costs.",
        "Price-only results: consistently adjusted OHLC required; dividends and taxes are not modeled.",
    ]
    return "\n".join(lines)


def shade_policy(ax, timestamps, zones, alpha=0.2):
    """Shade contiguous runs, avoiding one Matplotlib object for every minute."""
    state = np.asarray(zones, dtype=int)
    ts = pd.DatetimeIndex(timestamps)
    if len(state) < 2:
        return
    starts = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1]
    ends = np.r_[starts[1:], len(state)]
    labelled = set()
    for start, end in zip(starts, ends):
        value = state[start]
        if value == 0:
            continue
        color, name = ("green", "Buy zone") if value == -1 else ("red", "Sell zone")
        ax.axvspan(ts[start], ts[end] if end < len(ts) else ts[-1], color=color,
                   alpha=alpha, linewidth=0, label=name if value not in labelled else None)
        labelled.add(value)


def draw_plots(curve, ledger, output_dir, cfg, show=False, shade_alpha=.2,
               overlay_points=False, plot_resid=False):
    plt.rcParams["agg.path.chunksize"] = 10000
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(curve.timestamp, curve.account_return_pct, color="teal", linewidth=1,
            label=f"Normalized-gap strategy ({curve.account_return_pct.iloc[-1]:+.2f}%)")
    ax.plot(curve.timestamp, curve.buy_hold_return_pct, color="darkorange", linewidth=1,
            label=f"Buy & hold ({curve.buy_hold_return_pct.iloc[-1]:+.2f}%)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=.8)
    ax.set(title="Account return vs buy and hold - every bar", xlabel="Time", ylabel="Return (%)")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "returns_comparison.png", dpi=160)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    shade_policy(ax1, curve.timestamp, curve.threshold_zone, shade_alpha)
    ax1.plot(curve.timestamp, curve.close, color="brown", alpha=.45, linewidth=.8, label="Raw close")
    ax1.plot(curve.timestamp, curve.smooth, color="black", linewidth=1, label="Model Kalman + EMA baseline")
    if cfg.signal_baseline == "kalman":
        ax1.plot(curve.timestamp, curve.kalman, color="tab:orange", linewidth=1,
                 label="Raw Kalman trading baseline")
    if overlay_points:
        selected_baseline = curve.kalman if cfg.signal_baseline == "kalman" else curve.smooth
        for value, color in ((-1, "green"), (1, "red")):
            mask = curve.threshold_zone == value
            ax1.scatter(curve.timestamp[mask], selected_baseline[mask], s=5, color=color, alpha=.6)
    for action, marker, color in (("BUY", "^", "blue"), ("SELL", "v", "darkorange")):
        subset = ledger[ledger.action == action]
        if not subset.empty:
            ax1.scatter(subset.timestamp, subset.fill_price, marker=marker, color=color,
                        s=30, label=f"{action} fills", zorder=3)
    ax1.set(title="Normalized-gap trading zones and actual trade fills", ylabel="Price")
    ax1.legend(loc="best")
    ax1.grid(alpha=.2)
    ax2.step(curve.timestamp, curve.threshold_zone, where="post", color="slategray", linewidth=1)
    ax2.set_yticks([-1, 0, 1], labels=["Buy zone", "Hold", "Sell zone"])
    ax2.set_ylim(-1.3, 1.3)
    ax2.set(xlabel="Time", ylabel="Threshold policy")
    ax2.grid(alpha=.2)
    fig2.tight_layout()
    fig2.savefig(output_dir / "price_and_policy.png", dpi=160)

    fig3, axes = plt.subplots(2 if plot_resid else 1, 1, figsize=(14, 7 if plot_resid else 4),
                              sharex=True, squeeze=False)
    score = axes[0, 0]
    score.plot(curve.timestamp, curve.huber_tstat, color="purple", linewidth=.9, label="Huber slope t-stat")
    score.plot(curve.timestamp, curve.enter_t, color="green", linewidth=.8, label="+entry threshold")
    score.plot(curve.timestamp, -curve.enter_t, color="red", linewidth=.8, label="-entry threshold")
    score.plot(curve.timestamp, curve.exit_t, color="gray", linestyle="--", linewidth=.7, label="+/- exit threshold")
    score.plot(curve.timestamp, -curve.exit_t, color="gray", linestyle="--", linewidth=.7)
    score.axhline(0, color="black", linewidth=.6)
    score.set(title="Informational regime diagnostics (trading uses normalized-gap thresholds)", ylabel="t-stat")
    score.legend(loc="best")
    score.grid(alpha=.2)
    if plot_resid:
        res = axes[1, 0]
        res.plot(curve.timestamp, curve.z_res, color="steelblue", linewidth=.9, label="Residual z-score")
        for sign in (-1, 1):
            res.axhline(sign * cfg.resid_z_enter, color="red", linewidth=.8,
                        label="Entry veto threshold" if sign == 1 else None)
            res.axhline(sign * cfg.resid_z_exit, color="gray", linestyle="--", linewidth=.8,
                        label="Exit veto threshold" if sign == 1 else None)
        res.axhline(0, color="black", linewidth=.6)
        res.set(ylabel="Residual z")
        res.legend()
        res.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Time")
    fig3.tight_layout()
    fig3.savefig(output_dir / "regime_diagnostics.png", dpi=160)

    fig4, (raw_diff_ax, ema_diff_ax) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, sharey=True,
    )
    # Relative gaps are comparable across different price levels.
    # Positive input prices keep both Kalman and Kalman+EMA estimates positive.
    price_deviation_kalman_pct = curve["price_deviation_kalman_pct"]
    price_deviation_ema_pct = curve["price_deviation_ema_pct"]
    raw_diff_ax.plot(curve.timestamp, price_deviation_kalman_pct, color="tab:orange",
                     linewidth=1, label="Close deviation from Kalman (%)")
    raw_diff_ax.set_title("Actual closing price: percentage deviation from raw Kalman estimate")
    ema_diff_ax.plot(curve.timestamp, price_deviation_ema_pct, color="purple",
                     linewidth=1, label="Close deviation from Kalman+EMA (%)")
    ema_diff_ax.set_title("Actual closing price: percentage deviation from Kalman+EMA baseline")
    trading_ax = raw_diff_ax if cfg.signal_baseline == "kalman" else ema_diff_ax
    trading_ax.set_title(trading_ax.get_title() + " (trading signal)")
    trading_ax.axhline(cfg.buy_threshold_pct, color="green", linestyle=":", linewidth=1,
                      label=f"Buy below {cfg.buy_threshold_pct:+g}%")
    trading_ax.axhline(cfg.sell_threshold_pct, color="red", linestyle=":", linewidth=1,
                      label=f"Sell profitable above {cfg.sell_threshold_pct:+g}%")
    for difference_ax in (raw_diff_ax, ema_diff_ax):
        difference_ax.axhline(0, color="black", linestyle="--", linewidth=.8,
                              label="Price equals estimate")
        difference_ax.set_ylabel("Deviation (%)")
        difference_ax.grid(alpha=.2)
        difference_ax.legend(loc="upper right")
    ema_diff_ax.set_xlabel("Time")
    fig4.tight_layout()
    fig4.savefig(output_dir / "price_minus_kalman_normalized.png", dpi=160)

    if show:
        plt.show()
    for figure in (fig, fig2, fig3, fig4):
        plt.close(figure)


def main():
    defaults = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", "--folder", default="/home/fariborz/Downloads/live/test_stock/XLV",
                    help="Folder containing chronological daily OHLC CSVs for one instrument")
    ap.add_argument("--pattern", default="*.csv", help="Filename filter, e.g. TSLA_*_1min.csv")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--timezone", default="America/New_York", help="Timezone for naive input timestamps")
    ap.add_argument("--initial_capital", type=float, default=defaults.initial_capital)
    ap.add_argument("--buy_limit", type=int, default=defaults.buy_limit, help="Maximum TOTAL held shares")
    ap.add_argument("--chunk_shares", type=int, default=defaults.chunk_shares, help="Shares per qualifying buy bar")
    ap.add_argument("--buy_threshold_pct", type=float, default=defaults.buy_threshold_pct,
                    help="Buy when normalized gap is strictly below this percent (default -0.15)")
    ap.add_argument("--sell_threshold_pct", type=float, default=defaults.sell_threshold_pct,
                    help="Sell profitable lots when gap is strictly above this percent (default 0.15)")
    ap.add_argument("--signal_baseline", choices=("kalman", "kalman_ema"), default=defaults.signal_baseline,
                    help="Baseline for trading: kalman_ema (default, green curve) or kalman (raw filter)")
    ap.add_argument("--warmup_bars", type=int, default=defaults.warmup_bars,
                    help="Calibration bars before both accounts begin (default 200)")
    ap.add_argument("--slippage", type=float, default=defaults.slippage, help="Fraction per side, e.g. 0.0002")
    ap.add_argument("--fee_per_share", type=float, default=defaults.fee_per_share)
    ap.add_argument("--use_realized_profit_limit", action=argparse.BooleanOptionalAction,
                    default=defaults.use_realized_profit_limit,
                    help="Reinvest realized profits (on); --no-use_realized_profit_limit fixes the inventory-cost limit")
    ap.add_argument("--print_trades", action="store_true")
    model = ap.add_argument_group("Filter and informational regime diagnostic settings")
    for name in ("alpha", "q_scale", "r_scale", "enter_th", "exit_th", "eps_slope",
                 "huber_delta_mult", "huber_tol", "vol_gamma", "vol_clip_lo", "vol_clip_hi",
                 "vol_ref_ewm_alpha", "cusum_k", "cusum_h", "resid_z_enter", "resid_z_exit"):
        model.add_argument("--" + name, type=float, default=getattr(defaults, name),
                           help=f"Default: {getattr(defaults, name):g}")
    for name in ("w", "huber_max_iter", "vol_w", "resid_w"):
        model.add_argument("--" + name, type=int, default=getattr(defaults, name),
                           help=f"Default: {getattr(defaults, name)}")
    model.add_argument("--vol_ref_mode", choices=("median", "ewm"), default=defaults.vol_ref_mode,
                       help="median: frozen warm-up median; ewm: causal moving reference")
    model.add_argument("--cusum_scale_with_vol", action="store_true")
    model.add_argument("--cusum_reset_on_switch", action="store_true")
    ap.add_argument("--out_dir", default="normalized_threshold_results")
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                    help="Display saved plots (default on); --no-show runs without windows")
    ap.add_argument("--shade_alpha", type=float, default=.2)
    ap.add_argument("--overlay_points", action="store_true")
    ap.add_argument("--plot_resid", action="store_true")
    args = ap.parse_args()
    cfg = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    cfg.validate()
    if not np.isfinite(args.shade_alpha) or not 0 <= args.shade_alpha <= 1:
        ap.error("shade_alpha must be between 0 and 1")
    output = Path(args.out_dir).expanduser().resolve()
    data, load_info = load_folder(args.data_dir, args.pattern, args.timezone, args.recursive, output)
    print(f"Loaded {load_info['files']} files and {len(data)} chronological bars", flush=True)
    print("Computing Kalman filters, normalized gaps and diagnostic curves...", flush=True)
    signals, calibration = make_signals(data, cfg)
    print("Executing normalized-gap threshold trades...", flush=True)
    curve, ledger, open_lots, report = backtest(signals, cfg)
    summary = summary_text(curve, report, cfg, load_info, calibration)
    output.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output / "threshold_signals.csv", index=False)
    curve.to_csv(output / "equity_curve.csv", index=False)
    ledger.to_csv(output / "trades.csv", index=False)
    open_lots.to_csv(output / "unsold_lots.csv", index=False)
    (output / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary, flush=True)
    draw_plots(curve, ledger, output, cfg, args.show, args.shade_alpha,
               args.overlay_points, args.plot_resid)
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
