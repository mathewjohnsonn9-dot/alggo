from statistics import pstdev
from collections import deque
from math import isfinite
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# List the consecutive daily CSVs for the SAME stock that you want to trade.
# Add or uncomment paths below; timestamps determine the processing order.
csv_paths = [
    "/home/fariborz/Downloads/live/test_stock/VOO/VOO_2025-07-09_1min.csv",
    # "/home/fariborz/Downloads/live/test_stock/VOO/VOO_2025-07-10_1min.csv",
    # "/home/fariborz/Downloads/live/test_stock/VOO/VOO_2025-07-11_1min.csv",
]
previous_days_folder = Path(csv_paths[0]).parent / "previous days"

# "direction_rebound": dip below the threshold, then buy on the first rise.
# "direction_only": the previous D < threshold test; "filtered": band rules.
entry_mode = "direction_rebound"
direction_buy_threshold = -0.5

# In "filtered" mode, set both to False to compare without decline/staging gates.
enable_decline_filter = True
enable_staged_buying = True

# Starting values to test, not optimized trading parameters.
decline_settings = {
    "window": 10,
    "ema_span": 30,
    "ema_lag": 10,
    "warning_threshold": -0.5,
    "resume_threshold": -0.2,
    "confirm_bars": 3,
    "resume_bars": 5,
}


class DeclineDetector:
    def __init__(self, window=20, ema_span=60, ema_lag=10,
                 warning_threshold=-0.5, resume_threshold=-0.2,
                 confirm_bars=3, resume_bars=5):
        if min(window, ema_span, ema_lag, confirm_bars, resume_bars) < 1:
            raise ValueError("Decline detector periods must be positive")
        if not -1 <= warning_threshold < resume_threshold <= 1:
            raise ValueError("Require -1 <= warning threshold < resume threshold <= 1")
        self.window = window
        self.ema_lag = ema_lag
        self.warning_threshold = warning_threshold
        self.resume_threshold = resume_threshold
        self.confirm_bars = confirm_bars
        self.resume_bars = resume_bars
        self.alpha = 2 / (ema_span + 1)
        self.closes = deque(maxlen=window + 1)
        self.ema_values = deque(maxlen=ema_lag + 1)
        self.ema = None
        self.decline_count = 0
        self.stable_count = 0
        self.paused = False

    def update(self, close):
        # Called once per completed bar; never inspects a future close.
        self.closes.append(close)
        self.ema = close if self.ema is None else self.alpha * close + (1 - self.alpha) * self.ema
        self.ema_values.append(self.ema)
        ready = len(self.closes) == self.window + 1 and len(self.ema_values) == self.ema_lag + 1
        direction = float("nan")
        warning = False
        stable = False
        if ready:
            values = list(self.closes)
            total_movement = sum(abs(b - a) for a, b in zip(values, values[1:]))
            direction = (values[-1] - values[0]) / total_movement if total_movement else 0.0
            warning = direction < self.warning_threshold
            falling = self.ema < self.ema_values[0]
            self.decline_count = self.decline_count + 1 if warning and falling else 0
            stable = direction > self.resume_threshold and not falling
        else:
            self.decline_count = 0

        self.stable_count = self.stable_count + 1 if stable else 0
        # Release exactly one chunk at the START of each confirmed stable phase.
        # Continuing stability must not unlock another chunk every five bars.
        release_chunk = self.stable_count == self.resume_bars
        if self.decline_count >= self.confirm_bars:
            self.paused = True
        if self.paused and self.stable_count >= self.resume_bars:
            self.paused = False
        state = "PAUSED" if self.paused else "WARNING" if warning else "NORMAL"
        return {"direction": direction, "ema": self.ema, "state": state,
                "allow_buy": ready and not self.paused, "release_chunk": release_chunk}


class DirectionReboundEntry:
    def __init__(self, threshold):
        self.threshold = threshold
        self.previous_direction = None
        self.armed = False

    def update(self, direction):
        # The first valid value below the threshold also starts a dip.
        if not isfinite(direction):
            return False
        crossed_below = direction < self.threshold and (
            self.previous_direction is None or self.previous_direction >= self.threshold
        )
        if crossed_below:
            self.armed = True
        # Positive first difference: D[t] - D[t-1] > 0.
        # The rise can trigger while D is still below the threshold.
        turned_up = (
            self.armed
            and self.previous_direction is not None
            and direction > self.previous_direction
        )
        self.previous_direction = direction
        if turned_up:
            self.armed = False  # One signal per dip; a new down-crossing rearms it.
        return turned_up


def main():
    if entry_mode not in ("direction_rebound", "direction_only", "filtered"):
        raise ValueError("entry_mode must be 'direction_rebound', 'direction_only', or 'filtered'")
    data = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        raise ValueError("The CSV contains no prices")

    # Put CSVs for the SAME stock in the 'previous days' folder.
    previous_files = sorted(Path(previous_days_folder).glob("*.csv"))
    # if not previous_files:
    #     raise FileNotFoundError(f"No previous-day CSVs found in {previous_days_folder}")
    history = pd.concat(
        [pd.read_csv(path, usecols=["timestamp", "close"]) for path in previous_files]
        + [data[["timestamp", "close"]]],
        ignore_index=True,
    )
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history = history.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    history["day"] = history["timestamp"].dt.date
    first_day = data["timestamp"].iloc[0].date()
    # if history.loc[history["day"] < first_day, "day"].nunique() < 5:
    #     raise ValueError("The 'previous days' folder must contain at least 5 earlier trading days")

    seed_detection = {"state": "NORMAL"}  # Plot fallback when historical seeding is skipped.
    detector = DeclineDetector(**decline_settings)
    seed_dates = sorted(history.loc[history["day"] < first_day, "day"].unique())[-5:]
    for seed_close in history.loc[history["day"].isin(seed_dates), "close"]:
        seed_detection = detector.update(float(seed_close))
    rebound_entry = DirectionReboundEntry(direction_buy_threshold)
    rebound_entry.update(seed_detection.get("direction", float("nan")))
    # Anchor history before the first test day; the selected days form one period.
    previous_prices = history.loc[history["day"].isin(seed_dates), "close"].astype(float).tolist()

    initial_capital = 10_000.0
    cash = initial_capital
    buy_limit = 100
    chunk_shares = 15
    window = 25
    k1, k2 = 2, 2
    slippage = 0.0000  # Zero slippage; use 0.0002 for 0.02% per side
    fee_per_share = 0.0
    peak_cash_used = 0.0
    prices_so_far = []
    lower_values = []
    upper_values = []
    global_upper_values = []
    multi_day_upper_values = []
    ema_values = []
    direction_values = []
    detection_states = []
    day_prices = []
    bought_at = []  # Actual cost paid for each share still held
    total_money_spent = 0.0
    profit = 0.0
    cost_of_sold = 0.0
    pending = None
    buy_times = []
    buy_prices = []
    sell_times = []
    sell_prices = []
    buy_signal_times = []
    buy_signal_directions = []
    # One initial chunk, then one additional chunk per NEW stable phase.
    # This permission carries across days, like cash and held shares.
    chunk_available = True

    for bar in data.itertuples(index=False):
        # The previous bar's CLOSE signal trades at this OPEN.
        if pending == "BUY":
            buy_price = float(bar.open) * (1 + slippage) + fee_per_share
            number_to_buy = min(chunk_shares, buy_limit - len(bought_at), int(cash / buy_price))
            if number_to_buy:
                spent = number_to_buy * buy_price
                cash -= spent
                total_money_spent += spent
                peak_cash_used = max(peak_cash_used, initial_capital - cash)
                bought_at.extend([buy_price] * number_to_buy)
                buy_times.append(bar.timestamp)
                buy_prices.append(buy_price)
                chunk_available = False
                print(f"{bar.timestamp} Bought {number_to_buy} shares at ${buy_price:.4f}")

        elif pending == "SELL":
            sell_price = float(bar.open) * (1 - slippage) - fee_per_share
            profitable_shares = [price for price in bought_at if price < sell_price]
            if profitable_shares:
                for price in profitable_shares:
                    bought_at.remove(price)
                cash += len(profitable_shares) * sell_price
                profit += len(profitable_shares) * sell_price - sum(profitable_shares)
                cost_of_sold += sum(profitable_shares)
                sell_times.append(bar.timestamp)
                sell_prices.append(sell_price)
                if not bought_at:
                    chunk_available = True  # The previous position is fully closed.
                print(f"{bar.timestamp} Sold {len(profitable_shares)} shares at ${sell_price:.4f}")
        pending = None

        close_price = float(bar.close)
        prices_so_far.append(close_price)
        detection = detector.update(close_price)
        # Track every completed bar, including warmup; carry this state across days.
        rebound_signal = rebound_entry.update(detection["direction"])
        if detection["release_chunk"]:
            chunk_available = True
        ema_values.append(detection["ema"])
        direction_values.append(detection["direction"])
        detection_states.append(detection["state"])

        # Treat all selected days as one day; statistics never reset at midnight.
        day_prices.append(close_price)
        global_mean = sum(day_prices) / len(day_prices)
        global_std = pstdev(day_prices)
        global_upper = global_mean + global_std
        global_upper_values.append(global_upper)

        # Five earlier days plus the combined period only through the current bar.
        multi_day_prices = previous_prices + day_prices
        multi_day_mean = sum(multi_day_prices) / len(multi_day_prices)
        multi_day_std = pstdev(multi_day_prices)
        multi_day_upper = multi_day_mean + multi_day_std
        multi_day_upper_values.append(multi_day_upper)

        if len(prices_so_far) < 2 * window:
            lower_values.append(float("nan"))
            upper_values.append(float("nan"))
            continue  # Need 50 prices for a 25-price window lagged by 25

        window_prices = prices_so_far[-window:-1]
        mean = sum(window_prices) / len(window_prices)
        std = pstdev(window_prices)
        lower = mean - k1 * std
        upper = mean + k2 * std
        lower_values.append(lower)
        upper_values.append(upper)

        # This signal can only execute at the NEXT row's open.
        if entry_mode == "direction_rebound":
            # Wait for D to turn up after a dip, rather than buying every low bar.
            # As in the direction-only test, band/pause/staging gates are bypassed.
            buy_signal = (
                rebound_signal
                and len(bought_at) < buy_limit
                and peak_cash_used <= initial_capital
            )
        elif entry_mode == "direction_only":
            # Buy into negative direction, even if the decline detector is paused.
            # The band and staged-buying conditions are deliberately bypassed.
            buy_signal = (
                detection["direction"] < direction_buy_threshold
                and len(bought_at) < buy_limit
                and peak_cash_used <= initial_capital
            )
        else:
            buy_signal = (
                close_price <= lower
                and close_price < global_upper
                and close_price < multi_day_upper
                and (not enable_decline_filter or detection["allow_buy"])
                and (not enable_staged_buying or chunk_available)
                and len(bought_at) < buy_limit
                and peak_cash_used <= initial_capital
            )
        if buy_signal:
            pending = "BUY"
            if entry_mode == "direction_rebound":
                buy_signal_times.append(bar.timestamp)
                buy_signal_directions.append(detection["direction"])
        elif close_price >= upper and bought_at:
            pending = "SELL"

    last_close = prices_so_far[-1]
    open_pnl = sum(last_close - price for price in bought_at)
    ending_value = cash + len(bought_at) * last_close
    print(f"Shares bought but not sold: {len(bought_at)}")
    print(f"Total spent on all buys: ${total_money_spent:.2f}")
    print(f"Cost of shares still held: ${sum(bought_at):.2f}")
    print(f"Peak cash used: ${peak_cash_used:.2f}")
    if cost_of_sold:
        print(f"Profit from sold shares: ${profit:.2f} ({100 * profit / peak_cash_used:.2f}%)")
    else:
        print("Profit from sold shares: $0.00 (no sales yet)")
    # print(f"Gain/loss on unsold shares: ${open_pnl:.2f}")
    # print(f"Ending account value: ${ending_value:.2f}")
    # print(f"Account return: {100 * (ending_value / initial_capital - 1):.4f}%")

    fig, (price_ax, direction_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    price_ax.plot(data["timestamp"], prices_so_far, label="Close price")
    price_ax.plot(data["timestamp"], lower_values, label="Lower bound")
    price_ax.plot(data["timestamp"], upper_values, label="Upper bound")
    price_ax.plot(data["timestamp"], global_upper_values, color="purple", linestyle="--", label="Combined days: mean + std")
    price_ax.plot(data["timestamp"], multi_day_upper_values, color="brown", linestyle=":", label="5 prior days + combined days: mean + std")
    price_ax.plot(data["timestamp"], ema_values, color="black", linestyle="-.", label=f"EMA {decline_settings['ema_span']}", alpha=0.7)
    if buy_times:
        price_ax.scatter(buy_times, buy_prices, color="green", marker="^", s=80, label="Buy", zorder=5)
    if sell_times:
        price_ax.scatter(sell_times, sell_prices, color="red", marker="v", s=80, label="Sell", zorder=5)

    # A close signal controls the NEXT open. Shift the shading by one row
    # so execution markers align with the state active at that bar's open.
    active_states = [seed_detection["state"]] + detection_states[:-1]
    state_colors = {"WARNING": "orange", "PAUSED": "red"}
    labelled_states = set()
    start = 0
    timestamps = data["timestamp"].tolist()
    for end in range(1, len(active_states) + 1):
        if end == len(active_states) or active_states[end] != active_states[start]:
            state = active_states[start]
            if state in state_colors:
                right = timestamps[end] if end < len(timestamps) else timestamps[-1]
                label = ("Decline warning" if state == "WARNING" else
                         "Pause signal (ignored in direction test)" if entry_mode != "filtered" else
                         "Pause signal") if state not in labelled_states else None
                for ax in (price_ax, direction_ax):
                    ax.axvspan(timestamps[start], right, color=state_colors[state], alpha=0.12,
                               label=label if ax is price_ax else None, zorder=0)
                labelled_states.add(state)
            start = end

    direction_ax.plot(data["timestamp"], direction_values, color="navy", label=f"D{decline_settings['window']}")
    if buy_signal_times:
        direction_ax.scatter(buy_signal_times, buy_signal_directions, color="red", s=25,
                             label="Buy signal (close)", zorder=5)
    displayed_threshold = direction_buy_threshold if entry_mode != "filtered" else decline_settings["warning_threshold"]
    threshold_label = ("Dip threshold" if entry_mode == "direction_rebound" else
                       "Buy threshold" if entry_mode == "direction_only" else "Warning threshold")
    direction_ax.axhline(displayed_threshold, color="red", linestyle="--",
                         label=threshold_label)
    direction_ax.axhline(decline_settings["resume_threshold"], color="green", linestyle="--", label="Restart threshold")
    direction_ax.set_ylim(-1.05, 1.05)
    direction_ax.set_ylabel("Direction ratio")
    direction_ax.set_xlabel("Time")
    price_ax.set_ylabel("Price ($)")
    if entry_mode == "direction_rebound":
        price_ax.set_title(
            f"Entry: D{decline_settings['window']} dips below {direction_buy_threshold}, then rises | "
            "Band, pause, and staging gates bypassed"
        )
    elif entry_mode == "direction_only":
        price_ax.set_title(
            f"Entry: D{decline_settings['window']} < {direction_buy_threshold} | "
            "Band, pause, and staging gates bypassed"
        )
    else:
        price_ax.set_title(
            f"Decline filter: {'ON' if enable_decline_filter else 'OFF'} | "
            f"Staged buying: {'ON' if enable_staged_buying else 'OFF'}"
        )
    price_ax.legend(loc="best", fontsize=8, ncol=2)
    direction_ax.legend(loc="best", fontsize=8, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
