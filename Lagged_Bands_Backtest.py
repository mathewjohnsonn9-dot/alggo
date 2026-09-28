from statistics import pstdev
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/fariborz/Downloads/live/test_stock/AAPL/AAPL_2025-07-07_1min.csv"


def main():
    data = pd.read_csv(csv_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        raise ValueError("The CSV contains no prices")

    initial_capital = 10_000.0
    cash = initial_capital
    buy_limit = 100
    chunk_shares = 25
    window = 25
    k1, k2 = 2, 2
    slippage = 0.0000  # Zero slippage; use 0.0002 for 0.02% per side
    fee_per_share = 0.0
    peak_cash_used = 0.0
    prices_so_far = []
    lower_values = []
    upper_values = []
    global_upper_values = []
    day_prices = []
    current_day = None
    bought_at = []  # Actual cost paid for each share still held
    total_money_spent = 0.0
    profit = 0.0
    cost_of_sold = 0.0
    pending = None

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
                print(f"{bar.timestamp} Sold {len(profitable_shares)} shares at ${sell_price:.4f}")
        pending = None

        close_price = float(bar.close)
        prices_so_far.append(close_price)

        # Global statistics use only this day's closes through the current bar.
        if bar.timestamp.date() != current_day:
            current_day = bar.timestamp.date()
            day_prices = []
        day_prices.append(close_price)
        global_mean = sum(day_prices) / len(day_prices)
        global_std = pstdev(day_prices)
        global_upper = global_mean + global_std
        global_upper_values.append(global_upper)

        if len(prices_so_far) < 2 * window:
            lower_values.append(float("nan"))
            upper_values.append(float("nan"))
            continue  # Need 50 prices for a 25-price window lagged by 25

        window_prices = prices_so_far[-2 * window:-window]
        mean = sum(window_prices) / len(window_prices)
        std = pstdev(window_prices)
        lower = mean - k1 * std
        upper = mean + k2 * std
        lower_values.append(lower)
        upper_values.append(upper)

        # This signal can only execute at the NEXT row's open.
        if (
            close_price <= lower
            and close_price < global_upper
            and len(bought_at) < buy_limit
            and peak_cash_used <= initial_capital
        ):
            pending = "BUY"
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
        print(f"Profit from sold shares: ${profit:.2f} ({100 * profit / cost_of_sold:.2f}%)")
    else:
        print("Profit from sold shares: $0.00 (no sales yet)")
    # print(f"Gain/loss on unsold shares: ${open_pnl:.2f}")
    # print(f"Ending account value: ${ending_value:.2f}")
    # print(f"Account return: {100 * (ending_value / initial_capital - 1):.4f}%")

    plt.plot(data["timestamp"], prices_so_far, label="Close price")
    plt.plot(data["timestamp"], lower_values, label="Lower bound")
    plt.plot(data["timestamp"], upper_values, label="Upper bound")
    plt.plot(data["timestamp"], global_upper_values, color="purple", linestyle="--", label="Same day: mean + std")
    plt.xlabel("Time")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == "__main__":
    main()
