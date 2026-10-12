**Continuous minute beta: NVDA on VOO**

This script pairs daily files from two folders, aligns their timestamps, and plots a single beta timeline from the first paired file to the last. It keeps all estimator state across days. Days present in only one folder are reported and skipped.

The model is:

`NVDA log return = alpha + beta × VOO log return + residual`

Beta is a relationship between returns, not `NVDA price / VOO price`. A beta near 2 means an approximately twofold modeled return response to VOO. It does not mean NVDA costs twice as much, nor does it promise a reversal or a profitable trade.

**Run**

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt

python continuous_beta.py \
  --x_dir /home/fariborz/Downloads/live/test_stock/VOO \
  --y_dir /home/fariborz/Downloads/live/test_stock/NVDA \
  --ticker_x VOO \
  --ticker_y NVDA \
  --out_dir beta_results \
  --show
```

Replace those folder paths with the actual locations of your CSVs. Alternatively:

```bash
python continuous_beta.py \
  --data_root /path/to/train_stock \
  --ticker_x VOO --ticker_y NVDA \
  --out_dir beta_results
```

Expected filenames include `VOO_2025-07-25_1min.csv` and `NVDA_2025-07-25_1min.csv`. Dates written as `20250725` are also supported. Both filenames need the same day. File order and row order do not matter; the script sorts them. Files without a single recognizable date are listed in the summary and ignored. Duplicate files for one day raise an error; select a pattern such as `--x_glob '*_1min.csv' --y_glob '*_1min.csv'` rather than silently mixing intervals.

CSV columns normally are `timestamp,close`. Common alternative names are recognized. Override them with `--ts_col` and `--close_col` if needed. Invalid/nonpositive prices are removed; duplicate timestamps keep their last row. No missing prices or returns are invented.

**Timezones and gaps**

Timezone-aware timestamps preserve their offsets. Naive timestamps default to UTC. If your naive CSV times are Eastern, use:

```bash
--input_tz America/New_York
```

To restrict both assets to 09:30–15:59 Eastern, add `--session rth`. The default is `--session all`, using all supplied bars. Plot labels use Eastern time unless `--market_tz` is changed.

By default, only returns between consecutive timestamps approximately one minute apart update beta. The first price of a day after an overnight gap anchors the next return while retaining the previous filter state. Missing intraday minutes and skipped unpaired days similarly leave the model intact and skip the spanning return. Thus overnight price moves are not mislabeled as one-minute returns. You can explicitly include multi-minute/overnight returns with `--gap_policy include`; their durations then differ from the minute observations.

**Three lines on the same timeline**

| Line | Meaning |
|---|---|
| Kalman, selected by default | An evolving alpha/beta estimate. Each available return updates the previous state; large unexplained returns are downweighted. |
| EWMA | Covariance/variance estimate with an exponentially decaying history. Default half-life is 240 usable observations. This is an EWMA estimator, not a fitted DCC-GARCH model. |
| Since-start OLS | A full-history regression beta. Every usable return since the first day has equal weight. It is a useful anchor, although old observations eventually make it slow to respond. |

Change the primary method with `--method ewma` or `--method expanding`. Use `--no-compare` to hide the other two lines. Kalman noise estimates use only already observed data. The code never fits the entire evaluation day to initialize earlier observations, never smooths backward, and never resets at a day boundary.

The first 120 usable returns are marked as initial estimates (`beta_ready=0`), but they are still written and the selected estimate is plotted as a dotted line. The first price has no earlier return and therefore no estimated beta. A nearly constant driver also produces `beta_ready=0`. The count is cumulative across paired days; it does not require 120 new observations every morning. This is an availability/variation guard, not proof of statistical significance. Kalman shaded bands are approximate model uncertainty bands (±1.96 model standard deviations), not guaranteed coverage intervals.

**Does beta change each minute?**

The CSV contains both the exact estimated movement and two useful flags:

| Column | Meaning |
|---|---|
| `beta` | Selected estimate using data available through this timestamp. |
| `beta_delta` | `beta[t] − beta[t−1]`. |
| `beta_moved_this_minute` | 1 if the estimate is ready, the return was usable, and the absolute one-bar movement is at least 0.01 by default. |
| `beta_changed` | 1 when beta has stayed at least 0.10 away from the last confirmed level for three consecutive usable bars, in the same direction. This catches accumulated drift as well as larger jumps. |
| `beta_change_direction` | +1 for a confirmed increase, −1 for a decrease, 0 otherwise. |
| `beta_reference_before` / `beta_vs_reference` | The previous confirmed level and current deviation before any new confirmation. |
| `beta_ready` | Initial-data and driver-variation guard. |
| `observation_used` / `skip_reason` | Whether this row updated the estimators. |
| `observations_so_far` | Continuous usable-return count across the entire folder timeline. |
| `beta_kalman_before_update` | Kalman beta available before observing the current return. The post-update beta becomes available after the current close. |
| `hedge_x_shares_per_y_share` | `beta × close_y / close_x`: the price-adjusted return-beta share multiplier. For a long Y position, the corresponding signed X hedge is the negative of this multiplier. This is a model-implied quantity, not a trade recommendation. |

Thresholds are engineering settings for the plotted estimate, not a statistical test of whether the underlying market relationship changed. They need adjustment to the data and the desired responsiveness.

Example controls:

```bash
--change_threshold 0.10 --confirm_bars 3 --minute_change_threshold 0.01
```

To print every matched minute while processing, add `--print_every 1`. Every minute is written regardless of this printing setting. No row is fabricated when either asset has no corresponding bar.

**Make it steadier or more responsive**

- Steadier Kalman beta: reduce `--kalman_q_beta` from its default `0.00001` to `0.000001`.
- Faster Kalman beta: increase `--kalman_q_beta` to `0.0001`.
- Steadier EWMA: increase `--ewma_half_life` from `240` to `780`.
- Faster EWMA: reduce `--ewma_half_life` to `60`.
- Fewer confirmed-change markers: increase `--change_threshold` or `--confirm_bars`.

Steadier estimates respond more slowly to genuine changes. No smoothing setting eliminates that tradeoff. These defaults are starting values rather than values selected on your full historical dataset.

**Outputs**

| File | Contents |
|---|---|
| `beta_timeline.png` | Normalized prices, continuous beta estimates, confirmed-change markers, and one-bar beta movement. |
| `beta_minutes.csv` | All matched timestamps and minute-level estimates/flags, including initial/gap rows. |
| `beta_changes.csv` | Only rows confirming a persistent beta change. |
| `paired_days.csv` | Files used/skipped, alignment counts, removed rows, and usable returns by day. |
| `summary.json` | Date range, counts, final estimates, parameters, and undated files ignored. |

For years of data, processing and CSV writing occur one day at a time. Only plot samples are retained in memory, up to approximately 100,000 points by default. The estimator and minute CSV always use every usable observation. To plot every point use `--max_plot_points 0`; this requires more memory. Use `--gzip` for a compressed minute CSV.

The calendar axis spans the actual first and last timestamps. Use `--x_axis trading` to compress nights and missing periods. `--show` opens Matplotlib with its zoom/pan tools. For an additional self-contained zoomable HTML chart:

```bash
python -m pip install plotly
python continuous_beta.py --x_dir /path/VOO --y_dir /path/NVDA --html
```

Historical ranges can be restricted with `--start 2024-01-01 --end 2025-12-31`. Choose consistently adjusted close series if your source offers both raw and adjusted prices; this program does not process corporate actions itself.

**Verification**

The included unittest checks cover exact agreement with batch OLS/EWMA, invariance of earlier output when future data changes, state continuity across overnight gaps and unmatched days, missing-minute handling, constant-driver protection, known synthetic beta changes, robust outlier weighting, timezone/epoch equivalence, and change persistence.

```bash
python -m unittest discover -s tests -v
```

The `example/` folder contains a one-day demonstration reconstructed from the VOO/NVDA price columns in your uploaded `long_only_beta_runs/run1/live_state.csv` (July 25, 2025). Its naive timestamps were interpreted as UTC for the demonstration. It is not the requested full multi-day folder run; use your folders to produce that timeline.

You can reproduce the demonstration with:

```bash
python continuous_beta.py --data_root example/input --out_dir example_output
```

For background on expanding regression through a Kalman filter, see the [statsmodels RecursiveLS documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.recursive_ls.RecursiveLS.html). Full DCC-GARCH has a different covariance-estimation structure; see [Engle and Sheppard's paper](https://www.nber.org/papers/w8554). The estimator implementations here use NumPy/Pandas and do not require statsmodels, PyTorch, or GARCH fitting.
