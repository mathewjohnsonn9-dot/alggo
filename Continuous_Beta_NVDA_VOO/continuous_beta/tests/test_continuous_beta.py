import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "continuous_beta.py"
spec = importlib.util.spec_from_file_location("continuous_beta_under_test", MODULE)
beta = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = beta
spec.loader.exec_module(beta)


def arguments(**updates):
    args = beta.build_parser().parse_args([])
    for key, value in updates.items():
        setattr(args, key, value)
    return args


class EstimationTests(unittest.TestCase):
    def test_expanding_matches_batch_ols_including_intercept(self):
        rng = np.random.default_rng(19)
        x = rng.normal(1.1, 3, 500)
        y = .4 + 1.75 * x + rng.normal(0, .8, 500)
        online = beta.Moments()
        for xx, yy in zip(x, y):
            online.update(float(xx), float(yy))
        reference = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1]
        self.assertAlmostEqual(online.beta(.01), reference, places=12)

    def test_ewma_matches_direct_exponentially_weighted_regression(self):
        rng = np.random.default_rng(29)
        x, y = rng.normal(size=(2, 400))
        decay = math.exp(-math.log(2) / 80)
        online = beta.Moments(decay)
        for xx, yy in zip(x, y):
            online.update(float(xx), float(yy))
        weights = decay ** np.arange(len(x) - 1, -1, -1)
        mx, my = np.average(x, weights=weights), np.average(y, weights=weights)
        reference = np.sum(weights * (x - mx) * (y - my)) / np.sum(weights * (x - mx) ** 2)
        self.assertAlmostEqual(online.beta(.01), reference, places=12)

    def test_future_changes_cannot_rewrite_previous_beta_or_flags(self):
        rng = np.random.default_rng(42)
        x, y = rng.normal(0, .0003, size=(2, 600))
        y = 1.8 * x + .3 * y
        def replay(y_returns):
            engine = beta.BetaEngine(arguments(min_observations=20))
            xx, yy = 500 * np.exp(np.cumsum(x)), 150 * np.exp(np.cumsum(y_returns))
            dates = pd.date_range("2025-01-02 14:30", periods=len(x), freq="min", tz="UTC")
            return pd.DataFrame([engine.step(ts, float(px), float(py), "2025-01-02") for ts, px, py in zip(dates, xx, yy)])
        altered = y.copy()
        altered[300:] *= -10
        a, b = replay(y), replay(altered)
        pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300], check_exact=True)

    def test_constant_driver_does_not_claim_a_ready_beta(self):
        engine = beta.BetaEngine(arguments(min_observations=5))
        dates = pd.date_range("2025-01-02 14:30", periods=30, freq="min", tz="UTC")
        rows = [engine.step(ts, 500., 150 + i * .1, "2025-01-02") for i, ts in enumerate(dates)]
        self.assertFalse(any(r["beta_ready"] for r in rows))
        self.assertTrue(math.isnan(rows[-1]["beta_ewma"]))
        self.assertTrue(math.isnan(rows[-1]["beta_expanding"]))

    def test_kalman_tracks_known_change_and_keeps_covariance_positive(self):
        rng = np.random.default_rng(107)
        model = beta.KalmanBeta(arguments())
        observed = []
        for i in range(1800):
            true_beta = 1.0 if i < 900 else 2.0
            x = float(rng.normal(0, 3))
            y = .2 + true_beta * x + float(rng.normal(0, .7))
            model.update(x, y)
            observed.append(model.beta)
            self.assertGreaterEqual(model.p00 * model.p11 - model.p01 ** 2, -1e-10)
            self.assertTrue(math.isfinite(model.beta))
        self.assertLess(abs(float(np.median(observed[700:900])) - 1), .12)
        self.assertLess(abs(float(np.median(observed[1600:1800])) - 2), .12)

    def test_huber_downweights_large_unexplained_return(self):
        rng = np.random.default_rng(5)
        model = beta.KalmanBeta(arguments())
        for _ in range(400):
            x = float(rng.normal(0, 3))
            model.update(x, 1.4 * x + float(rng.normal(0, .5)))
        model.update(.1, 1000.)
        self.assertLess(model.last_weight, .1)
        self.assertTrue(math.isfinite(model.beta))


class TimelineTests(unittest.TestCase):
    def test_skip_overnight_carries_filter_and_observation_count(self):
        engine = beta.BetaEngine(arguments(min_observations=2))
        times = pd.date_range("2025-01-02 14:30", periods=5, freq="min", tz="UTC")
        for i, ts in enumerate(times):
            prior = engine.step(ts, 500 + i, 150 + 2 * i, "2025-01-02")
        next_day = engine.step(pd.Timestamp("2025-01-03 14:30", tz="UTC"), 600., 200., "2025-01-03")
        self.assertEqual(next_day["observation_used"], 0)
        self.assertEqual(next_day["observations_so_far"], prior["observations_so_far"])
        self.assertEqual(next_day["beta_kalman"], prior["beta_kalman"])
        following = engine.step(pd.Timestamp("2025-01-03 14:31", tz="UTC"), 600.3, 200.2, "2025-01-03")
        self.assertEqual(following["observations_so_far"], prior["observations_so_far"] + 1)
        self.assertAlmostEqual(following["return_x_log"], math.log(600.3 / 600.))

    def test_confirmation_and_gap_reset(self):
        tracker = beta.ChangeTracker(.1, 3)
        self.assertFalse(tracker.update(1., True, True)[0])
        self.assertFalse(tracker.update(1.2, True, True)[0])
        self.assertFalse(tracker.update(1.21, True, True)[0])
        self.assertFalse(tracker.update(1.21, True, False)[0])
        self.assertFalse(tracker.update(1.22, True, True)[0])
        self.assertFalse(tracker.update(1.23, True, True)[0])
        self.assertTrue(tracker.update(1.24, True, True)[0])
        self.assertEqual(tracker.events, 1)

    def test_timezone_and_epoch_parsing_agree(self):
        utc = beta.parse_timestamps(pd.Series(["2025-07-25T13:30:00Z"]), "America/New_York")
        local = beta.parse_timestamps(pd.Series(["2025-07-25 09:30:00"]), "America/New_York")
        epoch = beta.parse_timestamps(pd.Series([1753450200000]), "UTC")
        self.assertEqual(utc.iloc[0], local.iloc[0])
        self.assertEqual(utc.iloc[0], epoch.iloc[0])

    def test_duplicate_day_requires_explicit_file_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "VOO_2025-01-02_1min.csv").touch()
            (root / "VOO_2025-01-02_5min.csv").touch()
            with self.assertRaisesRegex(ValueError, "Multiple files"):
                beta.index_day_files(root, "*.csv", None, None)
            matches, _ = beta.index_day_files(root, "*_1min.csv", None, None)
            self.assertEqual(len(matches), 1)

    def test_missing_epoch_row_does_not_invalidate_other_timestamps(self):
        times = beta.parse_timestamps(pd.Series([1753450200000, np.nan, 1753450260000]), "UTC")
        self.assertEqual(times.notna().sum(), 2)
        self.assertEqual((times.iloc[2] - times.iloc[0]).total_seconds(), 60)

    def test_full_folder_run_skips_unpaired_day_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xdir, ydir = root / "VOO", root / "NVDA"
            xdir.mkdir(); ydir.mkdir()
            rng = np.random.default_rng(71)
            for day in ["2025-01-02", "2025-01-03", "2025-01-06"]:
                ts = pd.date_range(day + " 14:30", periods=150, freq="min", tz="UTC")
                rx = rng.normal(0, .0002, len(ts))
                ry = 1.7 * rx + rng.normal(0, .00005, len(ts))
                x = pd.DataFrame({"timestamp": ts, "close": 500 * np.exp(np.cumsum(rx))})
                y = pd.DataFrame({"timestamp": ts, "close": 150 * np.exp(np.cumsum(ry))})
                x.to_csv(xdir / f"VOO_{day}_1min.csv", index=False)
                if day != "2025-01-03":
                    if day == "2025-01-06":
                        y = y.drop(index=80)
                    y.to_csv(ydir / f"NVDA_{day}_1min.csv", index=False)
            args = arguments(x_dir=xdir, y_dir=ydir, out_dir=root / "output", no_plot=True, quiet=True)
            summary = beta.run(args)
            result = pd.read_csv(root / "output/beta_minutes.csv")
            self.assertEqual(summary["paired_file_days"], 2)
            self.assertEqual(summary["unpaired_days_skipped"], 1)
            self.assertEqual(summary["matched_bars"], 299)
            self.assertEqual(summary["usable_returns"], 296)
            first_day_last = result.loc[result.paired_day == "2025-01-02"].iloc[-1]
            second_day_first = result.loc[result.paired_day == "2025-01-06"].iloc[0]
            self.assertEqual(first_day_last.beta_kalman, second_day_first.beta_kalman)
            self.assertEqual(second_day_first.observations_so_far, 149)
            self.assertEqual(result.iloc[-1].observations_so_far, 296)
            self.assertTrue(result.timestamp.is_monotonic_increasing)


class RollingTests(unittest.TestCase):
    def test_each_window_matches_direct_zero_intercept_regression(self):
        rng = np.random.default_rng(302)
        x = rng.normal(1.0, 3, 100)
        y = .2 + 1.7 * x + rng.normal(0, .8, 100)
        window = 12
        model = beta.RollingBeta(arguments(window=window, fit_intercept=False))
        times = pd.date_range("2025-01-02 14:30", periods=len(x), freq="min", tz="UTC")
        for i, (ts, xx, yy) in enumerate(zip(times, x, y)):
            model.update(ts, float(xx), float(yy))
            if i < window - 1:
                self.assertTrue(math.isnan(model.beta))
            else:
                xw, yw = x[i - window + 1:i + 1], y[i - window + 1:i + 1]
                expected = float(xw @ yw / (xw @ xw))
                self.assertAlmostEqual(model.beta, expected, places=11)
                self.assertEqual(model.alpha, 0.0)
                self.assertAlmostEqual(model.rmse, float(np.sqrt(np.mean((yw - expected * xw) ** 2))), places=10)
                self.assertEqual(model.start, times[i - window + 1].isoformat())
                self.assertEqual(model.end, ts.isoformat())

    def test_optional_intercept_matches_direct_ols(self):
        rng = np.random.default_rng(303)
        x = rng.normal(2.0, 3, 90)
        y = 1.1 + 1.9 * x + rng.normal(0, .2, 90)
        window = 20
        model = beta.RollingBeta(arguments(window=window, fit_intercept=True))
        for i, (xx, yy) in enumerate(zip(x, y)):
            model.update(pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(minutes=i), float(xx), float(yy))
            if i >= window - 1:
                xw, yw = x[i - window + 1:i + 1], y[i - window + 1:i + 1]
                intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(window), xw]), yw, rcond=None)[0]
                self.assertAlmostEqual(model.beta, slope, places=11)
                self.assertAlmostEqual(model.alpha, intercept, places=11)

    def test_old_samples_are_removed_exactly(self):
        model = beta.RollingBeta(arguments(window=5))
        x = [1., 2., 3., 4., 5.] * 2
        slopes = [1.] * 5 + [3.] * 5
        for i, (xx, slope) in enumerate(zip(x, slopes)):
            model.update(pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(minutes=i), xx, slope * xx)
        self.assertEqual(len(model.samples), 5)
        self.assertAlmostEqual(model.beta, 3.)
        self.assertAlmostEqual(model.rmse, 0.)

    def test_rolling_window_survives_overnight_without_new_gap_sample(self):
        engine = beta.BetaEngine(arguments(window=4, method="rolling"))
        times = pd.date_range("2025-01-02 14:30", periods=5, freq="min", tz="UTC")
        for i, ts in enumerate(times):
            last = engine.step(ts, 500 + i * i, 150 + i * i, "2025-01-02")
        self.assertEqual(last["rolling_window_observations"], 4)
        self.assertEqual(last["beta_ready"], 1)
        gap = engine.step(pd.Timestamp("2025-01-03 14:30", tz="UTC"), 600., 200., "2025-01-03")
        self.assertEqual(gap["rolling_window_observations"], 4)
        self.assertEqual(gap["beta"], last["beta"])
        self.assertEqual(gap["rolling_window_start"], last["rolling_window_start"])
        after = engine.step(pd.Timestamp("2025-01-03 14:31", tz="UTC"), 600.3, 200.2, "2025-01-03")
        self.assertEqual(after["rolling_window_start"], times[2].isoformat())
        self.assertEqual(after["rolling_window_end"], "2025-01-03T14:31:00+00:00")

    def test_huber_window_downweights_a_single_large_outlier(self):
        rng = np.random.default_rng(305)
        x = rng.normal(0, 3, 80)
        y = 1.6 * x + rng.normal(0, .08, 80)
        y[np.argmax(np.abs(x))] += 200
        ols = beta.RollingBeta(arguments(window=80, rolling_loss="ols"))
        robust = beta.RollingBeta(arguments(window=80, rolling_loss="huber"))
        for i, (xx, yy) in enumerate(zip(x, y)):
            ts = pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(minutes=i)
            ols.update(ts, float(xx), float(yy))
            robust.update(ts, float(xx), float(yy))
        self.assertLess(abs(robust.beta - 1.6), .05)
        self.assertLess(abs(robust.beta - 1.6), abs(ols.beta - 1.6) / 10)

    def test_flat_driver_window_invalidates_previously_ready_estimate(self):
        model = beta.RollingBeta(arguments(window=4))
        for i, xx in enumerate([1., 2., 3., 4., 0., 0., 0., 0.]):
            model.update(pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(minutes=i), xx, 2 * xx)
            if i == 3:
                self.assertAlmostEqual(model.beta, 2.)
        self.assertTrue(math.isnan(model.beta))

    def test_periodic_rebase_preserves_the_current_window_fit(self):
        rng = np.random.default_rng(306)
        model = beta.RollingBeta(arguments(window=40, fit_intercept=True))
        for i in range(10_020):
            xx = float(rng.normal(2, 3))
            yy = 1.2 + 1.5 * xx + float(rng.normal(0, .5))
            model.update(pd.Timestamp("2025-01-02", tz="UTC") + pd.Timedelta(minutes=i), xx, yy)
        data = np.asarray([(v[1], v[2]) for v in model.samples])
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(40), data[:, 0]]), data[:, 1], rcond=None)[0]
        self.assertAlmostEqual(model.beta, slope, places=11)
        self.assertAlmostEqual(model.alpha, intercept, places=11)


if __name__ == "__main__":
    unittest.main()
