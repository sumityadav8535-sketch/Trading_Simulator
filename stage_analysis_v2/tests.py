from django.test import SimpleTestCase, TestCase
from django.urls import reverse

import numpy as np


class SupertrendMathTests(SimpleTestCase):
    def test_first_touch_marks_only_first_tag(self):
        from stage_analysis_v2.services.supertrend_swing import _STPack
        import pandas as pd

        n = 80
        close = np.linspace(100, 160, n)
        idx = pd.bdate_range("2024-01-01", periods=n)
        df = pd.DataFrame({
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000),
        }, index=idx)
        pack = _STPack("TEST", df, 10, 3.0)
        self.assertEqual(len(pack.first_touch), n)
        self.assertTrue(pack.first_touch.dtype == bool or pack.first_touch.dtype == np.bool_)

    def test_uptrend_turns_bullish(self):
        from stage_analysis_v2.services.supertrend_swing import supertrend_np

        n = 80
        close = np.linspace(100, 180, n)
        high = close + 1.0
        low = close - 1.0
        _st, direction = supertrend_np(high, low, close, period=10, multiplier=3.0)
        self.assertGreater(direction[-1], 0)

    def test_downtrend_stays_or_turns_bearish(self):
        from stage_analysis_v2.services.supertrend_swing import supertrend_np

        n = 80
        close = np.linspace(180, 100, n)
        high = close + 1.0
        low = close - 1.0
        _st, direction = supertrend_np(high, low, close, period=10, multiplier=3.0)
        self.assertLessEqual(direction[-1], 0)


class StageV2FormTests(SimpleTestCase):
    def test_backtest_form_includes_smallcap250_universe(self):
        from stage_analysis_v2.forms import StageV2BacktestForm

        values = [value for value, _label in StageV2BacktestForm.UNIVERSE_CHOICES]
        self.assertIn("nifty200", values)
        self.assertIn("nifty_smallcap250", values)


class StrategyCatalogTests(SimpleTestCase):
    def test_normalize_and_packs(self):
        from stage_analysis_v2.services.strategy_catalog import (
            is_supertrend_strategy,
            normalize_strategy,
            st_filter_pack,
        )

        self.assertEqual(normalize_strategy("nope"), "stage_v2")
        self.assertTrue(is_supertrend_strategy("st_pullback_quality"))
        self.assertTrue(is_supertrend_strategy("st_union_minervini"))
        self.assertFalse(is_supertrend_strategy("stage_v2"))
        self.assertEqual(st_filter_pack("st_pullback_trend_rsi"), "trend_rsi")
        self.assertEqual(st_filter_pack("st_pullback_quality"), "quality")

    def test_stage_leftovers_are_replaced_with_st_researched_params(self):
        from stage_analysis_v2.services.strategy_catalog import coerce_supertrend_params

        out = coerce_supertrend_params(
            "st_pullback_quality",
            risk_pct=2,
            max_hold_days=65,
            cooldown_days=40,
            max_pos_pct=100,
        )
        self.assertEqual(out["risk_pct"], "3.0")
        self.assertEqual(out["max_hold_days"], "90")
        self.assertEqual(out["cooldown_days"], "10")
        self.assertEqual(out["max_pos_pct"], "50")

    def test_union_stage_leftovers_use_union_defaults(self):
        from stage_analysis_v2.services.strategy_catalog import coerce_supertrend_params

        out = coerce_supertrend_params(
            "st_union_minervini",
            risk_pct=2,
            max_hold_days=65,
            cooldown_days=40,
            max_pos_pct=100,
        )
        self.assertEqual(out["risk_pct"], "2.0")
        self.assertEqual(out["max_hold_days"], "150")
        self.assertEqual(out["cooldown_days"], "8")
        self.assertEqual(out["max_pos_pct"], "80")

    def test_minervini_template_on_uptrend(self):
        from stage_analysis_v2.services.st_union_swing import is_minervini

        self.assertTrue(is_minervini(200, 180, 160, 140, 135, 100, 210))
        self.assertFalse(is_minervini(200, 180, 160, 140, 145, 100, 210))  # 200 SMA not rising
        self.assertFalse(is_minervini(150, 180, 160, 140, 135, 100, 210))  # below SMA50

    def test_custom_st_params_are_kept(self):
        from stage_analysis_v2.services.strategy_catalog import coerce_supertrend_params

        out = coerce_supertrend_params(
            "st_pullback_quality",
            risk_pct=5,
            max_hold_days=90,
            cooldown_days=10,
            max_pos_pct=50,
        )
        self.assertEqual(out["risk_pct"], "5.0")
        self.assertEqual(out["max_hold_days"], "90")
        self.assertEqual(out["max_pos_pct"], "50.0")

    def test_cup_is_not_supertrend(self):
        from stage_analysis_v2.services.strategy_catalog import (
            is_cup_strategy,
            is_supertrend_strategy,
            normalize_strategy,
        )

        self.assertEqual(normalize_strategy("cup_breakout"), "cup_breakout")
        self.assertTrue(is_cup_strategy("cup_breakout"))
        self.assertFalse(is_supertrend_strategy("cup_breakout"))
        self.assertFalse(is_cup_strategy("stage_v2"))

    def test_cup_stage_leftovers_are_replaced(self):
        from stage_analysis_v2.services.strategy_catalog import coerce_cup_params

        out = coerce_cup_params(
            "cup_breakout",
            risk_pct=2,
            max_hold_days=65,
            cooldown_days=40,
            max_pos_pct=100,
            target_rr=2.5,
        )
        self.assertEqual(out["max_hold_days"], "40")
        self.assertEqual(out["cooldown_days"], "0")
        self.assertEqual(out["max_pos_pct"], "100")
        self.assertEqual(out["risk_pct"], "10.0")


class StopExitReasonTests(SimpleTestCase):
    def _pos(self, *, entry=100.0, stop=90.0, entry_stop=90.0):
        import pandas as pd
        from stage_analysis_v2.services.backtester import _OpenPos

        return _OpenPos(
            symbol="TEST",
            entry_date=pd.Timestamp("2024-01-02"),
            signal_date=pd.Timestamp("2024-01-01"),
            entry_price=entry,
            stop=stop,
            target=125.0,
            qty=10,
            notional=entry * 10,
            quality_score=80,
            rs_rating=80.0,
            weekly_stage=2,
            entry_stop=entry_stop,
        )

    def test_initial_stop_is_stop_loss(self):
        from stage_analysis_v2.services.backtester import stop_exit_reason

        self.assertEqual(stop_exit_reason(self._pos()), "stop_loss")

    def test_raised_stop_is_trail_stop(self):
        from stage_analysis_v2.services.backtester import stop_exit_reason

        pos = self._pos(stop=108.0, entry_stop=90.0)
        self.assertEqual(stop_exit_reason(pos), "trail_stop")

    def test_same_bar_low_does_not_use_new_trail(self):
        from stage_analysis_v2.services.backtester import maybe_raise_stop, stop_exit_reason

        pos = self._pos(entry=100.0, stop=90.0, entry_stop=90.0)
        # Day rallies: low never tags the old 90 stop, close 110, new trail 108.
        # Old bug: raise stop to 108 then low(95) <= 108 → profitable "stop_loss".
        low, close, new_trail = 95.0, 110.0, 108.0
        hit = low <= pos.stop
        self.assertFalse(hit)
        maybe_raise_stop(pos, new_trail, close)
        self.assertEqual(pos.stop, 108.0)
        self.assertEqual(stop_exit_reason(pos), "trail_stop")

        # Next day actually tags the trail that was already in force.
        next_low = 107.0
        self.assertTrue(next_low <= pos.stop)
        self.assertGreater(pos.stop, pos.entry_price)
        self.assertEqual(stop_exit_reason(pos), "trail_stop")

    def test_close_trade_at_trail_is_profitable_with_trail_label(self):
        from stage_analysis_v2.services.backtester import _close_trade, stop_exit_reason
        import pandas as pd

        pos = self._pos(entry=100.0, stop=108.0, entry_stop=90.0)
        trade = _close_trade(
            pos,
            exit_price=pos.stop,
            exit_ts=pd.Timestamp("2024-02-01"),
            exit_reason=stop_exit_reason(pos),
            days_held=10,
        )
        self.assertGreater(trade.pnl, 0)
        self.assertEqual(trade.exit_reason, "trail_stop")
        self.assertNotEqual(trade.exit_reason, "stop_loss")


class BacktestPageTests(TestCase):
    def test_backtest_page_has_strategy_dropdown(self):
        resp = self.client.get(reverse("stage_analysis_v2:backtest"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'name="strategy"')
        self.assertContains(resp, "Supertrend Pullback + Quality")
        self.assertContains(resp, "Stage Analysis 2.0 — Stage 2 weekly")
        self.assertContains(resp, "Cup Breakout Strategy")
        self.assertContains(resp, "Load cup 100% pack")
        self.assertContains(resp, "Load RS 70 pack")
        self.assertContains(resp, 'name="min_rs_rating"')
        self.assertContains(resp, 'name="shared_capital"')
        self.assertContains(resp, "Shared capital")

    def test_book_stats_helper(self):
        from datetime import date
        from stage_analysis_v2.services.backtester import StageV2Trade, _book_stats

        trades = [
            StageV2Trade(
                symbol="AAA", signal_date="2024-01-01", entry_date="2024-01-02",
                exit_date="2024-01-10", entry_price=100, exit_price=110, stop_loss=90,
                target=125, quantity=10, pnl=100, pnl_pct=10, rr_achieved=1.0,
                exit_reason="target", quality_score=80, rs_rating=70, weekly_stage=2,
                days_held=6,
            ),
            StageV2Trade(
                symbol="BBB", signal_date="2024-01-01", entry_date="2024-01-02",
                exit_date="2024-01-05", entry_price=50, exit_price=45, stop_loss=40,
                target=70, quantity=5, pnl=-25, pnl_pct=-10, rr_achieved=-0.5,
                exit_reason="stop_loss", quality_score=60, rs_rating=50, weekly_stage=2,
                days_held=3,
            ),
        ]
        curve = [{"date": "2024-01-01", "equity": 1000}, {"date": "2024-01-10", "equity": 1075}]
        stats = _book_stats(trades, curve, 1000, skipped_cash=3, peak_parallel=2, final_cash=1075)
        self.assertEqual(stats["trades"], 2)
        self.assertEqual(stats["win_rate"], 50.0)
        self.assertEqual(stats["skipped_cash"], 3)
        self.assertEqual(stats["total_return_pct"], 7.5)

    def test_rs70_pack_sets_rs_and_sma150_filters(self):
        resp = self.client.get(reverse("stage_analysis_v2:backtest"))
        self.assertEqual(resp.status_code, 200)
        pack = resp.context["rs70_pack_defaults"]
        self.assertEqual(str(pack["min_rs_rating"]), "70")
        self.assertEqual(pack["ma_condition"], "above")
        self.assertEqual(str(pack["ma_period"]), "150")
        self.assertEqual(pack["ma_type"], "sma")
        self.assertEqual(pack["strategy"], "stage_v2")
        self.assertEqual(pack["universe"], "nifty200")

    def test_results_include_win_loss_buttons_and_trade_log(self):
        from datetime import date
        from django.template.loader import render_to_string
        from stage_analysis_v2.services.backtester import (
            StageV2BacktestResult,
            StageV2Trade,
            fill_performance_metrics,
            trades_as_json,
        )

        def trade(symbol, pnl):
            return StageV2Trade(
                symbol=symbol, signal_date="2024-01-01", entry_date="2024-01-02",
                exit_date="2024-01-10", entry_price=100, exit_price=110 if pnl > 0 else 90,
                stop_loss=90, target=125, quantity=10, pnl=pnl, pnl_pct=pnl / 10,
                rr_achieved=1.0 if pnl > 0 else -1.0,
                exit_reason="target" if pnl > 0 else "stop_loss",
                quality_score=80, rs_rating=70, weekly_stage=2, days_held=6,
            )

        result = StageV2BacktestResult(
            strategy_name="Stage 2.0",
            symbols=["WINNER", "LOSER"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            capital=100000,
            min_quality_score=0,
            market_filter=False,
            trades=[trade("WINNER", 100.0), trade("LOSER", -50.0)],
            total_trades=2,
            stage2_entries=2,
        )
        fill_performance_metrics(result)
        self.assertEqual(result.win_count, 1)
        self.assertEqual(result.loss_count, 1)
        log = trades_as_json(result)
        self.assertEqual([row["outcome"] for row in log], ["win", "loss"])
        self.assertEqual(log[0]["symbol"], "WINNER")
        self.assertEqual(log[1]["symbol"], "LOSER")

        page = self.client.get(reverse("stage_analysis_v2:backtest"))
        html = render_to_string(
            "stage_analysis_v2/backtest.html",
            {
                "form": page.context["form"],
                "bt_result": result,
                "signal_log": [],
                "trade_log": log,
                "charts": {},
                "symbol_list": ["WINNER", "LOSER"],
                "run_requested": True,
                "form_defaults": page.context["form_defaults"],
                "aggressive_300_defaults": page.context["aggressive_300_defaults"],
                "cup_100_defaults": page.context["cup_100_defaults"],
                "rs70_pack_defaults": page.context["rs70_pack_defaults"],
                "strategy_presets": page.context["strategy_presets"],
                "strategy_blurbs": page.context["strategy_blurbs"],
                "start_date_str": page.context["start_date_str"],
                "end_date_str": page.context["end_date_str"],
            },
            request=page.wsgi_request,
        )
        self.assertIn("js-open-wl", html)
        self.assertIn("Wins:", html)
        self.assertIn("Losses:", html)
        self.assertIn('data-outcome="win"', html)
        self.assertIn('data-outcome="loss"', html)
        self.assertIn("backtest-trade-log", html)
        self.assertIn("id=\"wl-review\"", html)
        self.assertIn("WINNER", html)
        self.assertIn("LOSER", html)


def _ohlc_from_close(close: np.ndarray, vol_last_mult: float = 3.0):
    """Build high/low/open around a close path; last bar is a strong breakout candle."""
    n = len(close)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.4
    low = np.minimum(open_, close) - 0.4
    # Last bar: close in the upper 30% of the range.
    high[-1] = close[-1] + 0.2
    low[-1] = close[-1] - 1.5
    open_[-1] = close[-1] - 0.6
    volume = np.full(n, 1000.0)
    volume[-1] = 1000.0 * vol_last_mult
    return open_, high, low, close, volume


def _rounded_cup_close():
    """Warmup + cosine cup (25% depth, ~100 days) + shallow handle + breakout close."""
    warmup = np.linspace(80.0, 100.0, 220)
    high, low_px = 100.0, 75.0  # 25%
    left = 40
    bottom = 18
    right = 42
    handle = 10
    left_t = np.linspace(0.0, np.pi, left)
    left_side = (high + low_px) / 2.0 + (high - low_px) / 2.0 * np.cos(left_t)
    bottom_side = np.full(bottom, low_px) + np.linspace(0.0, 1.2, bottom)
    right_t = np.linspace(np.pi, 2.0 * np.pi, right)
    right_side = (high + low_px) / 2.0 + (high - low_px) / 2.0 * np.cos(right_t)
    handle_side = np.linspace(99.2, 97.4, handle)
    breakout = np.array([102.0])
    return np.concatenate([warmup, left_side, bottom_side, right_side, handle_side, breakout])


class CupBreakoutDetectorTests(SimpleTestCase):
    def test_rounded_cup_is_detected_on_breakout_bar(self):
        from stage_analysis_v2.services.cup_breakout import CupParams, detect_cup_shape

        close = _rounded_cup_close()
        _o, high, low, close, _v = _ohlc_from_close(close)
        i = len(close) - 1
        setup = detect_cup_shape(high, low, close, i, CupParams())
        self.assertIsNotNone(setup)
        self.assertGreaterEqual(setup.depth_pct, 15.0)
        self.assertLessEqual(setup.depth_pct, 40.0)
        self.assertGreaterEqual(setup.duration, 30)
        self.assertLessEqual(setup.duration, 150)
        self.assertGreaterEqual(setup.recovery_pct, 90.0)
        self.assertTrue(setup.has_handle)
        self.assertLessEqual(setup.handle_depth_pct, 15.0)
        self.assertGreater(setup.breakout_price, setup.cup_high)

    def test_v_shape_is_rejected(self):
        from stage_analysis_v2.services.cup_breakout import CupParams, detect_cup_shape

        warmup = np.linspace(80.0, 100.0, 220)
        drop = np.linspace(100.0, 75.0, 5)
        bounce = np.linspace(75.0, 101.5, 5)
        close = np.concatenate([warmup, drop, bounce])
        _o, high, low, close, _v = _ohlc_from_close(close)
        i = len(close) - 1
        setup = detect_cup_shape(high, low, close, i, CupParams())
        self.assertIsNone(setup)

    def test_shallow_base_is_rejected(self):
        from stage_analysis_v2.services.cup_breakout import CupParams, detect_cup_shape

        warmup = np.linspace(80.0, 100.0, 220)
        high, low_px = 100.0, 94.0  # 6% — too shallow
        left_t = np.linspace(0.0, np.pi, 40)
        left_side = (high + low_px) / 2.0 + (high - low_px) / 2.0 * np.cos(left_t)
        bottom = np.full(16, low_px)
        right_t = np.linspace(np.pi, 2.0 * np.pi, 40)
        right_side = (high + low_px) / 2.0 + (high - low_px) / 2.0 * np.cos(right_t)
        close = np.concatenate([warmup, left_side, bottom, right_side, np.array([101.0])])
        _o, h, l, close, _v = _ohlc_from_close(close)
        setup = detect_cup_shape(h, l, close, len(close) - 1, CupParams())
        self.assertIsNone(setup)

    def test_no_look_ahead_future_bars_do_not_change_signal(self):
        from stage_analysis_v2.services.cup_breakout import CupParams, detect_cup_shape

        close = _rounded_cup_close()
        _o, high, low, close, _v = _ohlc_from_close(close)
        i = len(close) - 1
        params = CupParams()
        before = detect_cup_shape(high, low, close, i, params)
        self.assertIsNotNone(before)

        high2 = np.concatenate([high, [200.0, 210.0]])
        low2 = np.concatenate([low, [50.0, 40.0]])
        close2 = np.concatenate([close, [40.0, 210.0]])
        after = detect_cup_shape(high2, low2, close2, i, params)
        self.assertIsNotNone(after)
        self.assertEqual(before.left_idx, after.left_idx)
        self.assertEqual(before.low_idx, after.low_idx)
        self.assertAlmostEqual(before.cup_high, after.cup_high)
        self.assertAlmostEqual(before.cup_low, after.cup_low)
        self.assertAlmostEqual(before.breakout_price, after.breakout_price)

    def test_only_first_close_breakout_counts(self):
        from stage_analysis_v2.services.cup_breakout import CupParams, detect_cup_shape

        close = _rounded_cup_close()
        _o, high, low, close, _v = _ohlc_from_close(close)
        params = CupParams()
        i = len(close) - 1
        first = detect_cup_shape(high, low, close, i, params)
        self.assertIsNotNone(first)

        # Next day still above the rim — must not fire again.
        close2 = np.concatenate([close, [close[-1] + 0.8]])
        high2 = np.concatenate([high, [close2[-1] + 0.3]])
        low2 = np.concatenate([low, [close2[-1] - 1.0]])
        second = detect_cup_shape(high2, low2, close2, i + 1, params)
        self.assertIsNone(second)

    def test_retest_holds_above_rim(self):
        from stage_analysis_v2.services.cup_breakout import _retest_status

        cup_high = 100.0
        close = np.array([104.0, 100.4, 99.0])
        low = np.array([103.5, 99.5, 97.0])
        self.assertEqual(_retest_status(low, close, 0, cup_high, 0.02), "wait")
        self.assertEqual(_retest_status(low, close, 1, cup_high, 0.02), "ok")
        self.assertEqual(_retest_status(low, close, 2, cup_high, 0.02), "fail")

    def test_confirmed_peak_ignores_unconfirmed_right_side(self):
        from stage_analysis_v2.services.cup_breakout import _confirmed_peak

        h = np.array([1.0, 2.0, 5.0, 4.0, 3.0, 10.0])
        # Peak at index 2 needs 8-bar right confirmation; with i=5 it is not confirmed.
        self.assertFalse(_confirmed_peak(h, 2, 5, w=8))
        # A wide-enough series with a real peak.
        h2 = np.concatenate([np.linspace(1, 3, 10), [9.0], np.linspace(3, 2, 10)])
        L = 10
        i = len(h2)
        self.assertTrue(_confirmed_peak(h2, L, i, w=8))
        # Same peak is not confirmed if we evaluate before the right window prints.
        self.assertFalse(_confirmed_peak(h2, L, L + 4, w=8))

    def test_close_trade_copies_setup_debug(self):
        import pandas as pd
        from stage_analysis_v2.services.backtester import _OpenPos, _close_trade

        pos = _OpenPos(
            symbol="TEST",
            entry_date=pd.Timestamp("2024-01-02"),
            signal_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
            stop=90.0,
            target=120.0,
            qty=10,
            notional=1000.0,
            quality_score=80,
            rs_rating=70.0,
            weekly_stage=0,
            entry_stop=90.0,
            setup={"cup_high": 99.0, "cup_low": 75.0},
        )
        trade = _close_trade(
            pos,
            exit_price=110.0,
            exit_ts=pd.Timestamp("2024-01-10"),
            exit_reason="target",
            days_held=6,
        )
        self.assertEqual(trade.setup["cup_high"], 99.0)
        self.assertEqual(trade.setup["cup_low"], 75.0)


class SignalLogHelperTests(SimpleTestCase):
    def test_raw_signals_merge_with_taken_trades(self):
        from datetime import date
        from stage_analysis_v2.services.backtester import (
            StageV2BacktestResult,
            StageV2Trade,
            attach_signal_log_from_raw,
        )

        result = StageV2BacktestResult(
            strategy_name="test",
            symbols=["AAA", "BBB"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            capital=100000,
            min_quality_score=0,
            market_filter=False,
            trades=[
                StageV2Trade(
                    symbol="AAA",
                    signal_date="2024-01-10",
                    entry_date="2024-01-11",
                    exit_date="2024-01-20",
                    entry_price=100.0,
                    exit_price=110.0,
                    stop_loss=90.0,
                    target=125.0,
                    quantity=10,
                    pnl=100.0,
                    pnl_pct=10.0,
                    rr_achieved=1.0,
                    exit_reason="target",
                    quality_score=80,
                    rs_rating=70.0,
                    weekly_stage=2,
                    days_held=7,
                ),
            ],
        )
        attach_signal_log_from_raw(result, [
            {
                "symbol": "AAA",
                "signal_date": "2024-01-10",
                "entry_date": "2024-01-11",
                "stop_loss": 88.0,
                "target": 120.0,
                "quality_score": 75,
            },
            {
                "symbol": "BBB",
                "signal_date": "2024-01-12",
                "entry_date": "2024-01-13",
                "stop_loss": 50.0,
                "target": 80.0,
                "quality_score": 60,
            },
        ])
        by_sym = {r["symbol"]: r for r in result.signal_log}
        self.assertEqual(len(result.signal_log), 2)
        self.assertEqual(by_sym["AAA"]["status"], "taken")
        self.assertEqual(by_sym["AAA"]["entry_price"], 100.0)
        self.assertEqual(by_sym["AAA"]["stop_loss"], 90.0)
        self.assertEqual(by_sym["AAA"]["exit_date"], "2024-01-20")
        self.assertEqual(by_sym["BBB"]["status"], "skipped")
        self.assertEqual(by_sym["BBB"]["stop_loss"], 50.0)
        self.assertIsNone(by_sym["BBB"]["exit_price"])


class SignalReviewChartTests(SimpleTestCase):
    def test_chart_marks_signal_entry_stop_and_exit(self):
        import json
        import pandas as pd
        from trading.services.charts import build_signal_review_chart

        idx = pd.bdate_range("2024-01-02", periods=80)
        close = np.linspace(100, 130, len(idx))
        df = pd.DataFrame({
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(idx), 1000),
        }, index=idx)
        payload = json.loads(build_signal_review_chart(
            "TEST",
            signal_date="2024-02-15",
            entry_date="2024-02-16",
            exit_date="2024-03-05",
            stop_loss=95.0,
            target=140.0,
            entry_price=110.0,
            exit_price=120.0,
            df=df,
        ))
        names = [t.get("name") for t in payload["data"]]
        self.assertIn("OHLC", names)
        self.assertIn("Signal", names)
        self.assertIn("Entry", names)
        self.assertIn("Exit", names)
        self.assertIn("After signal", names)
        self.assertTrue(payload["layout"].get("shapes"))


class SignalChartApiTests(TestCase):
    def test_requires_symbol_and_date(self):
        url = reverse("stage_analysis_v2:signal_chart_api")
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 400)

    def test_returns_plotly_payload_for_known_symbol(self):
        from datetime import date, timedelta
        from trading.models import DailyPrice, Stock

        Stock.objects.create(symbol="TESTCO", name="Test Co")
        start = date(2024, 1, 2)
        for i in range(60):
            d = start + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            px = 100 + i * 0.2
            DailyPrice.objects.create(
                stock_id="TESTCO",
                date=d,
                open=px - 0.2,
                high=px + 0.8,
                low=px - 0.8,
                close=px,
                volume=1000,
            )
        url = reverse("stage_analysis_v2:signal_chart_api")
        resp = self.client.get(url, {
            "symbol": "TESTCO",
            "signal_date": "2024-02-01",
            "entry_date": "2024-02-02",
            "stop_loss": "95",
            "entry_price": "105",
        })
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["symbol"], "TESTCO")
        self.assertIn("chart", body)
        names = [t.get("name") for t in body["chart"]["data"]]
        self.assertIn("OHLC", names)
        self.assertIn("Signal", names)
