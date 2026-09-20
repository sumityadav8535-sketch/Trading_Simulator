from django.test import SimpleTestCase, TestCase
from unittest.mock import patch

from trading.models import Stock
from trading.services.market_data import get_universe_symbols, resolve_universe_symbols
from trading.services.nifty_smallcap250 import (
    Smallcap250Constituent,
    parse_constituents_csv,
)


SAMPLE_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Aarti Industries Ltd.,Chemicals,AARTIIND,EQ,INE769A01020
Aavas Financiers Ltd.,Financial Services,AAVAS,EQ,INE216P01012
"""


class Smallcap250ParseTests(SimpleTestCase):
    def test_parse_constituents_csv(self):
        rows = parse_constituents_csv(SAMPLE_CSV)
        self.assertEqual([r.symbol for r in rows], ["AARTIIND", "AAVAS"])
        self.assertEqual(rows[0].industry, "Chemicals")
        self.assertEqual(rows[1].name, "Aavas Financiers Ltd.")


class Smallcap250UniverseTests(TestCase):
    def test_mark_and_resolve_universe(self):
        Stock.objects.create(symbol="RELIANCE", name="Reliance", is_nifty200=True, is_active=True)
        Stock.objects.create(symbol="AAVAS", name="", is_active=True)

        constituents = [
            Smallcap250Constituent("AARTIIND", "Aarti Industries Ltd.", "Chemicals"),
            Smallcap250Constituent("AAVAS", "Aavas Financiers Ltd.", "Financial Services"),
        ]
        with patch(
            "trading.services.nifty_smallcap250.fetch_smallcap250_constituents",
            return_value=constituents,
        ):
            from trading.services.nifty_smallcap250 import ensure_nifty_smallcap250_marked

            symbols = ensure_nifty_smallcap250_marked()

        self.assertEqual(sorted(symbols), ["AARTIIND", "AAVAS"])
        aarti = Stock.objects.get(pk="AARTIIND")
        self.assertTrue(aarti.is_nifty_smallcap250)
        self.assertEqual(aarti.sector, "Chemicals")
        aavas = Stock.objects.get(pk="AAVAS")
        self.assertTrue(aavas.is_nifty_smallcap250)
        self.assertEqual(aavas.name, "Aavas Financiers Ltd.")
        self.assertFalse(Stock.objects.get(pk="RELIANCE").is_nifty_smallcap250)

        self.assertEqual(get_universe_symbols(nifty_smallcap250_only=True), ["AARTIIND", "AAVAS"])
        self.assertEqual(get_universe_symbols(nifty200_only=True), ["RELIANCE"])
        self.assertEqual(resolve_universe_symbols("smallcap250"), ["AARTIIND", "AAVAS"])
        self.assertEqual(resolve_universe_symbols("nifty_smallcap250"), ["AARTIIND", "AAVAS"])
        self.assertEqual(resolve_universe_symbols("nifty200"), ["RELIANCE"])


class GapRsiStrategyTests(SimpleTestCase):
    def test_rsi_band_and_gap_filters(self):
        from trading.services.intraday_gap import passes_gap_down, passes_rsi_band

        self.assertTrue(passes_rsi_band(45.0))
        self.assertTrue(passes_rsi_band(70.0))
        self.assertTrue(passes_rsi_band(68.0))
        self.assertTrue(passes_rsi_band(52.1))
        self.assertFalse(passes_rsi_band(44.9))
        self.assertFalse(passes_rsi_band(70.1))
        self.assertFalse(passes_rsi_band(None))

        self.assertTrue(passes_gap_down(-0.02))
        self.assertTrue(passes_gap_down(-0.06))
        self.assertFalse(passes_gap_down(-0.061))
        self.assertFalse(passes_gap_down(-0.08))
        self.assertFalse(passes_gap_down(-0.019))
        self.assertFalse(passes_gap_down(-0.16))
        self.assertFalse(passes_gap_down(0.03))

    def test_size_long_uses_risk_and_deploy_cap(self):
        from trading.services.intraday_gap import size_long

        sized = size_long(
            entry=100.0, stop=99.0, target=104.0, equity=100_000,
            risk_pct=5.0, leverage=5.0, max_deploy=0.35,
        )
        self.assertEqual(sized["qty"], 1750)  # deploy cap 5x * 0.35 * 100k / 100
        self.assertEqual(sized["risk_inr"], 1750.0)
        self.assertEqual(sized["target_pnl"], 7000.0)
        self.assertEqual(sized["rr"], 4.0)

        none = size_long(entry=100.0, stop=101.0, target=104.0)
        self.assertEqual(none["qty"], 0)

    def test_build_setup_requires_rsi_and_gap(self):
        from datetime import date

        from trading.services.intraday_gap import build_setup

        ok = build_setup("RELIANCE", 97.0, 100.0, 0.8, 52.0, date(2026, 8, 26))
        self.assertIsNotNone(ok)
        self.assertEqual(ok["side"], "long")
        self.assertEqual(ok["entry"], 97.0)
        self.assertEqual(ok["target"], 97.48)  # 0.5% from entry, not prior close
        self.assertEqual(ok["stop"], round(97.0 - 1.5 * 0.8, 2))
        self.assertGreater(ok["qty"], 0)
        self.assertEqual(ok["entry_time"], "09:30")

        hot = build_setup("RELIANCE", 97.0, 100.0, 0.8, 68.0, date(2026, 8, 26))
        self.assertIsNotNone(hot)
        crash = build_setup("RELIANCE", 93.0, 100.0, 0.8, 52.0, date(2026, 8, 26))
        self.assertIsNone(crash)
        self.assertIsNone(build_setup("RELIANCE", 97.0, 100.0, 0.8, 40.0, date(2026, 8, 26)))
        self.assertIsNone(build_setup("RELIANCE", 99.0, 100.0, 0.8, 52.0, date(2026, 8, 26)))
        self.assertIsNone(build_setup("RELIANCE", 103.0, 100.0, 0.8, 52.0, date(2026, 8, 26)))

    def test_prior_close_uses_1310_not_auction(self):
        from datetime import date, datetime
        from zoneinfo import ZoneInfo

        import pandas as pd

        from trading.services.intraday_gap import (
            prior_close_from_5m,
            scan_gap_setups,
        )

        ist = ZoneInfo("Asia/Kolkata")
        yest = date(2026, 8, 31)
        today = date(2026, 9, 1)

        def bars(session, rows):
            idx = [datetime(session.year, session.month, session.day, h, m, tzinfo=ist) for h, m, *_ in rows]
            return pd.DataFrame(
                {
                    "open": [r[2] for r in rows],
                    "high": [r[3] for r in rows],
                    "low": [r[4] for r in rows],
                    "close": [r[5] for r in rows],
                    "volume": [1] * len(rows),
                },
                index=pd.DatetimeIndex(idx),
            )

        yest_df = bars(yest, [
            (9, 15, 674.1, 674.1, 671.0, 671.3),
            (13, 10, 671.35, 671.35, 670.6, 670.7),
            (15, 10, 671.05, 671.7, 667.5, 667.5),
            (15, 15, 668.0, 691.05, 668.0, 691.05),
        ])
        today_df = bars(today, [
            (9, 15, 673.35, 673.35, 667.6, 671.25),
            (9, 30, 671.35, 671.75, 670.05, 670.05),
        ])
        df = pd.concat([yest_df, today_df])
        self.assertEqual(prior_close_from_5m(df, today), 667.5)
        self.assertEqual(prior_close_from_5m(df, today, "13:10"), 670.7)

        # 15:10 bar is a closing-auction spike (close = high, wide range) — skip it.
        spiked = bars(yest, [
            (9, 15, 2323.0, 2324.0, 2322.0, 2323.0),
            (15, 5, 2329.0, 2331.0, 2327.0, 2329.0),
            (15, 10, 2328.0, 2399.3, 2323.0, 2399.3),
        ])
        today_only = bars(today, [
            (9, 15, 2348.8, 2350.0, 2330.0, 2331.0),
        ])
        self.assertEqual(prior_close_from_5m(pd.concat([spiked, today_only]), today), 2329.0)

        daily = {
            "DLF": pd.DataFrame(
                {"close": [691.05], "rsi14": [62.0], "atr14": [3.0]},
                index=[yest],
            ),
        }
        live = scan_gap_setups(
            session=today,
            symbols=["DLF"],
            daily=daily,
            frames={"DLF": df},
            opens={
                "DLF": {
                    "session": today, "open": 673.35, "entry": 671.35,
                    "atr": 2.95, "pending": False, "bounce": True, "entry_time": "09:30",
                },
            },
        )
        self.assertEqual(live["counts"]["qualified"], 0)
        self.assertEqual(live["taken"], [])

    def test_select_trades_keeps_largest_gaps(self):
        from trading.services.intraday_gap import select_trades

        cands = [
            {"symbol": "A", "gap": -0.021, "gap_pct": -2.1},
            {"symbol": "B", "gap": -0.055, "gap_pct": -5.5},
            {"symbol": "C", "gap": -0.033, "gap_pct": -3.3},
        ]
        taken, watch = select_trades(cands, top_k=2, max_pos=4)
        self.assertEqual([t["symbol"] for t in taken], ["B", "C"])
        self.assertEqual([t["status"] for t in taken], ["trade", "trade"])
        self.assertEqual([w["symbol"] for w in watch], ["A"])
        self.assertEqual(watch[0]["status"], "watch")

        all_taken, none_watch = select_trades(cands, top_k=0, max_pos=0)
        self.assertEqual([t["symbol"] for t in all_taken], ["B", "C", "A"])
        self.assertEqual(none_watch, [])

    def test_scan_injects_opens_without_pickles(self):
        from datetime import date

        import pandas as pd

        from trading.services.intraday_gap import scan_gap_setups

        sess = date(2026, 8, 26)
        prior = date(2026, 8, 25)
        daily = {
            "AAA": pd.DataFrame(
                {"close": [100.0], "rsi14": [52.0], "atr14": [2.0]},
                index=[prior],
            ),
            "BBB": pd.DataFrame(
                {"close": [200.0], "rsi14": [40.0], "atr14": [3.0]},
                index=[prior],
            ),
            "CCC": pd.DataFrame(
                {"close": [80.0], "rsi14": [58.0], "atr14": [1.5]},
                index=[prior],
            ),
        }
        opens = {
            "AAA": {"session": sess, "open": 97.0, "atr": 0.8},
            "BBB": {"session": sess, "open": 190.0, "atr": 1.0},
            "CCC": {"session": sess, "open": 77.5, "atr": 0.5},
        }
        live = scan_gap_setups(
            session=sess,
            symbols=["AAA", "BBB", "CCC"],
            daily=daily,
            opens=opens,
        )
        taken_syms = [t["symbol"] for t in live["taken"]]
        self.assertIn("AAA", taken_syms)
        self.assertIn("CCC", taken_syms)
        self.assertNotIn("BBB", taken_syms)
        trade = next(t for t in live["taken"] if t["symbol"] == "AAA")
        self.assertEqual(trade["entry"], 97.0)
        self.assertEqual(trade["target"], 97.48)
        self.assertEqual(trade["stop"], 95.8)
        self.assertGreater(trade["qty"], 0)
        self.assertEqual(live["counts"]["qualified"], 2)
        self.assertEqual(live["counts"]["trades"], 2)

        falling = dict(opens)
        falling["AAA"] = {
            "session": sess, "open": 97.0, "entry": 96.4, "atr": 0.8,
            "pending": False, "bounce": False, "close915": 96.8, "entry_time": "09:30",
        }
        falling["CCC"] = {
            "session": sess, "open": 77.5, "entry": 77.8, "atr": 0.5,
            "pending": False, "bounce": True, "close915": 77.6, "entry_time": "09:30",
        }
        bounced = scan_gap_setups(
            session=sess, symbols=["AAA", "BBB", "CCC"], daily=daily, opens=falling,
        )
        taken_syms = [t["symbol"] for t in bounced["taken"]]
        self.assertNotIn("AAA", taken_syms)
        self.assertIn("CCC", taken_syms)

    def test_compute_long_target_is_not_prior_close(self):
        from trading.services.intraday_gap import compute_long_target

        self.assertEqual(compute_long_target(97.0, 100.0, 0.8, 96.52, "fill"), 100.0)
        self.assertEqual(compute_long_target(97.0, 100.0, 0.8, 96.52, "pct1.0"), 97.97)
        self.assertEqual(compute_long_target(97.0, 100.0, 0.8, 96.52, "half"), 98.5)

    def test_evaluate_gap_trade_win_loss_open(self):
        from datetime import date, datetime
        from zoneinfo import ZoneInfo

        import pandas as pd

        from trading.services.intraday_gap import evaluate_gap_trade

        ist = ZoneInfo("Asia/Kolkata")
        sess = date(2026, 8, 25)

        def bars(rows):
            idx = [datetime(sess.year, sess.month, sess.day, h, m, tzinfo=ist) for h, m, *_ in rows]
            return pd.DataFrame(
                {
                    "open": [r[2] for r in rows],
                    "high": [r[3] for r in rows],
                    "low": [r[4] for r in rows],
                    "close": [r[5] for r in rows],
                    "volume": [1] * len(rows),
                },
                index=pd.DatetimeIndex(idx),
            )

        setup = {
            "symbol": "AAA",
            "session": sess,
            "entry": 97.0,
            "stop": 96.0,
            "target": 100.0,
            "qty": 10,
            "entry_time": "09:15",
        }
        win_df = bars([
            (9, 15, 97.0, 97.4, 96.8, 97.2),
            (9, 20, 97.2, 100.5, 97.1, 100.2),
        ])
        win = evaluate_gap_trade(setup, win_df, now=datetime(2026, 8, 25, 16, 0, tzinfo=ist))
        self.assertEqual(win["result"], "win")
        self.assertEqual(win["reason"], "target")
        self.assertEqual(win["exit"], 100.0)
        self.assertEqual(win["pnl"], 30.0)

        loss_df = bars([
            (9, 15, 97.0, 97.4, 96.8, 97.2),
            (9, 20, 97.2, 97.5, 95.5, 96.2),
        ])
        loss = evaluate_gap_trade(setup, loss_df, now=datetime(2026, 8, 25, 16, 0, tzinfo=ist))
        self.assertEqual(loss["result"], "loss")
        self.assertEqual(loss["reason"], "sl")
        self.assertEqual(loss["exit"], 96.0)
        self.assertEqual(loss["pnl"], -10.0)

        open_df = bars([
            (9, 15, 97.0, 97.4, 96.8, 97.2),
            (10, 0, 97.2, 98.0, 97.0, 97.8),
        ])
        live = evaluate_gap_trade(setup, open_df, now=datetime(2026, 8, 25, 10, 5, tzinfo=ist))
        self.assertEqual(live["result"], "open")
        self.assertEqual(live["reason"], "open")
        self.assertEqual(live["last"], 97.8)
        self.assertEqual(live["pnl"], 8.0)

        eod = evaluate_gap_trade(setup, open_df, now=datetime(2026, 8, 26, 10, 0, tzinfo=ist))
        self.assertEqual(eod["result"], "win")
        self.assertEqual(eod["reason"], "eod")
        self.assertEqual(eod["exit"], 97.8)

        missing = evaluate_gap_trade(setup, None, now=datetime(2026, 8, 25, 16, 0, tzinfo=ist))
        self.assertEqual(missing["result"], "no_data")

        late = dict(setup)
        late["entry_time"] = "09:30"
        late["entry"] = 97.2
        late["stop"] = 96.0
        late_df = bars([
            (9, 15, 97.0, 97.4, 95.0, 96.5),
            (9, 20, 96.5, 97.0, 95.2, 96.8),
            (9, 30, 97.2, 100.5, 97.1, 100.2),
        ])
        skipped = evaluate_gap_trade(late, late_df, now=datetime(2026, 8, 25, 16, 0, tzinfo=ist))
        self.assertEqual(skipped["result"], "win")
        self.assertEqual(skipped["reason"], "target")

        future_df = bars([
            (9, 15, 97.0, 97.4, 96.8, 97.2),
            (10, 0, 97.2, 98.0, 97.0, 97.8),
            (15, 15, 99.0, 101.0, 98.8, 100.5),
        ])
        no_peek = evaluate_gap_trade(setup, future_df, now=datetime(2026, 8, 25, 10, 5, tzinfo=ist))
        self.assertEqual(no_peek["result"], "open")
        self.assertEqual(no_peek["last"], 97.8)

    def test_recent_gap_days_marks_yesterday_loss_and_today_open(self):
        from datetime import date, datetime
        from zoneinfo import ZoneInfo
        from unittest.mock import patch

        import pandas as pd

        from trading.services.intraday_data import MarketStatus
        from trading.services.intraday_gap import recent_gap_days

        ist = ZoneInfo("Asia/Kolkata")
        today = date(2026, 8, 26)
        yest = date(2026, 8, 25)
        prior = date(2026, 8, 24)

        def day_bars(session, rows):
            idx = [datetime(session.year, session.month, session.day, h, m, tzinfo=ist) for h, m, *_ in rows]
            return pd.DataFrame(
                {
                    "open": [r[2] for r in rows],
                    "high": [r[3] for r in rows],
                    "low": [r[4] for r in rows],
                    "close": [r[5] for r in rows],
                    "volume": [1] * len(rows),
                },
                index=pd.DatetimeIndex(idx),
            )

        # Yesterday: 9:30 stop hit. Today: still in play after 9:30.
        yest_df = day_bars(yest, [
            (9, 15, 97.0, 97.2, 96.9, 97.0),
            (9, 20, 97.0, 97.1, 90.0, 91.0),
            (9, 30, 97.0, 97.1, 90.0, 91.0),
            (13, 10, 100.0, 100.2, 99.8, 100.0),
            (15, 15, 104.0, 110.0, 103.0, 110.0),
        ])
        today_df = day_bars(today, [
            (9, 15, 97.0, 97.3, 96.95, 97.05),
            (9, 30, 97.10, 97.3, 96.95, 97.1),
            (10, 0, 97.1, 97.4, 97.0, 97.2),
        ])
        frames = {"AAA": pd.concat([yest_df, today_df])}
        daily = {
            "AAA": pd.DataFrame(
                {"close": [100.0, 100.0], "rsi14": [52.0, 52.0], "atr14": [2.0, 2.0]},
                index=[prior, yest],
            ),
        }
        market = MarketStatus(
            is_open=True,
            status="open",
            message="Market open",
            now_ist="10:05:00 IST",
            session_date=today.isoformat(),
        )
        now = datetime(2026, 8, 26, 10, 5, tzinfo=ist)
        with patch("trading.services.intraday_gap.get_market_status", return_value=market):
            recent = recent_gap_days(
                session=today,
                symbols=["AAA"],
                daily=daily,
                frames=frames,
                now=now,
            )
        self.assertEqual(recent["yesterday"]["headline"], "LOSS")
        self.assertEqual(recent["yesterday"]["outcome"], "loss")
        self.assertEqual(recent["yesterday"]["trades"][0]["symbol"], "AAA")
        self.assertEqual(recent["yesterday"]["trades"][0]["result"], "loss")
        self.assertEqual(recent["today"]["headline"], "OPEN")
        self.assertEqual(recent["today"]["outcome"], "open")
        self.assertEqual(recent["today"]["trades"][0]["result"], "open")
        self.assertEqual([c["title"] for c in recent["cards"]], ["Yesterday", "Today"])
        # Gap must use 15:10-or-before close (100), not the 15:15 auction spike (110).
        self.assertEqual(recent["today"]["trades"][0]["pdc"], 100.0)
        self.assertEqual(recent["today"]["trades"][0]["pdc_source"], "5m_15:10")


class GapPageTests(TestCase):
    @patch("trading.services.intraday_gap.load_gap_page")
    def test_page_shows_live_levels_and_history(self, mock_page):
        mock_page.return_value = {
            "story": {
                "net_pnl": 52400,
                "total_return_pct": 52.4,
                "win_rate": 31.8,
                "profit_factor": 2.43,
                "trades": 44,
                "max_dd_pct": 8.3,
                "oos_pnl": 23038,
                "worst_day": -2100,
                "months": [{"month": "2026-08", "trades": 12, "pnl": 19611, "wr": 33.0}],
            },
            "months": [{"month": "2026-08", "trades": 12, "pnl": 19611, "wr": 33.0}],
            "window": {"start": "2026-06-03", "end": "2026-08-25", "stocks": 203},
            "fill_stats": [],
            "trades": [
                {
                    "symbol": "INFY",
                    "side": "long",
                    "gap": -2.4,
                    "rsi": 51.2,
                    "entry_ts": "2026-07-15 09:15:00+05:30",
                    "exit_ts": "2026-07-15 10:05:00+05:30",
                    "entry": 1480.0,
                    "stop": 1472.5,
                    "target": 1516.0,
                    "exit": 1516.0,
                    "qty": 40,
                    "pnl": 1440.0,
                    "reason": "target",
                }
            ],
            "live": {
                "session": "2026-08-25",
                "is_today": True,
                "market": {"status": "open", "message": "Market open", "now_ist": "09:20:00 IST", "session_date": "2026-08-25"},
                "taken": [
                    {
                        "rank": 1,
                        "symbol": "RELIANCE",
                        "side": "long",
                        "gap_pct": -2.8,
                        "rsi": 54.0,
                        "entry": 1380.0,
                        "stop": 1371.5,
                        "target": 1420.0,
                        "qty": 55,
                        "risk_inr": 467.5,
                        "target_pnl": 2200.0,
                        "rr": 4.7,
                        "notional": 75900,
                    }
                ],
                "watch": [],
                "counts": {"scanned": 200, "rsi_band": 40, "qualified": 1, "trades": 1},
            },
            "strategy": {
                "entry": ["Gap down ≥ 2%", "RSI 45–65"],
                "exit": ["Target prior close"],
                "top_k": 2,
            },
            "recent_days": {
                "today": {"title": "Today", "label": "Wed 26 Aug", "headline": "WIN", "status": "traded", "outcome": "win", "count": 1, "wins": 1, "losses": 0, "open_count": 0, "pnl": 2200, "open_pnl": 0, "trades": [{"symbol": "RELIANCE", "gap_pct": -2.8, "entry": 1380.0, "exit": 1420.0, "pnl": 2200, "result": "win", "reason_label": "Target"}]},
                "yesterday": {"title": "Yesterday", "label": "Tue 25 Aug", "headline": "LOSS", "status": "traded", "outcome": "loss", "count": 1, "wins": 0, "losses": 1, "open_count": 0, "pnl": -900, "open_pnl": 0, "trades": [{"symbol": "INFY", "gap_pct": -2.1, "entry": 1480.0, "exit": 1472.5, "pnl": -900, "result": "loss", "reason_label": "Stop"}]},
                "cards": [
                    {"title": "Yesterday", "label": "Tue 25 Aug", "headline": "LOSS", "status": "traded", "outcome": "loss", "count": 1, "wins": 0, "losses": 1, "open_count": 0, "pnl": -900, "open_pnl": 0, "trades": [{"symbol": "INFY", "gap_pct": -2.1, "entry": 1480.0, "exit": 1472.5, "pnl": -900, "result": "loss", "reason_label": "Stop"}]},
                    {"title": "Today", "label": "Wed 26 Aug", "headline": "WIN", "status": "traded", "outcome": "win", "count": 1, "wins": 1, "losses": 0, "open_count": 0, "pnl": 2200, "open_pnl": 0, "trades": [{"symbol": "RELIANCE", "gap_pct": -2.8, "entry": 1380.0, "exit": 1420.0, "pnl": 2200, "result": "win", "reason_label": "Target"}]},
                ],
            },
        }
        resp = self.client.get("/intraday/gap/")
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn("RELIANCE", body)
        self.assertIn("₹1380.0", body)
        self.assertIn("₹1371.5", body)
        self.assertIn("₹1420.0", body)
        self.assertIn(">55<", body)
        self.assertIn("INFY", body)
        self.assertIn("Stop loss", body)
        self.assertIn("Backtest trades", body)
        self.assertIn("Yesterday", body)
        self.assertIn("LOSS", body)
        self.assertIn("WIN", body)
        self.assertIn("Stop", body)
        self.assertIn("Target", body)

    @patch("trading.services.intraday_gap.scan_gap_setups")
    def test_live_api(self, mock_scan):
        mock_scan.return_value = {"session": "2026-08-25", "taken": [], "watch": []}
        resp = self.client.get("/api/intraday/gap/live/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"], "2026-08-25")


class LocalhostAutoRefreshTests(SimpleTestCase):
    def test_previous_session_skips_weekend(self):
        from datetime import date

        from trading.services.localhost_auto_refresh import previous_session_date

        monday = date(2026, 8, 31)
        self.assertEqual(previous_session_date(monday), date(2026, 8, 28))
        self.assertEqual(
            previous_session_date(monday, {date(2026, 8, 27), date(2026, 8, 26)}),
            date(2026, 8, 27),
        )

    def test_combine_and_last_signal(self):
        from datetime import date

        from trading.services.localhost_auto_refresh import combine_fno_days, pick_last_signal

        today = date(2026, 9, 1)
        yest = date(2026, 8, 31)
        nifty_flat = {
            "instrument": "NIFTY",
            "trades": [],
            "bars_today": 75,
            "message": "No trades",
        }
        bn_win = {
            "instrument": "BANKNIFTY",
            "trades": [{
                "side": "SHORT", "result": "WIN", "entry_time": "10:15",
                "exit_time": "11:00", "pnl_inr": 4200,
            }],
            "bars_today": 75,
            "message": "1 trade",
        }
        today_day = combine_fno_days([nifty_flat, nifty_flat], today, "Today")
        yest_day = combine_fno_days([nifty_flat, bn_win], yest, "Yesterday")
        self.assertEqual(today_day["trade_count"], 0)
        self.assertEqual(today_day["status"], "flat")
        self.assertEqual(yest_day["trade_count"], 1)
        self.assertEqual(yest_day["wins"], 1)
        self.assertEqual(yest_day["net_pnl"], 4200)
        last = pick_last_signal([today_day, yest_day])
        self.assertEqual(last["date"], "2026-08-31")
        self.assertEqual(last["trades"][0]["instrument"], "BANKNIFTY")

    def test_should_fetch_while_market_open(self):
        from datetime import date
        from types import SimpleNamespace
        from unittest.mock import patch

        from trading.services.localhost_auto_refresh import _should_fetch

        with patch(
            "trading.services.localhost_auto_refresh.history_needs_update",
            return_value=False,
        ), patch(
            "trading.services.localhost_auto_refresh.get_market_status",
            return_value=SimpleNamespace(status="open"),
        ):
            self.assertTrue(_should_fetch(date(2026, 9, 1)))
        with patch(
            "trading.services.localhost_auto_refresh.history_needs_update",
            return_value=False,
        ), patch(
            "trading.services.localhost_auto_refresh.get_market_status",
            return_value=SimpleNamespace(status="closed"),
        ):
            self.assertFalse(_should_fetch(date(2026, 9, 1)))

    def test_disabled_during_tests(self):
        from trading.services.localhost_auto_refresh import (
            is_auto_refresh_enabled,
            reset_auto_refresh_for_tests,
            start_localhost_auto_refresh,
        )

        reset_auto_refresh_for_tests()
        self.assertFalse(is_auto_refresh_enabled())
        result = start_localhost_auto_refresh()
        self.assertFalse(result["started"])
        self.assertFalse(result["enabled"])

    def test_start_runs_only_once_per_process(self):
        from trading.services.localhost_auto_refresh import (
            _set_status,
            reset_auto_refresh_for_tests,
            start_localhost_auto_refresh,
        )

        reset_auto_refresh_for_tests()
        with patch(
            "trading.services.localhost_auto_refresh.is_auto_refresh_enabled",
            return_value=True,
        ), patch("trading.services.localhost_auto_refresh.threading.Thread") as thread:
            first = start_localhost_auto_refresh()
            self.assertTrue(first["started"])
            self.assertEqual(thread.call_count, 1)
            _set_status(running=False, phase="done", message="Signal check complete.")
            second = start_localhost_auto_refresh()
            self.assertFalse(second["started"])
            self.assertEqual(thread.call_count, 1)
            self.assertIn("Already ran", second["message"])
            forced = start_localhost_auto_refresh(force=True)
            self.assertTrue(forced["started"])
            self.assertEqual(thread.call_count, 2)


class LocalhostAutoRefreshApiTests(TestCase):
    def test_dashboard_shows_signal_board(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "startup-signal-board")
        self.assertContains(resp, "auto-refresh-banner")
        self.assertContains(resp, "api/intraday/startup/")

    def test_status_api_does_not_start_in_tests(self):
        from trading.services.localhost_auto_refresh import reset_auto_refresh_for_tests

        reset_auto_refresh_for_tests()
        resp = self.client.get("/api/intraday/startup/?autostart=1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data.get("enabled"))
        self.assertNotEqual(data.get("phase"), "fetching")

    def test_status_api_never_starts_a_job(self):
        with patch("trading.views.start_localhost_auto_refresh") as start:
            resp = self.client.get("/api/intraday/startup/?autostart=1")
        self.assertEqual(resp.status_code, 200)
        start.assert_not_called()

    def test_equity_sync_universe_includes_nifty200(self):
        from trading.models import Stock
        from trading.services.intraday_history_sync import equity_symbols_for_sync

        Stock.objects.create(symbol="RELIANCE", name="Reliance", is_nifty200=True, is_active=True)
        Stock.objects.create(symbol="INFY", name="Infosys", is_nifty100=True, is_nifty200=True, is_active=True)
        with patch(
            "trading.services.intraday_history_sync.ensure_nifty100_marked",
            return_value=["INFY"],
        ):
            symbols = equity_symbols_for_sync()
        self.assertIn("INFY", symbols)
        self.assertIn("RELIANCE", symbols)


class FundamentalSwingTests(SimpleTestCase):
    def test_percent_and_debt_normalisation(self):
        from trading.services.fundamental_swing import as_pct, de_ratio_from_yahoo

        self.assertEqual(as_pct(0.32), 32.0)
        self.assertEqual(as_pct(18.0), 18.0)
        self.assertAlmostEqual(de_ratio_from_yahoo(9.541), 0.09541)
        self.assertIsNone(as_pct(None))

    def test_quality_beats_junk(self):
        from trading.services.fundamental_swing import score_metrics

        quality = score_metrics({
            "roe": 28, "profit_margin": 18, "operating_margin": 22,
            "revenue_growth": 20, "earnings_growth": 30,
            "de_ratio": 0.2, "current_ratio": 1.8,
            "pe": 18, "peg": 0.9, "insider_pct": 25, "institution_pct": 40,
            "sector": "Industrials",
        })
        junk = score_metrics({
            "roe": 4, "profit_margin": 1, "operating_margin": 2,
            "revenue_growth": -5, "earnings_growth": -10,
            "de_ratio": 2.4, "current_ratio": 0.7,
            "pe": 90, "peg": 4, "insider_pct": 1, "institution_pct": 2,
            "sector": "Industrials",
        })
        self.assertGreater(quality["score"], 70)
        self.assertLess(junk["score"], 25)
        self.assertGreater(quality["score"], junk["score"])

    def test_filters_reject_expensive_thin_names(self):
        from trading.services.fundamental_swing import QUALITY_GROWTH, FundParams, passes_filters

        params = QUALITY_GROWTH
        ok, reasons = passes_filters({
            "roe": 20, "profit_margin": 12, "revenue_growth": 15,
            "earnings_growth": 20, "pe": 80, "de_ratio": 0.3,
            "sector": "Technology",
        }, params)
        self.assertFalse(ok)
        self.assertTrue(any("PE" in r for r in reasons))

        bank_ok, bank_reasons = passes_filters({
            "roe": 16, "profit_margin": 22, "revenue_growth": 12,
            "earnings_growth": 14, "pe": 15, "de_ratio": 6.0,
            "sector": "Financial Services", "industry": "Banks — Regional",
        }, params)
        self.assertTrue(bank_ok)
        self.assertEqual(bank_reasons, [])

        blocked, blocked_reasons = passes_filters({
            "roe": 16, "profit_margin": 22, "revenue_growth": 20,
            "earnings_growth": 30, "pe": 15, "de_ratio": 6.0,
            "sector": "Financial Services", "industry": "Banks — Regional",
        }, FundParams(exclude_financials=True, min_roe=12, min_profit_margin=5,
                      min_revenue_growth=15, min_earnings_growth=25, max_pe=55))
        self.assertFalse(blocked)
        self.assertTrue(any("Financial" in r for r in blocked_reasons))

    def test_point_in_time_ignores_future_statements(self):
        from datetime import date

        from trading.services.fundamental_swing import latest_statement_on_or_before

        statements = {
            "2024-03-31": {"Net Income": 100},
            "2025-03-31": {"Net Income": 200},
        }
        # FY2025 published ~2025-06-29. On 2025-04-01 it is not yet available.
        found = latest_statement_on_or_before(statements, date(2025, 4, 1), lag_days=90)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], "2024-03-31")
        later = latest_statement_on_or_before(statements, date(2025, 7, 1), lag_days=90)
        self.assertEqual(later[0], "2025-03-31")

    def test_statement_growth_and_margins(self):
        from datetime import date

        from trading.services.fundamental_swing import metrics_from_statements

        income = {
            "2024-03-31": {"Total Revenue": 100, "Net Income": 10, "Operating Income": 15, "Diluted EPS": 10},
            "2025-03-31": {"Total Revenue": 130, "Net Income": 16, "Operating Income": 22, "Diluted EPS": 16},
        }
        balance = {
            "2025-03-31": {"Stockholders Equity": 80, "Total Debt": 16, "Current Assets": 40, "Current Liabilities": 20},
        }
        m = metrics_from_statements(income, balance, {}, date(2025, 7, 1), price=320, financial_currency="INR")
        self.assertEqual(m["fy"], "2025-03-31")
        self.assertAlmostEqual(m["profit_margin"], 12.307, places=2)
        self.assertAlmostEqual(m["revenue_growth"], 30.0)
        self.assertAlmostEqual(m["earnings_growth"], 60.0)
        self.assertAlmostEqual(m["roe"], 20.0)
        self.assertAlmostEqual(m["de_ratio"], 0.2)
        self.assertAlmostEqual(m["pe"], 20.0)

    def test_equal_weight_window_return(self):
        from datetime import date

        import pandas as pd

        from trading.services.fundamental_swing import FundParams, run_window

        idx = pd.bdate_range("2024-06-28", "2025-07-05")
        a = pd.Series(100.0, index=idx)
        a.loc[idx >= "2025-06-30"] = 200.0  # +100%
        b = pd.Series(50.0, index=idx)
        b.loc[idx >= "2025-06-30"] = 75.0  # +50%
        calendar = pd.DatetimeIndex(idx)
        cache = {
            "stocks": {
                "AAA": {
                    "info": {"longName": "Aaa", "sector": "Industrials", "financialCurrency": "INR"},
                    "income": {
                        "2024-03-31": {"Total Revenue": 80, "Net Income": 12, "Operating Income": 14, "Diluted EPS": 8},
                        "2025-03-31": {"Total Revenue": 100, "Net Income": 18, "Operating Income": 20, "Diluted EPS": 12},
                    },
                    "balance": {"2024-03-31": {"Stockholders Equity": 60, "Total Debt": 10, "Current Assets": 30, "Current Liabilities": 15},
                                "2025-03-31": {"Stockholders Equity": 70, "Total Debt": 10, "Current Assets": 32, "Current Liabilities": 15}},
                    "cashflow": {},
                },
                "BBB": {
                    "info": {"longName": "Bbb", "sector": "Industrials", "financialCurrency": "INR"},
                    "income": {
                        "2024-03-31": {"Total Revenue": 90, "Net Income": 14, "Operating Income": 16, "Diluted EPS": 7},
                        "2025-03-31": {"Total Revenue": 110, "Net Income": 20, "Operating Income": 22, "Diluted EPS": 10},
                    },
                    "balance": {"2024-03-31": {"Stockholders Equity": 70, "Total Debt": 8, "Current Assets": 28, "Current Liabilities": 14},
                                "2025-03-31": {"Stockholders Equity": 80, "Total Debt": 8, "Current Assets": 30, "Current Liabilities": 14}},
                    "cashflow": {},
                },
            }
        }
        params = FundParams(top_n=2, min_score=0, min_roe=0, min_profit_margin=0,
                            min_revenue_growth=0, min_earnings_growth=0, max_pe=100)
        result = run_window(
            cache, ["AAA", "BBB"], date(2024, 7, 1), date(2025, 7, 1),
            params, {"AAA": a, "BBB": b}, calendar, capital=100_000,
        )
        # Equal weight of +100% and +50% ≈ +75%
        self.assertIsNotNone(result["total_return_pct"])
        self.assertGreater(result["total_return_pct"], 70)
        self.assertLess(result["total_return_pct"], 80)
        self.assertTrue(any(t.get("doubled") for t in result["holdings_history"]))

    def test_page_empty_without_cache(self):
        from trading.services.fundamental_swing import load_page

        with patch("trading.services.fundamental_swing.load_cache", return_value={"stocks": {}}):
            page = load_page(rebuild=True)
        self.assertTrue(page.get("needs_fetch"))
        self.assertEqual(page.get("picks"), [])
        self.assertEqual(page.get("today_book"), [])

    def test_rebalance_calendar_and_pack_alias(self):
        from datetime import date

        from trading.services.fundamental_swing import (
            AGGRESSIVE_GROWTH,
            QUALITY_GROWTH,
            last_rebalance_on_or_before,
            next_rebalance_after,
            params_by_name,
            parse_iso_date,
        )

        self.assertEqual(last_rebalance_on_or_before(date(2026, 9, 16)), date(2026, 7, 1))
        self.assertEqual(next_rebalance_after(date(2026, 9, 16)), date(2027, 7, 1))
        self.assertEqual(last_rebalance_on_or_before(date(2026, 6, 30)), date(2025, 7, 1))
        self.assertEqual(next_rebalance_after(date(2026, 6, 30)), date(2026, 7, 1))
        self.assertEqual(params_by_name("quality").name, QUALITY_GROWTH.name)
        self.assertEqual(params_by_name("Aggressive Growth").name, AGGRESSIVE_GROWTH.name)
        self.assertEqual(params_by_name("momentum").name, "Momentum Swing")
        self.assertEqual(params_by_name("momentum").rebalance, "21d")
        self.assertEqual(params_by_name("momentum").rank_by, "mom3")
        self.assertEqual(params_by_name("momentum").tech, "sma150")
        self.assertEqual(parse_iso_date("2024-03-15"), date(2024, 3, 15))
        self.assertIsNone(parse_iso_date("nope"))

    def test_live_books_split_today_and_upcoming(self):
        from datetime import date

        from trading.services.fundamental_swing import FundParams, live_books

        cache = {
            "stocks": {
                "AAA": {
                    "info": {
                        "longName": "Aaa", "sector": "Industrials", "financialCurrency": "INR",
                        "returnOnEquity": 0.3, "profitMargins": 0.2, "operatingMargins": 0.22,
                        "revenueGrowth": 0.4, "earningsGrowth": 0.5, "trailingPE": 18,
                        "debtToEquity": 20, "currentRatio": 1.8, "currentPrice": 120,
                    },
                    "income": {
                        "2024-03-31": {"Total Revenue": 80, "Net Income": 12, "Operating Income": 14, "Diluted EPS": 8},
                        "2025-03-31": {"Total Revenue": 100, "Net Income": 18, "Operating Income": 20, "Diluted EPS": 12},
                    },
                    "balance": {
                        "2024-03-31": {"Stockholders Equity": 60, "Total Debt": 10, "Current Assets": 30, "Current Liabilities": 15},
                        "2025-03-31": {"Stockholders Equity": 70, "Total Debt": 10, "Current Assets": 32, "Current Liabilities": 15},
                    },
                    "cashflow": {},
                },
                "BBB": {
                    "info": {
                        "longName": "Bbb", "sector": "Industrials", "financialCurrency": "INR",
                        "returnOnEquity": 0.18, "profitMargins": 0.12, "operatingMargins": 0.14,
                        "revenueGrowth": 0.22, "earningsGrowth": 0.28, "trailingPE": 22,
                        "debtToEquity": 30, "currentRatio": 1.5, "currentPrice": 80,
                    },
                    "income": {
                        "2024-03-31": {"Total Revenue": 90, "Net Income": 14, "Operating Income": 16, "Diluted EPS": 7},
                        "2025-03-31": {"Total Revenue": 110, "Net Income": 20, "Operating Income": 22, "Diluted EPS": 10},
                    },
                    "balance": {
                        "2024-03-31": {"Stockholders Equity": 70, "Total Debt": 8, "Current Assets": 28, "Current Liabilities": 14},
                        "2025-03-31": {"Stockholders Equity": 80, "Total Debt": 8, "Current Assets": 30, "Current Liabilities": 14},
                    },
                    "cashflow": {},
                },
            }
        }
        params = FundParams(top_n=1, min_score=0, min_roe=0, min_profit_margin=0,
                            min_revenue_growth=0, min_earnings_growth=0, max_pe=100)
        snap = live_books(cache, ["AAA", "BBB"], params, date(2025, 9, 1), today=date(2026, 9, 16))
        self.assertEqual(snap["last_rebalance"], "2025-07-01")
        self.assertEqual(snap["next_rebalance"], "2026-07-01")
        self.assertFalse(snap["asof_is_today"])
        self.assertEqual(len(snap["today_book"]), 1)
        self.assertEqual(len(snap["upcoming_book"]), 1)
        self.assertIn(snap["today_book"][0]["symbol"], {"AAA", "BBB"})
        self.assertIn("Held on", snap["today_title"])
        self.assertIn("Next in line", snap["upcoming_title"])

    def test_custom_backtest_window(self):
        from datetime import date

        import pandas as pd

        from trading.services.fundamental_swing import FundParams, run_custom_backtest

        idx = pd.bdate_range("2024-06-28", "2025-07-05")
        a = pd.Series(100.0, index=idx)
        a.loc[idx >= "2025-06-30"] = 200.0
        b = pd.Series(50.0, index=idx)
        b.loc[idx >= "2025-06-30"] = 75.0
        nifty = pd.DataFrame({"close": a})
        cache = {
            "stocks": {
                "AAA": {
                    "info": {"longName": "Aaa", "sector": "Industrials", "financialCurrency": "INR"},
                    "income": {
                        "2024-03-31": {"Total Revenue": 80, "Net Income": 12, "Operating Income": 14, "Diluted EPS": 8},
                        "2025-03-31": {"Total Revenue": 100, "Net Income": 18, "Operating Income": 20, "Diluted EPS": 12},
                    },
                    "balance": {
                        "2024-03-31": {"Stockholders Equity": 60, "Total Debt": 10, "Current Assets": 30, "Current Liabilities": 15},
                        "2025-03-31": {"Stockholders Equity": 70, "Total Debt": 10, "Current Assets": 32, "Current Liabilities": 15},
                    },
                    "cashflow": {},
                },
                "BBB": {
                    "info": {"longName": "Bbb", "sector": "Industrials", "financialCurrency": "INR"},
                    "income": {
                        "2024-03-31": {"Total Revenue": 90, "Net Income": 14, "Operating Income": 16, "Diluted EPS": 7},
                        "2025-03-31": {"Total Revenue": 110, "Net Income": 20, "Operating Income": 22, "Diluted EPS": 10},
                    },
                    "balance": {
                        "2024-03-31": {"Stockholders Equity": 70, "Total Debt": 8, "Current Assets": 28, "Current Liabilities": 14},
                        "2025-03-31": {"Stockholders Equity": 80, "Total Debt": 8, "Current Assets": 30, "Current Liabilities": 14},
                    },
                    "cashflow": {},
                },
            }
        }
        params = FundParams(top_n=2, min_score=0, min_roe=0, min_profit_margin=0,
                            min_revenue_growth=0, min_earnings_growth=0, max_pe=100)
        with patch("trading.services.fundamental_swing.load_close_series",
                   return_value={"AAA": a, "BBB": b, "NIFTY50": a}), \
             patch("trading.services.market_data.load_price_dataframe", return_value=nifty):
            result = run_custom_backtest(
                date(2024, 7, 1), date(2025, 7, 1),
                params=params, capital=100_000, cache=cache, symbols=["AAA", "BBB"],
            )
        self.assertIsNone(result.get("error"))
        self.assertGreater(result["total_return_pct"], 70)
        self.assertLess(result["total_return_pct"], 80)
        self.assertTrue(result["rebalances"])

    def test_tech_gate_and_monthly_schedule(self):
        from datetime import date

        import pandas as pd

        from trading.services.fundamental_swing import (
            FundParams,
            apply_tech_rank,
            passes_tech_row,
            rebalance_dates,
        )

        idx = pd.bdate_range("2024-01-02", "2024-06-28")
        calendar = pd.DatetimeIndex(idx)
        months = rebalance_dates(date(2024, 1, 2), date(2024, 6, 28), calendar, "monthly")
        self.assertGreaterEqual(len(months), 6)
        self.assertEqual(months[0], date(2024, 1, 2))
        days_21 = rebalance_dates(date(2024, 1, 2), date(2024, 6, 28), calendar, "21d")
        self.assertGreaterEqual(len(days_21), 5)

        above = pd.Series({"close": 110.0, "sma150": 100.0, "sma50": 105.0, "ret63": 20.0})
        below = pd.Series({"close": 90.0, "sma150": 100.0, "sma50": 95.0, "ret63": 20.0})
        self.assertTrue(passes_tech_row(above, None, "sma150"))
        self.assertFalse(passes_tech_row(below, None, "sma150"))

        ranked = apply_tech_rank(
            [{"symbol": "AAA", "score": 80}, {"symbol": "BBB", "score": 90}],
            date(2024, 6, 28),
            FundParams(rank_by="score", tech="none", top_n=2),
            {},
        )
        self.assertEqual([p["symbol"] for p in ranked], ["BBB", "AAA"])


class FundamentalSwingViewTests(TestCase):
    def test_fundamentals_page_renders(self):
        payload = {
            "strategy": "Fundamental Compounders",
            "needs_fetch": True,
            "picks": [],
            "watch": [],
            "windows": [],
            "year_rows": [],
            "pack_runs": {},
            "doublers": [],
            "coverage": {"universe": 0, "cached": 0, "passed": 0, "fetched_at": None},
            "params": {
                "name": "Quality Growth", "top_n": 10, "min_roe": 15,
                "min_profit_margin": 8, "min_revenue_growth": 8,
                "min_earnings_growth": 10, "max_pe": 45, "max_de": 1, "min_score": 50,
            },
            "disclaimer": "",
            "blurb": "",
            "asof": "2026-09-05",
        }
        with patch("trading.services.fundamental_swing.load_page", return_value=payload):
            resp = self.client.get("/fundamentals/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Fundamental")
        self.assertContains(resp, "Yahoo fundamentals are not cached yet")

    def test_pack_dropdown_includes_momentum_swing_even_if_cache_is_stale(self):
        payload = {
            "strategy": "Fundamental Compounders",
            "needs_fetch": False,
            "picks": [],
            "watch": [],
            "today_book": [],
            "upcoming_book": [],
            "upcoming_bench": [],
            "book_diff": {"stay_count": 0, "enter_count": 0, "leave_count": 0},
            "windows": [],
            "year_rows": [],
            "pack_runs": {},
            "doublers": [],
            "coverage": {"universe": 0, "cached": 1, "passed": 0, "fetched_at": None},
            "params": {"name": "Aggressive Growth", "top_n": 3, "min_roe": 12,
                       "min_profit_margin": 5, "min_revenue_growth": 20,
                       "min_earnings_growth": 25, "max_pe": 60, "max_de": 1.5, "min_score": 40},
            "packs": [{"name": "Aggressive Growth", "top_n": 3}],
            "disclaimer": "",
            "blurb": "",
            "asof": "2026-09-16",
            "asof_is_today": True,
            "today_title": "Today’s book",
            "upcoming_title": "Upcoming",
        }
        with patch("trading.services.fundamental_swing.load_page", return_value=payload):
            resp = self.client.get("/fundamentals/")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Momentum Swing")
        self.assertContains(resp, 'value="Momentum Swing"')
        self.assertContains(resp, "Aggressive Growth")

    def test_fundamentals_page_shows_today_upcoming_and_custom_backtest(self):
        payload = {
            "strategy": "Fundamental Compounders",
            "needs_fetch": False,
            "picks": [],
            "watch": [],
            "today_book": [{
                "symbol": "OFSS", "name": "Oracle", "sector": "Technology", "score": 86,
                "roe": 50, "earnings_growth": 120, "revenue_growth": 69, "pe": 31,
                "status": "stay", "status_label": "Staying", "reasons": ["ROE 50%"],
                "since_entry_pct": 12.5, "price": 12000,
            }],
            "upcoming_book": [{
                "symbol": "SHRIRAMFIN", "name": "Shriram", "sector": "Financial Services", "score": 79,
                "roe": 18, "earnings_growth": 30, "revenue_growth": 43, "pe": 18,
                "status": "enter", "status_label": "New next July", "reasons": ["Earnings growth 30%"],
                "price": 1041,
            }],
            "upcoming_bench": [{"symbol": "MUTHOOTFIN", "score": 78}],
            "book_diff": {"staying": [], "entering": ["SHRIRAMFIN"], "leaving": ["OFSS"],
                          "stay_count": 0, "enter_count": 1, "leave_count": 1},
            "windows": [],
            "year_rows": [],
            "pack_runs": {},
            "doublers": [],
            "coverage": {"universe": 200, "cached": 200, "passed": 3, "fetched_at": "2026-09-04"},
            "params": {
                "name": "Aggressive Growth", "top_n": 3, "min_roe": 12,
                "min_profit_margin": 5, "min_revenue_growth": 20,
                "min_earnings_growth": 25, "max_pe": 60, "max_de": 1.5, "min_score": 40,
            },
            "disclaimer": "",
            "blurb": "12-month equal-weight hold",
            "asof": "2026-09-16",
            "asof_is_today": True,
            "asof_label": "16 Sep 2026",
            "today_title": "Today’s book",
            "today_sub": "Held since 1 Jul 2026",
            "upcoming_title": "Upcoming · 1 Jul 2027",
            "upcoming_sub": "Highest-ranked names now",
        }
        custom = {
            "title": "16 Sep 2025 → 16 Sep 2026",
            "params_name": "Aggressive Growth",
            "total_return_pct": 21.4,
            "cagr_pct": 21.4,
            "max_drawdown_pct": -11.2,
            "benchmark_pct": 10.0,
            "win_rate": 66.7,
            "hit_100": False,
            "beat_benchmark": True,
            "doublers": 0,
            "trades": 3,
            "final_equity": 121400,
            "rebalances": [{"date": "2025-07-01", "count": 1, "picks": [{"symbol": "OFSS", "score": 86}]}],
            "holdings_history": [{
                "symbol": "OFSS", "entry_date": "2025-07-01", "exit_date": "2026-07-01",
                "entry_px": 100, "exit_px": 121, "return_pct": 21.0,
            }],
            "equity_curve": [],
        }
        with patch("trading.services.fundamental_swing.load_page", return_value=payload), \
             patch("trading.services.fundamental_swing.run_custom_backtest", return_value=custom):
            resp = self.client.get("/fundamentals/?start=2025-09-16&end=2026-09-16&run=1")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Held now")
        self.assertContains(resp, "Next rebalance")
        self.assertContains(resp, "OFSS")
        self.assertContains(resp, "SHRIRAMFIN")
        self.assertContains(resp, "MUTHOOTFIN")
        self.assertContains(resp, "Run backtest")
        self.assertContains(resp, "Backtest 16 Sep 2025")
        self.assertContains(resp, "Beat Nifty")

    def test_custom_backtest_rejects_inverted_dates(self):
        payload = {
            "strategy": "Fundamental Compounders",
            "needs_fetch": False,
            "picks": [],
            "watch": [],
            "today_book": [],
            "upcoming_book": [],
            "upcoming_bench": [],
            "book_diff": {"stay_count": 0, "enter_count": 0, "leave_count": 0},
            "windows": [],
            "year_rows": [],
            "pack_runs": {},
            "doublers": [],
            "coverage": {"universe": 0, "cached": 1, "passed": 0, "fetched_at": None},
            "params": {
                "name": "Aggressive Growth", "top_n": 3, "min_roe": 12,
                "min_profit_margin": 5, "min_revenue_growth": 20,
                "min_earnings_growth": 25, "max_pe": 60, "max_de": 1.5, "min_score": 40,
            },
            "disclaimer": "",
            "blurb": "",
            "asof": "2026-09-16",
            "asof_is_today": True,
            "today_title": "Today’s book",
            "upcoming_title": "Upcoming",
        }
        with patch("trading.services.fundamental_swing.load_page", return_value=payload):
            resp = self.client.get("/fundamentals/?start=2026-09-16&end=2025-01-01&run=1")
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "End date must be after the start date")

