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
        self.assertEqual(ok["target"], 97.97)  # 1.0% from entry, not prior close
        self.assertEqual(ok["stop"], round(97.0 - 0.6 * 0.8, 2))
        self.assertGreater(ok["qty"], 0)
        self.assertEqual(ok["entry_time"], "09:30")

        hot = build_setup("RELIANCE", 97.0, 100.0, 0.8, 68.0, date(2026, 8, 26))
        self.assertIsNotNone(hot)
        crash = build_setup("RELIANCE", 93.0, 100.0, 0.8, 52.0, date(2026, 8, 26))
        self.assertIsNone(crash)
        self.assertIsNone(build_setup("RELIANCE", 97.0, 100.0, 0.8, 40.0, date(2026, 8, 26)))
        self.assertIsNone(build_setup("RELIANCE", 99.0, 100.0, 0.8, 52.0, date(2026, 8, 26)))
        self.assertIsNone(build_setup("RELIANCE", 103.0, 100.0, 0.8, 52.0, date(2026, 8, 26)))

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
        self.assertEqual(trade["target"], 97.97)
        self.assertEqual(trade["stop"], 96.52)
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
        ])
        today_df = day_bars(today, [
            (9, 15, 97.0, 97.3, 96.95, 97.05),
            (9, 30, 97.10, 97.3, 96.95, 97.1),
            (10, 0, 97.1, 97.8, 97.0, 97.6),
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

