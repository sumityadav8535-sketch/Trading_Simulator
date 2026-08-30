"""Run Stage Analysis 2.0 backtest on Nifty 200."""
from __future__ import annotations

from datetime import date, timedelta

from django.core.management.base import BaseCommand

from stage_analysis_v2.services.backtester import (
    DEFAULT_ENTRY_ON,
    DEFAULT_ENTRY_STAGE,
    DEFAULT_EXIT_MODE,
    DEFAULT_MA_TYPE,
    DEFAULT_MAX_HOLD_DAYS,
    DEFAULT_STOP_MA_MULT,
    DEFAULT_TARGET_RR,
    DEFAULT_TRAIL_MA_MULT,
    ENTRY_ON_CHOICES,
    EntryFilters,
    EXIT_MODE_LABELS,
    MA_COND_NONE,
    VALID_ENTRY_ON,
    VALID_ENTRY_STAGES,
    VALID_EXIT_MODES,
    run_stage_v2_backtest,
)
from stage_analysis_v2.services.cup_breakout import (
    CUP_ENTRY_NEXT_OPEN,
    CUP_EXIT_EMA20,
    VALID_CUP_ENTRIES,
    VALID_CUP_EXITS,
    CupParams,
    run_cup_breakout_backtest,
)
from stage_analysis_v2.services.strategy_catalog import (
    STRATEGY_CUP,
    VALID_STRATEGIES,
    is_cup_strategy,
    is_supertrend_strategy,
    is_union_strategy,
    normalize_strategy,
)
from stage_analysis_v2.services.st_union_swing import run_st_union_backtest
from stage_analysis_v2.services.supertrend_swing import run_supertrend_swing_backtest
from stage_analysis_v2.services.tech_filters import (
    DEFAULT_TECH_FILTER,
    TECH_FILTER_LABELS,
    VALID_TECH_FILTERS,
)
from trading.services.market_data import resolve_universe_symbols


class Command(BaseCommand):
    help = "Backtest Stage Analysis 2.0 / Supertrend / Cup Breakout on Nifty 200"

    def add_arguments(self, parser):
        parser.add_argument(
            "--strategy",
            type=str,
            default="stage_v2",
            choices=sorted(VALID_STRATEGIES),
            help="Engine: stage_v2 (default), cup_breakout, st_pullback_quality, …",
        )
        parser.add_argument("--years", type=float, default=1.0, help="Lookback years (default 1)")
        parser.add_argument(
            "--universe",
            type=str,
            default="nifty200",
            help="nifty200 (default) or nifty_smallcap250",
        )
        parser.add_argument("--capital", type=float, default=1_000_000, help="Starting capital (INR)")
        parser.add_argument("--min-quality", type=int, default=0, help="Min quality score (0-100)")
        parser.add_argument("--market-filter", action="store_true", help="Only trade when Nifty is Stage 1/2")
        parser.add_argument("--strict", action="store_true", help="Strict V2: quality>=75, market filter on")
        parser.add_argument(
            "--exit-mode",
            type=str,
            default=DEFAULT_EXIT_MODE,
            choices=sorted(VALID_EXIT_MODES),
            help=(
                "Exit rule: stage_4_only (default), trail_ma_s4, "
                "stage_3_4 (legacy), no_stage"
            ),
        )
        parser.add_argument(
            "--tech-filter",
            type=str,
            default=DEFAULT_TECH_FILTER,
            choices=sorted(VALID_TECH_FILTERS),
            help=(
                "Tech confirmation: daily_mtf (default), not_extended, "
                "ema_stack, bb_mid, confluence, none, …"
            ),
        )
        parser.add_argument(
            "--entry-stage",
            type=int,
            default=DEFAULT_ENTRY_STAGE,
            choices=sorted(VALID_ENTRY_STAGES),
            help="Weekly stage to enter: 1, 2 (default), 3, or 4",
        )
        parser.add_argument(
            "--entry-on",
            type=str,
            default=DEFAULT_ENTRY_ON,
            choices=sorted(VALID_ENTRY_ON),
            help="transition (default) or in_stage",
        )
        parser.add_argument("--min-price", type=float, default=0.0, help="Min stock price (0=off)")
        parser.add_argument("--max-price", type=float, default=0.0, help="Max stock price (0=off)")
        parser.add_argument("--min-volume", type=float, default=0.0, help="Min absolute volume (0=off)")
        parser.add_argument(
            "--min-volume-ratio",
            type=float,
            default=0.0,
            help="Min volume vs 20d avg e.g. 1.5 (0=off)",
        )
        parser.add_argument(
            "--ma-condition",
            type=str,
            default=MA_COND_NONE,
            choices=["none", "above", "below"],
            help="Price vs MA filter",
        )
        parser.add_argument("--ma-period", type=int, default=50, help="MA period when MA filter on")
        parser.add_argument(
            "--ma-type",
            type=str,
            default=DEFAULT_MA_TYPE,
            choices=["sma", "ema"],
            help="MA type",
        )
        parser.add_argument(
            "--target-rr",
            type=float,
            default=DEFAULT_TARGET_RR,
            help="Target reward:risk (default 2.5)",
        )
        parser.add_argument(
            "--max-hold-days",
            type=int,
            default=DEFAULT_MAX_HOLD_DAYS,
            help="Max holding trading days (default 65)",
        )
        parser.add_argument(
            "--stop-ma-mult",
            type=float,
            default=DEFAULT_STOP_MA_MULT,
            help="Stop = 30w MA × this (default 0.95)",
        )
        parser.add_argument(
            "--trail-ma-mult",
            type=float,
            default=DEFAULT_TRAIL_MA_MULT,
            help="Trail stop = 30w MA × this when trail mode (default 0.98)",
        )
        parser.add_argument("--top", type=int, default=15, help="Show top N trades by P&L")
        parser.add_argument("--risk-pct", type=float, default=None, help="Risk percent of equity per trade")
        parser.add_argument("--cooldown-days", type=int, default=None, help="Symbol cooldown after exit")
        parser.add_argument(
            "--entry-mode",
            type=str,
            default=CUP_ENTRY_NEXT_OPEN,
            choices=sorted(VALID_CUP_ENTRIES),
            help="Cup entry: next_open (default) or retest",
        )
        parser.add_argument(
            "--cup-exit",
            type=str,
            default=CUP_EXIT_EMA20,
            choices=sorted(VALID_CUP_EXITS),
            help="Cup exit: ema20_trail (default), target_r, measured_move",
        )

    def handle(self, *args, **options):
        end = date.today()
        start = end - timedelta(days=int(options["years"] * 365))
        min_q = options["min_quality"]
        mkt = options["market_filter"]
        exit_mode = options["exit_mode"]
        tech_filter = options["tech_filter"]
        entry_stage = options["entry_stage"]
        entry_on = options["entry_on"]
        if options["strict"]:
            min_q = 75
            mkt = True

        filters = EntryFilters(
            min_price=float(options["min_price"] or 0),
            max_price=float(options["max_price"] or 0),
            min_volume=float(options["min_volume"] or 0),
            min_volume_ratio=float(options["min_volume_ratio"] or 0),
            ma_period=int(options["ma_period"] or 0),
            ma_type=options["ma_type"] or DEFAULT_MA_TYPE,
            ma_condition=options["ma_condition"] or MA_COND_NONE,
        )

        symbols = resolve_universe_symbols(options.get("universe") or "nifty200")
        strategy_id = normalize_strategy(options.get("strategy"))
        self.stdout.write(
            f"{strategy_id} backtest | {start} → {end} | "
            f"{len(symbols)} symbols | capital Rs {options['capital']:,.0f}"
        )
        if is_cup_strategy(strategy_id):
            self._run_cup(symbols, start, end, options)
            return
        if is_union_strategy(strategy_id):
            r = run_st_union_backtest(
                symbols=symbols,
                start_date=start,
                end_date=end,
                capital=options["capital"],
                risk_pct=options.get("risk_pct"),
                max_hold_days=options.get("max_hold_days"),
                cooldown_days=options.get("cooldown_days"),
            )
            self._print_result(r, options["top"])
            return
        if is_supertrend_strategy(strategy_id):
            r = run_supertrend_swing_backtest(
                symbols=symbols,
                start_date=start,
                end_date=end,
                capital=options["capital"],
                strategy_id=strategy_id,
                risk_pct=options.get("risk_pct"),
                max_hold_days=options.get("max_hold_days"),
                cooldown_days=options.get("cooldown_days"),
            )
            self._print_result(r, options["top"])
            return
        if min_q:
            self.stdout.write(f"  Min quality score: {min_q}")
        if mkt:
            self.stdout.write("  Market filter: Nifty Stage 1/2 only")
        on_label = dict(ENTRY_ON_CHOICES).get(entry_on, entry_on)
        self.stdout.write(f"  Entry stage: {entry_stage} ({on_label})")
        self.stdout.write(f"  Stock filters: {filters.active_summary()}")
        self.stdout.write(
            f"  Exit mode: {exit_mode} ({EXIT_MODE_LABELS.get(exit_mode, exit_mode)})"
        )
        self.stdout.write(
            f"  Target: {options['target_rr']}R · max hold {options['max_hold_days']}d · "
            f"stop MA×{options['stop_ma_mult']} · trail MA×{options['trail_ma_mult']}"
        )
        self.stdout.write(
            f"  Tech filter: {tech_filter} ({TECH_FILTER_LABELS.get(tech_filter, tech_filter)})"
        )

        r = run_stage_v2_backtest(
            symbols=symbols,
            start_date=start,
            end_date=end,
            capital=options["capital"],
            min_quality_score=min_q,
            market_filter=mkt,
            exit_mode=exit_mode,
            tech_filter=tech_filter,
            entry_stage=entry_stage,
            entry_on=entry_on,
            entry_filters=filters,
            target_rr=options["target_rr"],
            max_hold_days=options["max_hold_days"],
            stop_ma_mult=options["stop_ma_mult"],
            trail_ma_mult=options["trail_ma_mult"],
        )
        self._print_result(r, options["top"])

    def _run_cup(self, symbols, start, end, options):
        from stage_analysis_v2.services.backtester import DEFAULT_MAX_HOLD_DAYS, DEFAULT_TARGET_RR

        hold = options.get("max_hold_days")
        if hold == DEFAULT_MAX_HOLD_DAYS:
            hold = None
        rr = options.get("target_rr")
        if rr == DEFAULT_TARGET_RR:
            rr = 2.0
        params = CupParams(
            entry_mode=options.get("entry_mode") or CUP_ENTRY_NEXT_OPEN,
            cup_exit_mode=options.get("cup_exit") or CUP_EXIT_EMA20,
            target_rr=float(rr),
        )
        self.stdout.write(
            f"  Cup {params.cup_min_days}–{params.cup_max_days}d · "
            f"depth {params.min_depth_pct:g}–{params.max_depth_pct:g}% · "
            f"entry {params.entry_mode} · exit {params.cup_exit_mode}"
        )
        r = run_cup_breakout_backtest(
            symbols=symbols,
            start_date=start,
            end_date=end,
            capital=options["capital"],
            strategy_id=STRATEGY_CUP,
            risk_pct=options.get("risk_pct"),
            max_hold_days=hold,
            cooldown_days=options.get("cooldown_days"),
            params=params,
        )
        self._print_result(r, options["top"], cup=True)

    def _print_result(self, r, top: int, cup: bool = False):
        from stage_analysis_v2.services.backtester import fill_performance_metrics

        fill_performance_metrics(r)
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=== RESULTS ==="))
        self.stdout.write(f"Strategy:           {r.strategy_name}")
        self.stdout.write(f"Stocks scanned:     {r.stocks_scanned}")
        self.stdout.write(f"Signals:            {r.stage2_entries}")
        self.stdout.write(f"Total trades:       {r.total_trades}")
        self.stdout.write(f"Win rate:           {r.win_rate}%")
        self.stdout.write(f"CAGR:               {r.cagr_pct}%")
        self.stdout.write(f"Profit factor:      {r.profit_factor}")
        self.stdout.write(f"Average trade:      Rs {r.avg_trade:,.0f}")
        self.stdout.write(f"Max drawdown:       {r.max_drawdown_pct}%")
        self.stdout.write(f"Sharpe:             {r.sharpe}")
        self.stdout.write(f"Avg hold (days):    {r.avg_hold_days}")
        self.stdout.write(f"Total return:       {r.total_return_pct}%")
        self.stdout.write(f"Final capital:      Rs {r.final_capital:,.0f}")
        self.stdout.write(f"Avg R achieved:     {r.avg_rr}")
        self.stdout.write(f"Exit breakdown:     {r.exit_breakdown}")

        if r.monthly_returns:
            self.stdout.write("")
            self.stdout.write("Monthly P&L:")
            for m in r.monthly_returns:
                sign = "+" if m["pnl"] >= 0 else ""
                self.stdout.write(f"  {m['month']}: {sign}{m['pnl']:,.0f} ({sign}{m['return_pct']}%)")

        if r.trades and top:
            self.stdout.write("")
            self.stdout.write(f"Top {top} winners:")
            for t in sorted(r.trades, key=lambda x: x.pnl, reverse=True)[:top]:
                extra = ""
                if cup and t.setup:
                    extra = (
                        f" cup {t.setup.get('cup_duration')}d "
                        f"depth {t.setup.get('cup_depth_pct')}% "
                        f"vol {t.setup.get('volume_multiple')}x"
                    )
                self.stdout.write(
                    f"  {t.symbol} {t.entry_date}→{t.exit_date} "
                    f"Q{t.quality_score} RS{t.rs_rating:.0f} "
                    f"PnL {t.pnl:+,.0f} ({t.exit_reason}){extra}"
                )
            self.stdout.write("")
            self.stdout.write(f"Top {top} losers:")
            for t in sorted(r.trades, key=lambda x: x.pnl)[:top]:
                self.stdout.write(
                    f"  {t.symbol} {t.entry_date}→{t.exit_date} "
                    f"Q{t.quality_score} RS{t.rs_rating:.0f} "
                    f"PnL {t.pnl:+,.0f} ({t.exit_reason})"
                )