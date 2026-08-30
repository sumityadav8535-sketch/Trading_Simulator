"""
Data models for Confluence Trend Pullback Swing Strategy.
"""
from django.db import models
from django.utils import timezone


class StrategyConfig(models.Model):
    """Singleton-style configuration for strategy thresholds."""

    name = models.CharField(max_length=100, default="Default")
    is_active = models.BooleanField(default=True)

    # Fundamental filters (optional in screener)
    enable_fundamental_filters = models.BooleanField(default=False)
    sales_growth_min = models.FloatField(default=8.0, help_text="YoY sales growth % min")
    profit_growth_min = models.FloatField(default=10.0, help_text="Profit growth % min")
    roe_min = models.FloatField(default=12.0, help_text="5-yr avg ROE % min")
    debt_equity_max = models.FloatField(default=1.0)
    peg_max = models.FloatField(default=2.0)

    # Technical filters
    adx_min = models.FloatField(default=20.0)
    adx_preferred = models.FloatField(default=25.0)
    rsi_low = models.FloatField(default=40.0)
    rsi_high = models.FloatField(default=65.0)
    volume_multiplier = models.FloatField(default=1.5, help_text="Entry vol vs 20d avg")
    ema_pullback_tolerance_pct = models.FloatField(default=2.0)
    fib_low = models.FloatField(default=0.382)
    fib_high = models.FloatField(default=0.618)

    # Risk management
    risk_pct = models.FloatField(default=2.0, help_text="Max risk per trade % of capital")
    min_risk_reward = models.FloatField(default=2.0)
    atr_sl_buffer = models.FloatField(default=0.5, help_text="ATR multiplier below SL")

    capital_default = models.DecimalField(max_digits=14, decimal_places=2, default=100_000)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Strategy Configuration"
        verbose_name_plural = "Strategy Configurations"

    def __str__(self) -> str:
        return f"{self.name} ({'active' if self.is_active else 'inactive'})"

    @classmethod
    def get_active(cls) -> "StrategyConfig":
        obj = cls.objects.filter(is_active=True).first()
        if obj is None:
            obj = cls.objects.create(name="Default")
        return obj


class Stock(models.Model):
    """NSE equity with optional fundamental metrics."""

    symbol = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=200, blank=True, default="")
    sector = models.CharField(max_length=100, blank=True, default="")
    is_nifty200 = models.BooleanField(default=False, db_index=True)
    is_nifty100 = models.BooleanField(default=False, db_index=True)
    is_nifty_smallcap250 = models.BooleanField(default=False, db_index=True)
    is_active = models.BooleanField(default=True)

    # Fundamental metrics (placeholder / manual / future API)
    sales_growth_yoy = models.FloatField(null=True, blank=True)
    profit_growth = models.FloatField(null=True, blank=True)
    roe_5yr_avg = models.FloatField(null=True, blank=True)
    debt_equity = models.FloatField(null=True, blank=True)
    peg = models.FloatField(null=True, blank=True)
    institutional_interest = models.CharField(
        max_length=20,
        blank=True,
        default="",
        help_text="stable / increasing / decreasing (placeholder)",
    )

    last_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    avg_volume_20d = models.BigIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["symbol"]

    def __str__(self) -> str:
        return self.symbol

    def passes_fundamental_filters(self, config: StrategyConfig) -> bool:
        if not config.enable_fundamental_filters:
            return True
        checks = []
        if self.sales_growth_yoy is not None:
            checks.append(self.sales_growth_yoy >= config.sales_growth_min)
        if self.profit_growth is not None:
            checks.append(self.profit_growth >= config.profit_growth_min)
        if self.roe_5yr_avg is not None:
            checks.append(self.roe_5yr_avg >= config.roe_min)
        if self.debt_equity is not None:
            checks.append(self.debt_equity <= config.debt_equity_max)
        if self.peg is not None:
            checks.append(self.peg <= config.peg_max)
        return all(checks) if checks else True


class DailyPrice(models.Model):
    """Daily OHLCV bar."""

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="prices")
    date = models.DateField(db_index=True)
    open = models.DecimalField(max_digits=12, decimal_places=2)
    high = models.DecimalField(max_digits=12, decimal_places=2)
    low = models.DecimalField(max_digits=12, decimal_places=2)
    close = models.DecimalField(max_digits=12, decimal_places=2)
    volume = models.BigIntegerField(default=0)

    class Meta:
        ordering = ["-date"]
        unique_together = [["stock", "date"]]
        indexes = [models.Index(fields=["stock", "date"])]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date}"


class Signal(models.Model):
    """Generated A+ setup signal."""

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="signals")
    date = models.DateField(db_index=True)
    confluence_score = models.PositiveSmallIntegerField(default=0)
    is_valid = models.BooleanField(default=False)
    entry_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    stop_loss = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    target_1r = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    target_2r = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    target_3r = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    risk_reward = models.DecimalField(max_digits=6, decimal_places=2, null=True, blank=True)
    position_size = models.PositiveIntegerField(default=0)
    capital_used = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    reasons = models.JSONField(default=list)
    rejection_reasons = models.JSONField(default=list)
    indicator_snapshot = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-confluence_score"]
        unique_together = [["stock", "date"]]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date} score={self.confluence_score}"


class WatchlistItem(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="watchlist_items")
    notes = models.CharField(max_length=255, blank=True, default="")
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-added_at"]

    def __str__(self) -> str:
        return self.stock_id


class TradeJournalEntry(models.Model):
    """Manual or signal-derived trade log."""

    STATUS_OPEN = "open"
    STATUS_CLOSED = "closed"
    STATUS_CHOICES = [(STATUS_OPEN, "Open"), (STATUS_CLOSED, "Closed")]

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name="journal_entries")
    signal = models.ForeignKey(Signal, null=True, blank=True, on_delete=models.SET_NULL)
    entry_date = models.DateField()
    exit_date = models.DateField(null=True, blank=True)
    entry_price = models.DecimalField(max_digits=12, decimal_places=2)
    exit_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    quantity = models.PositiveIntegerField()
    stop_loss = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    target = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_OPEN)
    notes = models.TextField(blank=True, default="")
    pnl = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-entry_date"]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.entry_date}"

    def compute_pnl(self) -> None:
        if self.exit_price is not None:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity


class BacktestRun(models.Model):
    """Stored backtest summary."""

    name = models.CharField(max_length=200)
    symbols = models.TextField(help_text="Comma-separated symbols")
    start_date = models.DateField()
    end_date = models.DateField()
    capital = models.DecimalField(max_digits=14, decimal_places=2)
    total_trades = models.PositiveIntegerField(default=0)
    win_rate = models.FloatField(default=0.0)
    profit_factor = models.FloatField(default=0.0)
    max_drawdown_pct = models.FloatField(default=0.0)
    avg_rr = models.FloatField(default=0.0)
    total_return_pct = models.FloatField(default=0.0)
    equity_curve = models.JSONField(default=list)
    trades = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


class PaperAccount(models.Model):
    """Fake-money paper trading account for forward-testing strategies."""

    name = models.CharField(max_length=100, default="Paper Account")
    is_active = models.BooleanField(default=True, db_index=True)
    auto_trade = models.BooleanField(
        default=True,
        help_text="When on, engine auto-places paper orders on active F&O signals",
    )
    starting_capital = models.DecimalField(max_digits=14, decimal_places=2, default=500_000)
    cash = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        default=500_000,
        help_text="Free cash (not blocked as margin)",
    )
    margin_blocked = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    risk_pct = models.FloatField(default=1.5, help_text="Risk per trade % of equity")
    max_trades_per_day = models.PositiveSmallIntegerField(default=4)
    instruments = models.JSONField(
        default=list,
        help_text='e.g. ["NIFTY", "BANKNIFTY"]',
    )
    realized_pnl = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    peak_equity = models.DecimalField(max_digits=14, decimal_places=2, default=500_000)
    last_tick_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_active", "-updated_at"]

    def __str__(self) -> str:
        return f"{self.name} (₹{self.cash:,.0f})"

    def save(self, *args, **kwargs):
        if not self.instruments:
            self.instruments = ["NIFTY", "BANKNIFTY"]
        super().save(*args, **kwargs)

    @property
    def equity(self):
        """Cash + margin blocked (MTM applied on close; open MTM computed in engine)."""
        return self.cash + self.margin_blocked

    @classmethod
    def get_active(cls) -> "PaperAccount":
        obj = cls.objects.filter(is_active=True).first()
        if obj is None:
            obj = cls.objects.create(
                name="Paper Account",
                starting_capital=500_000,
                cash=500_000,
                peak_equity=500_000,
                instruments=["NIFTY", "BANKNIFTY"],
                auto_trade=True,
            )
        return obj


class PaperPosition(models.Model):
    """Open paper F&O position (index futures proxy)."""

    SIDE_SHORT = "SHORT"
    SIDE_LONG = "LONG"
    SIDE_CHOICES = [(SIDE_SHORT, "Short"), (SIDE_LONG, "Long")]

    STATUS_OPEN = "open"
    STATUS_CLOSED = "closed"
    STATUS_CHOICES = [(STATUS_OPEN, "Open"), (STATUS_CLOSED, "Closed")]

    account = models.ForeignKey(PaperAccount, on_delete=models.CASCADE, related_name="positions")
    instrument = models.CharField(max_length=20, db_index=True)
    side = models.CharField(max_length=5, choices=SIDE_CHOICES, default=SIDE_SHORT)
    lots = models.PositiveIntegerField()
    lot_size = models.PositiveIntegerField()
    entry_price = models.DecimalField(max_digits=12, decimal_places=2)
    entry_time = models.DateTimeField()
    signal_bar_time = models.CharField(max_length=32, blank=True, default="")
    stop_loss = models.DecimalField(max_digits=12, decimal_places=2)
    target = models.DecimalField(max_digits=12, decimal_places=2)
    margin_blocked = models.DecimalField(max_digits=14, decimal_places=2)
    risk_pts = models.FloatField(default=0)
    ml_prob = models.FloatField(null=True, blank=True)
    strategy_name = models.CharField(max_length=100, blank=True, default="")
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_OPEN, db_index=True)
    notes = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-entry_time"]
        indexes = [models.Index(fields=["account", "status", "instrument"])]

    def __str__(self) -> str:
        return f"{self.side} {self.instrument} x{self.lots} @ {self.entry_price}"

    @property
    def quantity(self) -> int:
        return int(self.lots) * int(self.lot_size)


class PaperTrade(models.Model):
    """Closed paper trade with full P&L."""

    EXIT_SL = "stop_loss"
    EXIT_TARGET = "target"
    EXIT_FORCE = "force_exit"
    EXIT_MANUAL = "manual"
    EXIT_CHOICES = [
        (EXIT_SL, "Stop Loss"),
        (EXIT_TARGET, "Target"),
        (EXIT_FORCE, "Force Exit (EOD)"),
        (EXIT_MANUAL, "Manual"),
    ]

    account = models.ForeignKey(PaperAccount, on_delete=models.CASCADE, related_name="trades")
    position = models.OneToOneField(
        PaperPosition, null=True, blank=True, on_delete=models.SET_NULL, related_name="closed_trade"
    )
    instrument = models.CharField(max_length=20, db_index=True)
    side = models.CharField(max_length=5)
    lots = models.PositiveIntegerField()
    lot_size = models.PositiveIntegerField()
    entry_price = models.DecimalField(max_digits=12, decimal_places=2)
    exit_price = models.DecimalField(max_digits=12, decimal_places=2)
    entry_time = models.DateTimeField()
    exit_time = models.DateTimeField()
    stop_loss = models.DecimalField(max_digits=12, decimal_places=2)
    target = models.DecimalField(max_digits=12, decimal_places=2)
    exit_reason = models.CharField(max_length=20, choices=EXIT_CHOICES)
    pnl = models.DecimalField(max_digits=14, decimal_places=2)
    pnl_pts = models.FloatField(default=0)
    r_multiple = models.FloatField(default=0)
    risk_pts = models.FloatField(default=0)
    ml_prob = models.FloatField(null=True, blank=True)
    strategy_name = models.CharField(max_length=100, blank=True, default="")
    session_date = models.DateField(db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-exit_time"]

    def __str__(self) -> str:
        return f"{self.instrument} {self.side} PnL={self.pnl}"


class PaperEvent(models.Model):
    """Audit log for paper engine actions."""

    LEVEL_INFO = "info"
    LEVEL_TRADE = "trade"
    LEVEL_WARN = "warn"
    LEVEL_ERROR = "error"
    LEVEL_CHOICES = [
        (LEVEL_INFO, "Info"),
        (LEVEL_TRADE, "Trade"),
        (LEVEL_WARN, "Warn"),
        (LEVEL_ERROR, "Error"),
    ]

    account = models.ForeignKey(PaperAccount, on_delete=models.CASCADE, related_name="events")
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default=LEVEL_INFO)
    message = models.TextField()
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"[{self.level}] {self.message[:60]}"