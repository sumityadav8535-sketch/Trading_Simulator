"""Report DailyPrice (NSE daily chart) coverage."""
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from django.db.models import Count, Max, Min

from trading.models import DailyPrice, Stock


def main():
    agg = DailyPrice.objects.aggregate(mn=Min("date"), mx=Max("date"), n=Count("id"))
    print("=== NSE DailyPrice (daily charts) ===")
    print(f"Total bars:           {agg['n']:,}")
    print(f"Date range (global):  {agg['mn']}  →  {agg['mx']}")
    print(f"Stocks in master:     {Stock.objects.count()}")
    print(f"Stocks with prices:   {DailyPrice.objects.values('stock_id').distinct().count()}")

    per = list(
        DailyPrice.objects.values("stock__symbol")
        .annotate(n=Count("id"), mn=Min("date"), mx=Max("date"))
        .order_by("n")
    )
    if per:
        ns = [r["n"] for r in per]
        print(f"Bars per stock:       min={min(ns)}  median={sorted(ns)[len(ns)//2]}  max={max(ns)}")
        print(f"Fewest:  {per[0]['stock__symbol']}  n={per[0]['n']}  {per[0]['mn']}→{per[0]['mx']}")
        print(f"Most:    {per[-1]['stock__symbol']}  n={per[-1]['n']}  {per[-1]['mn']}→{per[-1]['mx']}")

    print("\n=== Key symbols ===")
    for sym in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY50", "NIFTY", "^NSEI"]:
        s = Stock.objects.filter(symbol=sym).first()
        if not s:
            alt = list(Stock.objects.filter(symbol__icontains=sym.replace("^", "")).values_list("symbol", flat=True)[:5])
            print(f"{sym}: not found (alts={alt})")
            continue
        a = DailyPrice.objects.filter(stock=s).aggregate(mn=Min("date"), mx=Max("date"), n=Count("id"))
        print(f"{s.symbol:12}  bars={a['n']:5}  {a['mn']} → {a['mx']}")

    start_y = Counter()
    end_y = Counter()
    for r in DailyPrice.objects.values("stock_id").annotate(mn=Min("date"), mx=Max("date")):
        if r["mn"]:
            start_y[r["mn"].year] += 1
        if r["mx"]:
            end_y[r["mx"].year] += 1
    print("\n=== Stock start-year distribution ===")
    for y, c in sorted(start_y.items()):
        print(f"  {y}: {c} stocks first bar")
    print("=== Stock end-year distribution ===")
    for y, c in sorted(end_y.items()):
        print(f"  {y}: {c} stocks last bar")


if __name__ == "__main__":
    main()
