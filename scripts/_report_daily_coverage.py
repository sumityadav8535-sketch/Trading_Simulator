import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django

django.setup()

from collections import Counter

from django.db.models import Count, Max, Min

from trading.constants import NIFTY50_SYMBOL
from trading.models import DailyPrice, Stock


def main():
    n200 = list(
        Stock.objects.filter(is_nifty200=True, is_active=True)
        .exclude(symbol=NIFTY50_SYMBOL)
        .values_list("symbol", flat=True)
    )
    agg = DailyPrice.objects.filter(stock_id__in=n200).aggregate(
        n=Count("id"), mn=Min("date"), mx=Max("date")
    )
    nifty = DailyPrice.objects.filter(stock_id=NIFTY50_SYMBOL).aggregate(
        n=Count("id"), mn=Min("date"), mx=Max("date")
    )
    rows = list(
        DailyPrice.objects.filter(stock_id__in=n200)
        .values("stock_id")
        .annotate(n=Count("id"), mn=Min("date"), mx=Max("date"))
    )
    starts = Counter(str(r["mn"].year) for r in rows if r["mn"])
    from2015 = sum(1 for r in rows if r["mn"] and r["mn"].year <= 2015)
    have = {r["stock_id"] for r in rows}
    missing = [s for s in n200 if s not in have]
    print(f"Nifty 200 symbols: {len(n200)}")
    print(f"Nifty 200 bars:    {agg['n']:,}   {agg['mn']} -> {agg['mx']}")
    print(f"NIFTY50:           {nifty['n']:,}   {nifty['mn']} -> {nifty['mx']}")
    print(f"First-bar years:   {dict(sorted(starts.items()))}")
    print(f"Stocks with 2015+ history: {from2015}/{len(rows)}")
    print(f"No prices: {missing or 'none'}")
    for s in ["TATAMOTORS", "TMPV", "TMCV", "ZOMATO", "ETERNAL", "RELIANCE", "TCS"]:
        r = next((x for x in rows if x["stock_id"] == s), None)
        print(f"  {s:12} {r}")


if __name__ == "__main__":
    main()
