import os, sys, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()
from trading.models import Stock, DailyPrice
from trading.services.market_data import load_price_dataframe

# DB stocks with NIFTY in name
print("DB NIFTY symbols:", list(Stock.objects.filter(symbol__icontains='NIFTY').values_list('symbol', flat=True)))
for sym in ['NIFTY50', 'NIFTY 50', 'NIFTY_50']:
    c = DailyPrice.objects.filter(stock_id=sym).count()
    if c:
        print(f"{sym}: {c} bars")

# Check sqlite source
p = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nse_data.sqlite')
if os.path.exists(p):
    con = sqlite3.connect(p)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("sqlite tables:", [r[0] for r in cur.fetchall()])
    for q in ["SELECT DISTINCT symbol FROM stocks WHERE symbol LIKE '%NIFTY%' LIMIT 20",
              "SELECT symbol FROM stocks WHERE symbol LIKE '%NIFTY%' LIMIT 20"]:
        try:
            cur.execute(q)
            print("sqlite nifty:", [r[0] for r in cur.fetchall()])
            break
        except Exception as e:
            print("query failed:", e)
    con.close()