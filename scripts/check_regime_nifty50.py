import os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")
import django; django.setup()
from trading.services.market_regime import get_market_regime_status, is_market_bullish, load_proxy_frames

for d in [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]:
    r = get_market_regime_status(eval_date=d)
    frames = load_proxy_frames()
    on = is_market_bullish(frames, d)
    print(f"{d}: ON={on} | {r['message']}")
    if r.get("index"):
        i = r["index"]
        print(f"  close={i['close']} 50EMA={i['ema_50']} 200EMA={i.get('ema_200')}")