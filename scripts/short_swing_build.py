"""Precompute Swing Trading multi-year windows."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "confluence_trader.settings")

import django

django.setup()

from trading.services.short_swing import DEFAULT_PARAMS, build_results, save_results  # noqa: E402


def main() -> None:
    print("Building RS Pullback Swing results…", flush=True)
    payload = build_results(params=DEFAULT_PARAMS)
    path = save_results(payload)
    print(f"wrote {path}", flush=True)
    for w in payload.get("windows") or []:
        print(
            f"  {w.get('title')}: {w.get('total_return_pct')}%  "
            f"WR {w.get('win_rate')}%  DD {w.get('max_drawdown_pct')}%  "
            f"hold {w.get('avg_hold')}d  nifty {w.get('benchmark_pct')}%",
            flush=True,
        )
    print(f"live signals {len(payload.get('picks') or [])}  open {len(payload.get('open_book') or [])}", flush=True)


if __name__ == "__main__":
    main()
