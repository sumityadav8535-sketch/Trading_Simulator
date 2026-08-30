import json
from collections import defaultdict
from pathlib import Path

p = json.loads(Path("data/intraday_scale_fade.json").read_text(encoding="utf-8"))
print("hit_100", p["hit_100"], "n_hits", p["n_hits"])
hits = p["hits"]
by = defaultdict(list)
for r in hits:
    pr = r["params"]
    key = (pr.get("leverage"), pr.get("max_deploy"), pr.get("family"))
    by[key].append(r)
print("unique lev/deploy/family among first hits:")
for k, v in sorted(by.items(), key=lambda kv: -kv[1][0]["total_return_pct"])[:25]:
    r = v[0]
    print(
        f"  lev={k[0]:4g} d={k[1]:4g} {str(k[2])[:22]:22s} "
        f"ret={r['total_return_pct']:7.1f} DD={r['max_dd_pct']:5.1f} "
        f"WR={r['win_rate']:5.1f}"
    )

lev5 = [r for r in hits if r["params"].get("leverage") == 5]
print("lev5 hits", len(lev5))
if lev5:
    r = max(lev5, key=lambda x: x["total_return_pct"])
    print("best lev5 hit", r["name"], r["total_return_pct"], r["max_dd_pct"], r["months"])

# lowest DD among 100% hits
best_dd = min(hits, key=lambda x: x["max_dd_pct"])
print("lowest DD 100% hit", best_dd["name"], best_dd["total_return_pct"], "DD", best_dd["max_dd_pct"])
print("  months", best_dd["months"])
print("  params", best_dd["params"])

# lev 5 from top even if <100
print("winner overall", p["winner"]["name"], p["winner"]["total_return_pct"], p["winner"]["max_dd_pct"])
print("winner months", p["winner"]["months"])
print("winner params", p["winner"]["params"])
