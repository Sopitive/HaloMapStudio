#!/usr/bin/env python3
"""Consume `hms-app --film-follow` output: one line per zone enter/exit, and a running total of
the time each occupant has spent inside each watched zone.

    hms-app --film-follow match.temp --watch-label zone_a --watch-label zone_b \
            --watch-radius 6 2>follow.log | python3 -I follow_zones.py
"""
import json, sys

seen = set()                       # de-duplicate by event id (safe across restarts with --since)
since = {}                         # (zone_id, occupant_id) -> seconds of the ENTER
total = {}                         # (name, zone) -> seconds inside (closed visits only)
for line in sys.stdin:
    e = json.loads(line)
    if e["id"] in seen:
        continue
    seen.add(e["id"])
    t = e["type"]
    if t == "reset":               # a different match replaced the file: start over
        since.clear(); total.clear(); seen.clear()
        print("-- reset:", e["reason"])
    elif t in ("zone_enter", "zone_exit"):
        zone = e["zone_label"] or e["zone_id"]
        who = e["name"] or f'{e["occupant_type"]} {e["occupant_id"]}'
        key = (e["zone_id"], e["occupant_id"])
        print(f'{e["seconds"]:9.2f}s  {who:<16} {t[5:]:<5} {zone}', flush=True)
        if t == "zone_enter":
            since[key] = e["seconds"]
        elif key in since:
            k = (who, zone)
            total[k] = total.get(k, 0.0) + e["seconds"] - since.pop(key)
for (who, zone), s in sorted(total.items()):
    print(f"{who} spent {s:.2f}s inside {zone}")
