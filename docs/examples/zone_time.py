#!/usr/bin/env python3
"""Total time each occupant spent inside each zone, from `film inside` CSV.

    python3 -I hms_cmd.py match.mov "film inside label=zone_a" | python3 -I zone_time.py
    python3 -I zone_time.py inside.csv

Also prints the gap between leaving one zone and entering the next, per occupant.
"""
import csv, sys
from collections import defaultdict

rows = csv.DictReader(open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin)
open_at = {}                       # (zone_id, occupant_id) -> (seconds of the ENTER, who, zone)
total = defaultdict(float)         # (occupant, zone) -> seconds inside
last_exit = {}                     # occupant -> (zone, seconds of the EXIT)
last_t = 0.0
for r in rows:
    if r["event"] not in ("ENTER", "EXIT"):
        continue                   # skip INSIDE snapshot rows
    t, z, who = float(r["seconds"]), r["zone_label"] or r["zone_id"], r["name"] or r["occupant_id"]
    key, last_t = (r["zone_id"], r["occupant_id"]), t
    if r["event"] == "ENTER":
        open_at[key] = (t, who, z)
        if who in last_exit and last_exit[who][0] != z:
            print(f"{who}: {last_exit[who][0]} -> {z} in {t - last_exit[who][1]:.2f}s")
    elif key in open_at:
        total[(who, z)] += t - open_at.pop(key)[0]
        last_exit[who] = (z, t)
for t0, who, z in open_at.values():             # still inside at the last row: counted up to it
    total[(who, z)] += last_t - t0
for (who, z), s in sorted(total.items()):
    print(f"{who} spent {s:.2f}s inside {z}")
