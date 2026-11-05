#!/usr/bin/env python3
"""What happened in a live match, from kak1's decision logs.

    python3 live_report.py /tmp/kak1logs            # every kak1_*.json in the folder
    python3 live_report.py a.json b.json --match -1  # only the last match

Input: `aws logs filter-log-events --output json` exports, one per agent (kak1_gk.json,
kak1_def.json, ...), or files with one log line per line. kak1 writes one JSON line per
decision (kaklib/fast_agent.py): pid, src, ms, cmd, params, tick, t, me, ball, owner,
team, score. Positions are on the LIVE pitch.

Per match it prints: score and side, calls per player (missed ticks), where the decision
came from (rule / llm / fallback_* / override_*), commands per player, latency, the
position extremes (to check the pitch size), and each goal with the ticks before it.

Matches are split where the clock goes back (a new match restarts gameTime) or the logs
pause for more than 60 s.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

ROLE = {0: "GK", 1: "DEF", 2: "MIDL", 3: "MIDR", 4: "FWD"}   # 1-2-1 (kaklib FORMATION default)
GAP_MS = 60_000


def _records(path):
    """Decision records from one file: a CloudWatch export or raw lines."""
    text = open(path).read()
    events = []
    try:
        doc = json.loads(text)
        events = [(e.get("timestamp", 0), e.get("message", "")) for e in doc.get("events", [])]
    except json.JSONDecodeError:
        events = [(0, line) for line in text.splitlines()]
    out = []
    for ts, msg in events:
        i = msg.find("{")
        if i < 0:
            continue
        try:
            r = json.loads(msg[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "pid" in r and "src" in r:
            r["_ts"] = ts
            out.append(r)
        elif isinstance(r, dict) and "payload_check" in r:
            out.append({"_ts": ts, "_payload": r["payload_check"]})
    return out


def _split(recs):
    """Group records into matches by time order."""
    recs = sorted(recs, key=lambda r: (r["_ts"], r.get("t") or 0))
    matches, cur, last_ts, last_t = [], [], None, None
    for r in recs:
        t = r.get("t")
        new = (last_ts is not None and r["_ts"] and last_ts and r["_ts"] - last_ts > GAP_MS) or \
              (last_t is not None and t is not None and t + 10 < last_t)
        if new and cur:
            matches.append(cur)
            cur = []
        cur.append(r)
        last_ts = r["_ts"] or last_ts
        if t is not None:
            last_t = t
    if cur:
        matches.append(cur)
    return matches


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0


def _fmt_pos(p):
    return "-" if not p else f"({p[0]:+.2f},{p[1]:+.2f})"


def report(match, idx):
    dec = [r for r in match if "pid" in r]
    payloads = [r["_payload"] for r in match if "_payload" in r]
    if not dec:
        return
    team = next((r["team"] for r in dec if r.get("team") is not None), None)
    side = {0: "HOME (attacks +x)", 1: "AWAY (attacks -x)"}.get(team, "?")
    last = max(dec, key=lambda r: (r.get("t") or 0, r["_ts"]))
    sc = last.get("score") or {}
    us, them = (sc.get("home", 0), sc.get("away", 0)) if team == 0 else (sc.get("away", 0), sc.get("home", 0))
    t_max = max((r.get("t") or 0) for r in dec)
    print(f"\n=== Match {idx}: kak1 {us} - {them}, we were {side}, clock reached {t_max} s ===")

    by_pid = defaultdict(list)
    for r in dec:
        by_pid[r["pid"]].append(r)

    print("\nCalls and sources per player")
    srcs = sorted({r["src"].split(":")[0] for r in dec})
    print(f"  {'pid':5s}{'calls':>6s}{'ticks':>7s}  " + "".join(f"{s:>18s}" for s in srcs))
    for pid in sorted(by_pid):
        rs = by_pid[pid]
        ticks = {r.get("tick") for r in rs if r.get("tick") is not None}
        cnt = Counter(r["src"].split(":")[0] for r in rs)
        print(f"  {pid}{ROLE.get(pid, ''):>4s}{len(rs):6d}{len(ticks):7d}  " +
              "".join(f"{cnt.get(s, 0):18d}" for s in srcs))
    errs = Counter(r["src"] for r in dec if r["src"].startswith("fallback_error"))
    if errs:
        print("  errors: " + ", ".join(f"{k} x{v}" for k, v in errs.most_common()))

    print("\nCommands per player")
    for pid in sorted(by_pid):
        cnt = Counter(r.get("cmd") for r in by_pid[pid])
        print(f"  {pid}{ROLE.get(pid, ''):>4s}  " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()))

    ms = [r.get("ms") for r in dec if isinstance(r.get("ms"), (int, float))]
    if ms:
        print(f"\nDecision time in our code (ms): p50 {_pct(ms, 50)}, p95 {_pct(ms, 95)}, max {max(ms)}")
        llm = [r["ms"] for r in dec if r["src"] == "llm"]
        if llm:
            print(f"  model turns only: p50 {_pct(llm, 50)}, p95 {_pct(llm, 95)}")

    xs = [p for r in dec for p in (r.get("me"), r.get("ball")) if p]
    if xs:
        bx = [r["ball"] for r in dec if r.get("ball")]
        print("\nPosition extremes (live units)")
        print(f"  everyone + ball: x {min(p[0] for p in xs):+.2f}..{max(p[0] for p in xs):+.2f}, "
              f"y {min(p[1] for p in xs):+.2f}..{max(p[1] for p in xs):+.2f}")
        if bx:
            print(f"  ball:            x {min(p[0] for p in bx):+.2f}..{max(p[0] for p in bx):+.2f}, "
                  f"y {min(p[1] for p in bx):+.2f}..{max(p[1] for p in bx):+.2f}")
        gk = [r["me"] for r in by_pid.get(0, []) if r.get("me")]
        if gk:
            print(f"  our GK:          x {min(p[0] for p in gk):+.2f}..{max(p[0] for p in gk):+.2f}, "
                  f"median {statistics.median(p[0] for p in gk):+.2f}")

    print("\nGoals (the 3 decision ticks before each; positions are live units)")
    goals, prev = [], None
    for r in sorted(dec, key=lambda r: (r.get("t") or 0, r["_ts"])):
        s = r.get("score")
        if not s:
            continue
        key = (s.get("home", 0), s.get("away", 0))
        if prev is not None and key != prev:
            home_scored = key[0] > prev[0]
            ours = home_scored == (team == 0)
            goals.append((r.get("t"), "FOR" if ours else "AGAINST", prev, key))
        prev = key
    if not goals:
        print("  none")
    ticks = sorted({(r.get("t") or 0) for r in dec})
    for t, who, before, after in goals:
        print(f"\n  t={t} s  goal {who}  {before[0]}-{before[1]} -> {after[0]}-{after[1]}")
        earlier = [x for x in ticks if x < t][-3:]
        for tt in earlier:
            rows = sorted((r for r in dec if (r.get("t") or 0) == tt), key=lambda r: r["pid"])
            if not rows:
                continue
            ball = rows[0].get("ball")
            owner = rows[0].get("owner")
            print(f"    t={tt:<4} ball {_fmt_pos(ball)} owner {owner}")
            for r in rows:
                p = r.get("params") or {}
                tgt = ""
                if "target_x" in p:
                    tgt = f" -> ({p['target_x']:+.2f},{p['target_y']:+.2f})"
                elif "target_player_id" in p:
                    tgt = f" -> {p['target_player_id']}"
                print(f"      {r['pid']}{ROLE.get(r['pid'], ''):>4s} at {_fmt_pos(r.get('me'))}  "
                      f"{r.get('cmd')}{tgt}  [{r['src']}]")

    if payloads:
        print("\nPayload checks: " + json.dumps(payloads[0])[:600])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="folders or files")
    ap.add_argument("--match", type=int, help="only this match (0-based; -1 = last)")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    recs = [r for f in files for r in _records(f)]
    if not recs:
        sys.exit("no kak1 decision lines found")
    matches = _split(recs)
    print(f"{len(recs)} records from {len(files)} files, {len(matches)} match(es)")
    chosen = list(enumerate(matches))
    if args.match is not None:
        chosen = [chosen[args.match]]
    for i, m in chosen:
        report(m, i)


if __name__ == "__main__":
    main()
