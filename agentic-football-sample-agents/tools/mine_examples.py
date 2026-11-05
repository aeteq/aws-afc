#!/usr/bin/env python3
"""Labelled on-ball decisions from kak1's decision logs: what we did with the ball, and
what happened next. Input for few-shot examples and the between-match coach.

    python3 mine_examples.py /tmp/kak1logs --source live > live.jsonl
    python3 mine_examples.py simlogs/ --source sim >> examples.jsonl
    python3 mine_examples.py /tmp/kak1logs --summary          # counts per command/outcome

Reads the same files as live_report.py. For every decision taken WITH the ball (log
owner == "me", including the keeper's distributions) it looks at the next ticks:

  lost         the opponent has the ball within LOST_TICKS decision ticks
  kept         we still have it LOST_TICKS ticks later (any of our players)
  shot         one of our players shoots within SHOT_TICKS ticks
  goal_for     we score within GOAL_TICKS ticks
  goal_against they score within GOAL_TICKS ticks

and gives a score: goal_for +5, shot +2, kept +1, lost -2, goal_against -5 (summed).

Each line holds the decision (cmd, params, source), where everyone was (live units, in
OUR attack direction: our goal at -half, theirs at +half, so home and away examples read
the same), the labels and the score. Teammate positions come from our other players'
log lines at the same tick; opponents only if the log line carries "opps".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_report import ROLE, _records, _split  # noqa: E402

LOST_TICKS, SHOT_TICKS, GOAL_TICKS = 2, 3, 5
WEIGHTS = {"goal_for": 5, "shot": 2, "kept": 1, "lost": -2, "goal_against": -5}


def _flip(p, s):
    return None if not p else [round(p[0] * s, 2), round(p[1] * s, 2)]


def _ours(owner):
    return owner in ("me", "team")


def mine(match, source, match_idx):
    dec = [r for r in match if "pid" in r and r.get("t") is not None]
    if not dec:
        return []
    team = next((r["team"] for r in dec if r.get("team") is not None), 0)
    s = 1 if team == 0 else -1
    by_t = defaultdict(dict)
    for r in dec:
        by_t[r["t"]][r["pid"]] = r
    ticks = sorted(by_t)

    def owner_at(t):
        rows = by_t[t].values()
        if any(r.get("owner") == "me" for r in rows):
            return "us"
        o = next(iter(rows)).get("owner")
        return "us" if _ours(o) else o

    def score_at(t):
        sc = next(iter(by_t[t].values())).get("score") or {}
        h, a = sc.get("home", 0), sc.get("away", 0)
        return (h, a) if team == 0 else (a, h)

    out = []
    for i, t in enumerate(ticks):
        for pid, r in by_t[t].items():
            if r.get("owner") != "me":
                continue
            labels = set()
            nxt = ticks[i + 1:]
            for k, tt in enumerate(nxt[:LOST_TICKS]):
                if owner_at(tt) == "opponent":
                    labels.add("lost")
                    break
            else:
                if len(nxt) >= LOST_TICKS and owner_at(nxt[LOST_TICKS - 1]) == "us":
                    labels.add("kept")
            if r.get("cmd") == "SHOOT" or any(x.get("cmd") == "SHOOT" for tt in nxt[:SHOT_TICKS]
                                               for x in by_t[tt].values()):
                labels.add("shot")
            us0, them0 = score_at(t)
            for tt in nxt[:GOAL_TICKS]:
                us1, them1 = score_at(tt)
                if us1 > us0:
                    labels.add("goal_for")
                if them1 > them0:
                    labels.add("goal_against")
            mates = {str(q): _flip(x.get("me"), s) for q, x in sorted(by_t[t].items()) if q != pid}
            ex = {
                "source": source, "match": match_idx, "t": t, "pid": pid, "role": ROLE.get(pid, "?"),
                "src": r.get("src"), "cmd": r.get("cmd"), "params": _flip_params(r.get("params"), s),
                "me": _flip(r.get("me"), s), "ball": _flip(r.get("ball"), s), "mates": mates,
                "opps": {str(k): _flip(v, s) for k, v in (r.get("opps") or {}).items()} or None,
                "score": list(score_at(t)), "labels": sorted(labels),
                "value": sum(WEIGHTS[x] for x in labels),
            }
            out.append(ex)
    return out


def _flip_params(p, s):
    p = dict(p or {})
    if "target_x" in p:
        p["target_x"] = round(p["target_x"] * s, 2)
        p["target_y"] = round(p["target_y"] * s, 2)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--source", default="live", help="tag for every example (live, sim, ...)")
    ap.add_argument("--min-ticks", type=int, default=40,
                    help="skip matches with fewer decision ticks (test calls, aborted matches)")
    ap.add_argument("--summary", action="store_true", help="print counts instead of JSON lines")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    recs = [r for f in files for r in _records(f)]
    examples = []
    for i, m in enumerate(_split(recs)):
        if len({r.get("t") for r in m if "pid" in r}) < args.min_ticks:
            continue
        examples += mine(m, args.source, i)
    if not args.summary:
        for ex in examples:
            print(json.dumps(ex, separators=(",", ":")))
        return
    print(f"{len(examples)} on-ball decisions")
    by = defaultdict(list)
    for ex in examples:
        by[(ex["role"], ex["cmd"], ex["params"].get("type") or ex["params"].get("method") or "")].append(ex)
    print(f"  {'role':5s}{'command':24s}{'n':>4s}{'value':>7s}  outcomes")
    for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        c = Counter(l for ex in xs for l in ex["labels"])
        v = sum(ex["value"] for ex in xs) / len(xs)
        print(f"  {k[0]:5s}{(k[1] + ' ' + k[2]).strip():24s}{len(xs):4d}{v:+7.2f}  " +
              ", ".join(f"{l} {n}" for l, n in c.most_common()))


if __name__ == "__main__":
    main()
