#!/usr/bin/env python3
"""Turn labelled on-ball decisions (tools/mine_examples.py output, JSON lines) into a few short
'what has worked' lines for the choice model's prompt. Writes kaklib/lessons.json.

    python3 ../tools/mine_examples.py /tmp/kak1logs --source live > /tmp/live.jsonl
    python3 make_lessons.py /tmp/live.jsonl [--version live-3m]

Only actions seen at least MIN_N times get a line; each line states its count, so the model
(and we) can see how thin the evidence is. Live examples only (source == "live")."""
import argparse
import json
import os
from collections import defaultdict

MIN_N = 5


def action(ex):
    c, p = ex.get("cmd"), ex.get("params") or {}
    if c == "PASS":
        return f"{p.get('type', '?')} passes"
    if c == "GK_DISTRIBUTE":
        return f"keeper {p.get('method', '?').lower()}s"
    if c == "MOVE_TO":
        return "dribbles"
    if c == "SHOOT":
        return "shots"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("examples")
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaklib", "lessons.json"))
    a = ap.parse_args()
    stats = defaultdict(lambda: {"n": 0, "lost": 0, "shot": 0, "goal_for": 0, "goal_against": 0})
    for line in open(a.examples):
        ex = json.loads(line)
        if ex.get("source", "live") != "live":
            continue
        k = action(ex)
        if not k:
            continue
        s = stats[k]
        s["n"] += 1
        for lab in ex.get("labels", []):
            if lab in s:
                s[lab] += 1
    lines = []
    for k, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        if s["n"] < MIN_N:
            continue
        if k == "shots":
            lines.append(f"Shots: scored {s['goal_for']} of {s['n']}.")
            continue
        parts = [f"lost the ball within 2 turns {s['lost']} of {s['n']} times"]
        if s["shot"]:
            parts.append(f"led to a shot {s['shot']} times")
        if s["goal_for"]:
            parts.append(f"led to our goal {s['goal_for']} times")
        if s["goal_against"]:
            parts.append(f"was followed by a goal against {s['goal_against']} times")
        lines.append(f"{k[0].upper() + k[1:]}: " + ", ".join(parts) + ".")
    out = {"version": a.version, "source": os.path.basename(a.examples), "lines": lines[:8]}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
