#!/usr/bin/env python3
"""Play practice matches and collect their logs, with nobody at the portal.

    export AFC_TEAM_CODE=...                         # never commit it
    python3 autoplay.py --check                      # log in, show the team and its practice matches
    python3 autoplay.py --matches 3 --opponent balanced --label a74f040-choice --profile afc \
        --notify kakoushias/t11 --notify pangratios/t14

For each match: start a practice match through the Player Portal's API (what the portal's
"practice" button does), wait until it is completed, wait for CloudWatch to catch up, pull
the five kak1 agents' logs for that match's time window into matches/<time>_<id>/, run
live_report.py and mine_examples.py on them, append the labelled decisions to
matches/all_examples.jsonl, and tell the agents listed in --notify (hc send).

Matches use shared event capacity, so it is polite on purpose: one match at a time (it waits
for any match already in progress), at least --gap seconds between matches, at most
--matches per run, and it stops at the first 429 / 409 / "slot" / "capacity" answer.

It only plays and measures. Changing code and deploying stay with the agents (review, bench).
API (from the portal's public bundle, 2026-11-05):
  POST {API}/teams/login {"team_code"} -> {team_id, team_code, ...}
  auth header: Authorization: Bearer team:<team_code>
  POST {API}/practice-matches {"team_id", "opponent": balanced|aggressive|defensive} -> {match_id}
  GET  {API}/practice-matches -> {items: [{status: in_progress|completed|..., ...}]}
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_report  # noqa: E402
import mine_examples  # noqa: E402

API = "https://l3fmtx4zp0.execute-api.us-east-1.amazonaws.com/prod"
AGENTS = ("gk", "def", "mid", "fwd1", "fwd2")
OUT = os.path.join(HERE, "..", "matches")
POLL_S, MATCH_TIMEOUT_S, LOG_LAG_S = 15, 15 * 60, 90


class Stop(Exception):
    """The API said no (capacity, rate limit): stop the run."""


def _http(method, path, token=None, body=None):
    req = urllib.request.Request(API + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "kak1-autoplay",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:300]
        if e.code in (409, 429) or any(w in text.lower() for w in ("slot", "capacity", "limit")):
            raise Stop(f"{method} {path}: {e.code} {text}")
        raise RuntimeError(f"{method} {path}: {e.code} {text}")


def login(code):
    team = _http("POST", "/teams/login", body={"team_code": code})
    if not team.get("team_id"):
        raise SystemExit(f"login gave no team_id: {list(team)}")
    return team, f"team:{team.get('team_code', code)}"


def practice_matches(token):
    return (_http("GET", "/practice-matches", token) or {}).get("items") or []


def wait_idle(token):
    """Wait while one of our practice matches is in progress."""
    t0 = time.time()
    while any(m.get("status") == "in_progress" for m in practice_matches(token)):
        if time.time() - t0 > MATCH_TIMEOUT_S:
            raise Stop("a practice match has been in progress for 15 min; not starting another")
        time.sleep(POLL_S)


def play(token, team_id, opponent):
    started = time.time()
    res = _http("POST", "/practice-matches", token, {"team_id": team_id, "opponent": opponent})
    mid = res.get("match_id") or res.get("id")
    print(f"started practice match {mid} vs {opponent}", flush=True)
    while True:
        time.sleep(POLL_S)
        items = practice_matches(token)
        me = next((m for m in items if (m.get("match_id") or m.get("id")) == mid), None)
        status = (me or {}).get("status")
        if status and status != "in_progress":
            return mid, me, started, time.time()
        if time.time() - started > MATCH_TIMEOUT_S:
            raise Stop(f"match {mid} not finished after 15 min")


def pull_logs(profile, region, t_start, t_end, dest):
    import boto3
    logs = boto3.Session(profile_name=profile, region_name=region).client("logs")
    os.makedirs(dest, exist_ok=True)
    for a in AGENTS:
        groups = logs.describe_log_groups(
            logGroupNamePrefix=f"/aws/bedrock-agentcore/runtimes/kak1_{a}_agent")["logGroups"]
        events = []
        for g in groups:
            kw = dict(logGroupName=g["logGroupName"], startTime=int(t_start * 1000),
                      endTime=int(t_end * 1000))
            for page in logs.get_paginator("filter_log_events").paginate(**kw):
                events += page.get("events", [])
        with open(os.path.join(dest, f"kak1_{a}.json"), "w") as f:
            json.dump({"events": events}, f)


def analyse(dest, label, match_id, opponent):
    files = [os.path.join(dest, f"kak1_{a}.json") for a in AGENTS]
    recs = [r for f in files if os.path.exists(f) for r in live_report._records(f)]
    full = [m for m in live_report._split(recs) if len({r.get("t") for r in m if "pid" in r}) >= 40]
    if not full:
        return None
    m = full[-1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        live_report.report(m, 0)
    open(os.path.join(dest, "report.txt"), "w").write(buf.getvalue())
    ex = mine_examples.mine(m, "live", 0)
    with open(os.path.join(OUT, "all_examples.jsonl"), "a") as f:
        for e in ex:
            f.write(json.dumps(dict(e, match_id=match_id, label=label, opponent=opponent),
                               separators=(",", ":")) + "\n")
    dec = [r for r in m if "pid" in r]
    team = next((r["team"] for r in dec if r.get("team") is not None), 0)
    last = max(dec, key=lambda r: (r.get("t") or 0))
    sc = last.get("score") or {}
    us, them = (sc.get("home", 0), sc.get("away", 0)) if team == 0 else (sc.get("away", 0), sc.get("home", 0))
    return {"us": us, "them": them, "side": "home" if team == 0 else "away",
            "on_ball": len(ex), "shots": sum(1 for r in dec if r.get("cmd") == "SHOOT"),
            "src": sorted({r["src"].split(":")[0] for r in dec})}


def notify(addrs, text):
    for a in addrs:
        subprocess.run(["hc", "send", a, text], capture_output=True, timeout=60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="log in and list practice matches, play nothing")
    ap.add_argument("--matches", type=int, default=1, help="matches to play this run (max 6)")
    ap.add_argument("--opponent", action="append", choices=("balanced", "aggressive", "defensive"),
                    help="repeat to rotate (default balanced)")
    ap.add_argument("--gap", type=int, default=180, help="seconds between matches (min 180)")
    ap.add_argument("--label", default="", help="what is deployed, e.g. a74f040-choice")
    ap.add_argument("--profile", default=os.getenv("AWS_PROFILE"))
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--notify", action="append", default=[], help="hc address to message after each match")
    args = ap.parse_args()

    code = os.getenv("AFC_TEAM_CODE")
    if not code:
        raise SystemExit("set AFC_TEAM_CODE (the team code from the organiser; never commit it)")
    team, token = login(code)
    team_id = team["team_id"]
    if args.check:
        print(json.dumps({k: team.get(k) for k in ("team_id", "team_name", "name") if k in team}))
        for m in practice_matches(token)[:10]:
            print(json.dumps(m)[:300])
        return
    os.makedirs(OUT, exist_ok=True)
    opponents = args.opponent or ["balanced"]
    results = []
    try:
        for i in range(min(args.matches, 6)):
            if i:
                time.sleep(max(args.gap, 180))
            wait_idle(token)
            opp = opponents[i % len(opponents)]
            mid, info, t0, t1 = play(token, team_id, opp)
            time.sleep(LOG_LAG_S)
            stamp = dt.datetime.fromtimestamp(t0).strftime("%Y%m%d-%H%M")
            dest = os.path.join(OUT, f"{stamp}_{mid}")
            pull_logs(args.profile, args.region, t0 - 30, t1 + LOG_LAG_S, dest)
            json.dump({"match_id": mid, "opponent": opp, "label": args.label, "api": info,
                       "started": t0, "finished": t1}, open(os.path.join(dest, "match.json"), "w"), indent=2)
            r = analyse(dest, args.label, mid, opp)
            results.append((opp, r))
            line = (f"AUTOPLAY {i + 1}/{args.matches} [{args.label}] vs {opp}: "
                    + (f"kak1 {r['us']}-{r['them']} ({r['side']}), shots {r['shots']}, on-ball {r['on_ball']}, "
                       f"src {','.join(r['src'])}" if r else "no full match found in the logs")
                    + f". Logs + report: {os.path.abspath(dest)}")
            print(line, flush=True)
            notify(args.notify, line)
    except Stop as e:
        print(f"stopped: {e}", flush=True)
        notify(args.notify, f"AUTOPLAY stopped: {e}")
    played = [r for _, r in results if r]
    if played:
        w = sum(r["us"] > r["them"] for r in played)
        d = sum(r["us"] == r["them"] for r in played)
        gf, ga = sum(r["us"] for r in played), sum(r["them"] for r in played)
        print(f"\n[{args.label}] {len(played)} matches: {w}W {d}D {len(played) - w - d}L, goals {gf}-{ga}")


if __name__ == "__main__":
    main()
