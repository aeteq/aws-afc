#!/usr/bin/env python3
"""Play simulated Agentic Football matches locally.

    python3 simulator/run_match.py                       # series: sample vs every built-in team, home and away
    python3 simulator/run_match.py --matches 50          # more matches per side (default 20)
    python3 simulator/run_match.py --team aggressive     # run the series for another team
    python3 simulator/run_match.py --home sample --away aggressive --seed 3
    python3 simulator/run_match.py --home py:examples/my_team.py --away sample --replay match.html

Another team's agents, home and away (team specs: simulator/external.py):

    python3 simulator/run_match.py --opponent rivals=http://127.0.0.1:9080-9084 --matches 5
    python3 simulator/run_match.py --home mine=@my-arns.txt --away rivals=@rivals-arns.txt --replay m.html
    python3 simulator/run_match.py --opponent py:../their-repo/sim_team.py:commands

Teams: sample, aggressive, defensive, idle, box, diamond, pyramid, 3-1 (see simulator/teams.py), or a spec.
Physics are a stand-in for the real server's; see simulator/engine.py before reading
anything into a scoreline.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import MATCH_SECONDS, Match  # noqa: E402
from external import parse_spec  # noqa: E402
from teams import TEAM_NAMES, all_team_names, make  # noqa: E402


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _label(team: str) -> str:
    return team if team in all_team_names() else parse_spec(team)[0]


def play_one(home: str, away: str, seed: int, replay: str | None, deadline: float | None,
             realtime: bool = False) -> None:
    controllers = make(home, deadline), make(away, deadline)
    r = Match(*controllers, seed=seed, record=replay is not None, realtime=realtime).run()
    home, away = r.home, r.away
    print(f"\n  {home.upper()} {r.score[0]} - {r.score[1]} {away.upper()}   (seed {seed})\n")
    for e in r.events:
        if e[1] == "goal":
            scorer = home if e[2] == 0 else away
            who = f"player {e[3]}" if e[3] is not None else "own goal"
            print(f"  {_clock(e[0]):>5}  GOAL {scorer:10s} {who:9s}  {e[4]}")
    hs, as_ = r.stats
    total = (hs.possession + as_.possession) or 1.0
    rows = [("shots (on target)", f"{hs.shots} ({hs.on_target})", f"{as_.shots} ({as_.on_target})"),
            ("possession", f"{100 * hs.possession / total:.0f}%", f"{100 * as_.possession / total:.0f}%"),
            ("passes completed", f"{hs.passes_completed}/{hs.passes}", f"{as_.passes_completed}/{as_.passes}"),
            ("tackles won", hs.tackles_won, as_.tackles_won),
            ("saves", hs.saves_made, as_.saves_made),
            ("wasted commands", hs.wasted_commands, as_.wasted_commands),
            ("rejected commands", hs.rejected_commands, as_.rejected_commands)]
    print(f"\n  {'':20s}{home:>12s}{away:>12s}")
    for label, h, a in rows:
        print(f"  {label:20s}{str(h):>12s}{str(a):>12s}")
    for c in controllers:
        if hasattr(c, "summary"):
            print(f"\n  {c.summary()}")
    if replay:
        from replay import write_replay
        write_replay(replay, r)
        print(f"\n  replay written to {os.path.abspath(replay)}")


def series(team: str, matches: int, seed: int, opponents: list[str] | None = None,
           deadline: float | None = None) -> None:
    opponents = opponents or [t for t in TEAM_NAMES if t != team] + [team]  # mirror match last
    print(f"\n{_label(team).upper()}: {matches} matches per side vs each opponent "
          f"({MATCH_SECONDS / 60:.0f}-minute simulated matches)\n")
    print(f"  {'opponent':11s}{'side':6s}{'W':>4s}{'D':>4s}{'L':>4s}{'GF':>6s}{'GA':>5s}"
          f"{'shots':>8s}{'poss':>7s}{'pass%':>7s}{'wasted':>8s}")
    totals = [0, 0, 0, 0, 0]
    for opp in opponents:
        for side in ("home", "away"):
            w = d = l = gf = ga = shots = wasted = 0
            poss = poss_all = passes = completed = 0.0
            remote = {}  # name -> [calls, late, unreadable] for teams that answer over a network
            for m in range(matches):
                s = seed + m
                ours, theirs = make(team, deadline), make(opp, deadline)
                if side == "home":
                    r = Match(ours, theirs, seed=s).run()
                    us, them, goals = r.stats[0], r.stats[1], r.score
                else:
                    r = Match(theirs, ours, seed=s).run()
                    us, them, goals = r.stats[1], r.stats[0], r.score[::-1]
                for c in (ours, theirs):
                    if hasattr(c, "calls"):
                        counts = remote.setdefault(c.name, [0, 0, 0])
                        for i, v in enumerate((c.calls, c.late, c.errors)):
                            counts[i] += v
                gf += goals[0]
                ga += goals[1]
                w, d, l = w + (goals[0] > goals[1]), d + (goals[0] == goals[1]), l + (goals[0] < goals[1])
                shots += us.shots
                poss += us.possession
                poss_all += us.possession + them.possession
                passes += us.passes
                completed += us.passes_completed
                wasted += us.wasted_commands
            pass_pct = f"{100 * completed / passes:.0f}%" if passes else "-"
            print(f"  {_label(opp):11.11s}{side:6s}{w:4d}{d:4d}{l:4d}{gf:6d}{ga:5d}{shots / matches:8.1f}"
                  f"{100 * poss / (poss_all or 1):6.0f}%{pass_pct:>7s}{wasted / matches:8.1f}")
            for name, (calls, late, errors) in remote.items():
                print(f"  {'':17s}{name}: {calls} agent calls, {late} late, {errors} unreadable")
            if opp != team:
                for i, v in enumerate((w, d, l, gf, ga)):
                    totals[i] += v
    w, d, l, gf, ga = totals
    print(f"\n  vs others: {w}W {d}D {l}L, goals {gf}-{ga}   "
          "(shots, wasted: per match; wasted = PASS/SHOOT sent without the ball)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    team_help = f"{', '.join(all_team_names())}, or a team spec (simulator/external.py)"
    parser.add_argument("--team", default="sample", help=f"team for the series: {team_help}")
    parser.add_argument("--opponent", action="append", metavar="TEAM",
                        help="series against only this opponent (repeatable)")
    parser.add_argument("--home", help=f"single match: {team_help}")
    parser.add_argument("--away")
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deadline", type=float, metavar="SECONDS",
                        help="remote agents: drop answers later than this (default 1, 0 = wait)")
    parser.add_argument("--replay", metavar="FILE.html", help="single match only: write a replay viewer")
    parser.add_argument("--realtime", action="store_true",
                        help="single match only: one turn every 2 s of wall clock, both teams "
                             "asked at once, as live (a 5-minute match takes 5 minutes)")
    args = parser.parse_args()

    if args.home or args.away:
        play_one(args.home or "sample", args.away or "aggressive", args.seed, args.replay,
                 args.deadline, args.realtime)
    else:
        series(args.team, args.matches, args.seed, args.opponent, args.deadline)


if __name__ == "__main__":
    main()
