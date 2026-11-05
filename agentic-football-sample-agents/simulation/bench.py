#!/usr/bin/env python3
"""Regression bench: is this version of a team better or worse than another one?

    python3 bench.py                                   # ai-team-kak1 now vs origin/main
    python3 bench.py --base HEAD~3 --seeds 40          # vs an older commit, more matches
    python3 bench.py --team ../ai-team-kak1 --base none   # no baseline, just the table
    python3 bench.py --live                            # live-like mode (engine.LiveMatch)

Plays NEW and BASE on the same seeds, both sides (home and away), against the same
opponents, plus NEW vs BASE head to head. Prints one line per opponent and a verdict.
Exit code 1 = regression: NEW is clearly worse than BASE (head to head, or paired goal
difference across the opponent pool, with the 95% interval below zero).

Why both sides: the engine moves HOME players first, which gives AWAY a one-step edge in
every chase and press, so a team's results differ a lot by side. Why the same seeds: the
engine is seeded, so NEW and BASE face the same coin tosses and the difference is the code.

BASE comes from git (`git archive`, no checkout), so this works in a sparse checkout and
never touches your working tree. Each team is loaded in isolation: two versions of kak1
both `import tactics`, and without isolation the second would silently reuse the first.

The simulator is not the live server (pitch size, match length, physics: SIMULATION.md).
Use this to catch regressions and compare versions, not to predict live scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "engine", "simulator")
SAMPLES = os.path.dirname(HERE)                       # agentic-football-sample-agents/
REPO_PREFIX = "agentic-football-sample-agents"
sys.path.insert(0, ENGINE)

from engine import LiveMatch, Match  # noqa: E402

BUILTINS = ["sample", "aggressive", "defensive", "box", "diamond", "pyramid", "3-1"]
AWS_FALLBACK = os.path.join(HERE, "teams", "sample_fallback.py")
# Stand-ins for the live, model-driven AWS teams (built from their prompts and live logs).
FILE_TEAMS = {"aws": AWS_FALLBACK,
              "nova": os.path.join(HERE, "teams", "nova_balanced.py"),
              "novafix": os.path.join(HERE, "teams", "nova_balanced_fixed.py")}


# --- loading teams in isolation ------------------------------------------------------------

class _PyTeam:
    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def commands(self, game_state, team_id, player_id):
        return self._fn(game_state, team_id, player_id)


def _load_isolated(name: str, path: str):
    """Import FILE.py so that the modules it imports (tactics, state, fallback, ...) are its
    own copies: snapshot sys.modules/sys.path, load, then forget everything it added."""
    before_mods, before_path = set(sys.modules), list(sys.path)
    # Names the teams and the engine both use; a cached copy would leak between teams.
    for clash in ("tactics", "state", "fallback", "parsing", "fast_agent", "agent_base",
                  "json_tolerant", "_bootstrap"):
        sys.modules.pop(clash, None)
    try:
        spec = importlib.util.spec_from_file_location(f"bench_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for m in set(sys.modules) - before_mods:
            sys.modules.pop(m, None)
        sys.path[:] = before_path
    fn = getattr(mod, "commands", None)
    if not callable(fn):
        raise SystemExit(f"{name}: {path} has no commands()")
    return lambda: _PyTeam(name, fn)


_FACTORIES: dict = {}
_MATCH = Match


def _init_worker(specs: dict, live: bool = False):
    global _MATCH
    if live:
        _use_live_settings()
        _MATCH = LiveMatch
    from teams import make
    for name, spec in specs.items():
        if spec.startswith("py:"):
            _FACTORIES[name] = _load_isolated(name, spec[3:])
        else:
            _FACTORIES[name] = (lambda s=spec: make(s, None))


def _play(job):
    """(us, them, side, seed) -> (goals for, goals against, our shots, our possession share)."""
    us, them, side, seed = job
    a, b = _FACTORIES[us](), _FACTORIES[them]()
    if side == "home":
        r = _MATCH(a, b, seed=seed).run()
        ours, theirs, (gf, ga) = r.stats[0], r.stats[1], r.score
    else:
        r = _MATCH(b, a, seed=seed).run()
        ours, theirs, (ga, gf) = r.stats[1], r.stats[0], r.score
    tot = (ours.possession + theirs.possession) or 1.0
    return gf, ga, ours.shots, ours.possession / tot


def _use_live_settings():
    """kak1's deployed settings; read when tactics.py is imported, so set before any load."""
    os.environ.update(PITCH_HALF_X="7", BALL_LATERAL="z", MATCH_SECONDS="120")
    import teams
    teams.DOC_SCALE_BUILTINS = True   # built-in teams see documented coordinates (teams.DocScale)


# --- stats -------------------------------------------------------------------------------

def _mean_ci(xs):
    """Mean and 95% half-width (normal approximation; fine for n >= ~20)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, float("inf")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def _row(results):
    w = sum(gf > ga for gf, ga, *_ in results)
    d = sum(gf == ga for gf, ga, *_ in results)
    l_ = len(results) - w - d
    gf = sum(r[0] for r in results)
    ga = sum(r[1] for r in results)
    pts = (3 * w + d) / max(len(results), 1)
    gd, ci = _mean_ci([r[0] - r[1] for r in results])
    shots = sum(r[2] for r in results) / max(len(results), 1)
    poss = 100 * sum(r[3] for r in results) / max(len(results), 1)
    return w, d, l_, gf, ga, pts, gd, ci, shots, poss


# --- base version from git -----------------------------------------------------------------

def _export_base(ref: str, rel_team: str, dest: str) -> str:
    """Extract REF's team folder (relative to agentic-football-sample-agents/) and lib/ into
    dest; return the base sim_team.py path."""
    paths = [f"{REPO_PREFIX}/{rel_team}", f"{REPO_PREFIX}/lib"]
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout.strip()
    blob = subprocess.run(["git", "archive", "--format=tar", ref, *paths], cwd=top,
                          capture_output=True, check=True).stdout
    with tarfile.open(fileobj=BytesIO(blob)) as tf:
        tf.extractall(dest, filter="data")
    path = os.path.join(dest, REPO_PREFIX, rel_team, "sim_team.py")
    if not os.path.exists(path):
        raise SystemExit(f"{ref} has no {rel_team}/sim_team.py")
    return path


def _describe(ref):
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %s", ref], cwd=HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return ref


# --- main ---------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--team", default=os.path.join(SAMPLES, "ai-team-kak1"),
                    help="team folder with a sim_team.py (default: ai-team-kak1)")
    ap.add_argument("--base", default="origin/main",
                    help="git ref to compare against, or 'none' (default: origin/main)")
    ap.add_argument("--base-team", help="team folder in the repo to take BASE from, relative to "
                    "agentic-football-sample-agents/ (default: the same as --team). Needed when "
                    "--team is a copy outside the repo, e.g. --base-team ai-team-kak1")
    ap.add_argument("--seeds", type=int, default=20, help="matches per side per opponent (default 20)")
    ap.add_argument("--seed", type=int, default=1, help="first seed (default 1)")
    ap.add_argument("--opponents", default=",".join(BUILTINS),
                    help="comma list of built-ins, 'aws' (teams/sample_fallback.py, same code as "
                         "'sample'), 'nova' / 'novafix' (stand-ins for the live Nova-driven AWS teams, "
                         "with and without their possession bug)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--live", action="store_true",
                    help="live-like mode (engine.LiveMatch: live pitch units, 120 s matches, "
                         "measured constants); both versions run with kak1's deployed settings")
    args = ap.parse_args()

    team_file = os.path.join(os.path.abspath(args.team), "sim_team.py")
    if not os.path.exists(team_file):
        raise SystemExit(f"no sim_team.py in {args.team}")
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]

    with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
        specs = {"new": "py:" + team_file}
        for o in opponents:
            specs[o] = "py:" + FILE_TEAMS[o] if o in FILE_TEAMS else o
        has_base = args.base.lower() != "none"
        if has_base:
            rel = args.base_team or os.path.relpath(os.path.abspath(args.team), SAMPLES)
            if rel.startswith(".."):
                raise SystemExit("--team is outside the repo: say which repo team is the base, "
                                 "e.g. --base-team ai-team-kak1")
            specs["base"] = "py:" + _export_base(args.base, rel, tmp)

        seeds = range(args.seed, args.seed + args.seeds)
        jobs = []
        players = ["new"] + (["base"] if has_base else [])
        for us in players:
            for o in opponents:
                jobs += [(us, o, side, s) for side in ("home", "away") for s in seeds]
        if has_base:
            jobs += [("new", "base", side, s) for side in ("home", "away") for s in seeds]

        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker,
                                 initargs=(specs, args.live)) as pool:
            out = dict(zip(jobs, pool.map(_play, jobs, chunksize=8)))

    n = args.seeds
    print(f"\nNEW = {os.path.relpath(team_file, os.getcwd())} (working tree)")
    if has_base:
        print(f"BASE = {args.base}: {_describe(args.base)}")
    if args.live:
        print(f"{n} seeds x home+away per opponent (simulator LIVE mode: live pitch units, "
              "120-second matches)\n")
    else:
        print(f"{n} seeds x home+away per opponent (simulator, documented pitch, 5-minute matches)\n")
    hdr = f"  {'opponent':10s}{'':5s}{'W':>4s}{'D':>4s}{'L':>4s}{'GF':>5s}{'GA':>5s}{'pts/m':>7s}{'GD/m':>7s}{'±95%':>6s}{'shots':>7s}{'poss':>6s}"
    print(hdr)
    paired = []                        # per (opponent, side, seed): new GD - base GD
    for o in opponents:
        for us in players:
            res = [out[(us, o, side, s)] for side in ("home", "away") for s in seeds]
            w, d, l_, gf, ga, pts, gd, ci, shots, poss = _row(res)
            print(f"  {o:10.10s}{us:5s}{w:4d}{d:4d}{l_:4d}{gf:5d}{ga:5d}{pts:7.2f}{gd:+7.2f}{ci:6.2f}{shots:7.1f}{poss:5.0f}%")
        if has_base:
            for side in ("home", "away"):
                for s in seeds:
                    a, b = out[("new", o, side, s)], out[("base", o, side, s)]
                    paired.append((a[0] - a[1]) - (b[0] - b[1]))

    worse = False
    if has_base:
        print()
        for side in ("home", "away"):
            res = [out[("new", "base", side, s)] for s in seeds]
            w, d, l_, gf, ga, pts, gd, ci, *_ = _row(res)
            print(f"  NEW vs BASE, NEW {side}: {w}W {d}D {l_}L, goals {gf}-{ga}, GD/m {gd:+.2f} ±{ci:.2f}")
        h2h = [out[("new", "base", side, s)] for side in ("home", "away") for s in seeds]
        w, d, l_, gf, ga, pts, gd, ci, *_ = _row(h2h)
        print(f"  NEW vs BASE, both:     {w}W {d}D {l_}L, goals {gf}-{ga}, GD/m {gd:+.2f} ±{ci:.2f}")
        pm, pci = _mean_ci(paired)
        print(f"  Paired GD/m vs the pool, NEW minus BASE: {pm:+.2f} ±{pci:.2f}  ({len(paired)} pairs)")
        h2h_worse = gd + ci < 0
        pool_worse = pm + pci < 0
        better = (gd - ci > 0) or (pm - pci > 0)
        worse = h2h_worse or pool_worse
        verdict = ("REGRESSION" if worse else "BETTER" if better else "NO CLEAR DIFFERENCE")
        print(f"\n  Verdict: {verdict}")
    sys.exit(1 if worse else 0)


if __name__ == "__main__":
    main()
