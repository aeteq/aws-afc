"""Teams the simulator can field. Each one answers commands(game_state, team_id, player_id).

Every controller is called once per player, so each player is its own "agent", as on the
server.

  sample      the rule-based fallback every AWS sample team shares (lib/fallback.py), i.e.
              what any of the five Strands sample teams plays whenever its model fails
  aggressive  rule-based approximation of the extremely-aggressive sample team's prompts
  defensive   rule-based approximation of the extremely-defensive sample team's prompts
  idle        sends nothing; the engine's default AI plays (a baseline)
  box, diamond, pyramid, 3-1
              classic 5-a-side shapes (simulator/opponents.py)

The sample teams' real decisions come from an LLM, which cannot run here without Bedrock.
"sample" is their exact fallback code, while "aggressive" and "defensive" are only
approximations of what their prompts ask for, built with the same fallback machinery.

Any other name is a team spec for your own team: a Python function, agents on localhost,
or deployed AgentCore runtimes. See simulator/external.py.
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from fallback import (DEF_CONFIG, FWD1_CONFIG, FWD2_CONFIG, GK_CONFIG, MID_CONFIG,  # noqa: E402
                      build_fallback)


class RuleBased:
    def __init__(self, name, configs):
        self.name = name
        self._fns = {pid: build_fallback(cfg) for pid, cfg in configs.items()}

    def commands(self, game_state, team_id, player_id):
        return self._fns[player_id](game_state, team_id, player_id)


class Idle:
    name = "idle"

    def commands(self, game_state, team_id, player_id):
        return []


SAMPLE = {0: GK_CONFIG, 1: DEF_CONFIG, 2: MID_CONFIG, 3: FWD1_CONFIG, 4: FWD2_CONFIG}

# "Push high, press at maximum intensity, shoot from 30-40 units, never track back."
AGGRESSIVE = {
    0: GK_CONFIG,
    1: replace(DEF_CONFIG, possession_action="SHOOT_OR_ADVANCE", shoot_threshold=30.0,
               default_x_factor=0.1, default_x_ref="opp_goal", mark_threshold=0.0,
               press_distance=40.0, press_intensity=1.0),
    2: replace(MID_CONFIG, shoot_threshold=35.0, default_x_factor=0.4, default_x_ref="opp_goal",
               press_distance=40.0, press_intensity=1.0, shoot_power=1.0),
    3: replace(FWD1_CONFIG, shoot_threshold=40.0, default_x_factor=0.7, advance_x_factor=0.85,
               press_distance=40.0, press_intensity=1.0, shoot_power=1.0),
    4: replace(FWD2_CONFIG, shoot_threshold=40.0, default_x_factor=0.7, advance_x_factor=0.85,
               press_distance=40.0, press_intensity=1.0, shoot_power=1.0),
}

# "Never cross halfway, mark tight, pass back, shoot only inside ~15 units."
DEFENSIVE = {
    0: GK_CONFIG,
    1: replace(DEF_CONFIG, default_x_factor=0.75, mark_threshold=45.0, press_distance=12.0),
    2: replace(MID_CONFIG, possession_action="PASS", default_x_factor=0.5, default_x_ref="my_goal",
               default_y=0, press_distance=15.0),
    3: replace(FWD1_CONFIG, shoot_threshold=15.0, default_x_factor=0.15, default_x_ref="my_goal",
               support_x_factor=0.1, advance_x_factor=0.2, press_distance=15.0, support_sprint=False),
    4: replace(FWD2_CONFIG, shoot_threshold=15.0, default_x_factor=0.15, default_x_ref="my_goal",
               support_x_factor=0.1, advance_x_factor=0.2, press_distance=15.0, support_sprint=False),
}


# Live mode (run_match.py --live, bench.py --live). The built-in teams are written for the
# documented pitch (x -55..55). Fed live units they would send their players to the corners,
# so in live mode they get the conversion a pitch-aware team does (DocScale), as kak1 does
# itself. A team spec (py:, http, arn) gets the live payload unchanged, as on the server:
# py:../teams/sample_fallback.py plays the AWS fallback exactly as it would run live.
DOC_SCALE_BUILTINS = False


class DocScale:
    """Wraps a documented-pitch team for engine.LiveMatch: live payload in, documented
    coordinates to the team (ball made flat: y = sideways), MOVE_TO targets back to live."""

    def __init__(self, inner):
        from engine import LIVE_SX, LIVE_SY
        self.inner, self.name, self.sx, self.sy = inner, inner.name, LIVE_SX, LIVE_SY

    def commands(self, game_state, team_id, player_id):
        gs = game_state
        if gs.get("playMode") != "KICK_OFF":            # the kick-off payload is already documented
            gs = dict(gs, ball=self._ball(gs.get("ball") or {}),
                      players=[dict(p, position=self._xy(p.get("position")), velocity=self._xy(p.get("velocity")))
                               for p in gs.get("players") or []])
        out = []
        for c in self.inner.commands(gs, team_id, player_id) or []:
            prm = c.get("parameters") if isinstance(c, dict) else None
            if isinstance(prm, dict) and c.get("commandType") == "MOVE_TO":
                prm = dict(prm, target_x=prm.get("target_x", 0.0) / self.sx,
                           target_y=prm.get("target_y", 0.0) / self.sy)
                c = dict(c, parameters=prm)
            elif isinstance(prm, dict) and c.get("commandType") == "FOLLOW_PLAYER" and "distance" in prm:
                c = dict(c, parameters=dict(prm, distance=prm["distance"] / self.sx))
            out.append(c)
        return out

    def _xy(self, d):
        d = d or {}
        return dict(d, x=d.get("x", 0.0) * self.sx, y=d.get("y", 0.0) * self.sy)

    def _ball(self, b):
        flat = {k: {"x": b[k].get("x", 0.0) * self.sx, "y": b[k].get("z", 0.0) * self.sy}
                for k in ("position", "velocity") if isinstance(b.get(k), dict)}
        return dict(b, **flat)


def make(name: str, deadline: float | None = None):
    """A built-in team by name, or any other team spec (see simulator/external.py)."""
    team = _make(name, deadline)
    if DOC_SCALE_BUILTINS and name in all_team_names():
        team = DocScale(team)
    return team


def _make(name: str, deadline: float | None = None):
    teams = {
        "sample": lambda: RuleBased("sample", SAMPLE),
        "aggressive": lambda: RuleBased("aggressive", AGGRESSIVE),
        "defensive": lambda: RuleBased("defensive", DEFENSIVE),
        "idle": Idle,
    }
    if name not in teams:
        from opponents import OPPONENTS  # formation teams
        if name in OPPONENTS:
            return OPPONENTS[name]()
        from external import make_external
        return make_external(name, deadline)
    return teams[name]()


TEAM_NAMES = ("sample", "aggressive", "defensive", "idle")


def all_team_names() -> tuple:
    """Every built-in name: TEAM_NAMES plus the formation teams in simulator/opponents.py."""
    from opponents import OPPONENTS
    return TEAM_NAMES + tuple(sorted(OPPONENTS))
