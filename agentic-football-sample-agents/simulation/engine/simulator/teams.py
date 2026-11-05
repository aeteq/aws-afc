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


def make(name: str, deadline: float | None = None):
    """A built-in team by name, or any other team spec (see simulator/external.py)."""
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
