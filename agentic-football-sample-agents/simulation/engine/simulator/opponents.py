"""More opponents: classic 5-a-side formations.

  box       2-2   two at the back, two up
  diamond   1-2-1 one back, two wide in midfield, one up top
  pyramid   2-1-1 two at the back, one in midfield, one up top
  3-1       3-1   three deep, one target forward
  (the "sample" team plays a 1-1-2 "Y")

All are built from lib/fallback.py like the sample teams, so they share its behaviour
apart from shape and pressing range.
"""

from dataclasses import replace

from fallback import DEF_CONFIG, FWD1_CONFIG, FWD2_CONFIG, GK_CONFIG, MID_CONFIG
from teams import RuleBased


def _at(cfg, ref, factor, y, **kw):
    return replace(cfg, default_x_ref=ref, default_x_factor=factor, default_y=y, **kw)


BOX = {  # 2-2
    0: GK_CONFIG,
    1: _at(DEF_CONFIG, "my_goal", 0.6, -10, mark_threshold=30.0),
    2: _at(MID_CONFIG, "my_goal", 0.6, 10, possession_action="PASS", press_distance=15.0),
    3: _at(FWD1_CONFIG, "opp_goal", 0.3, -10),
    4: _at(FWD2_CONFIG, "opp_goal", 0.3, 10),
}

DIAMOND = {  # 1-2-1
    0: GK_CONFIG,
    1: _at(DEF_CONFIG, "my_goal", 0.65, 0, mark_threshold=30.0),
    2: _at(MID_CONFIG, "my_goal", 0.15, 12),
    3: _at(FWD1_CONFIG, "my_goal", 0.15, -12, support_x_factor=0.4),
    4: _at(FWD2_CONFIG, "opp_goal", 0.5, 0),
}

PYRAMID = {  # 2-1-1
    0: GK_CONFIG,
    1: _at(DEF_CONFIG, "my_goal", 0.65, -8, mark_threshold=30.0),
    2: _at(MID_CONFIG, "my_goal", 0.65, 8, possession_action="PASS", press_distance=15.0),
    3: _at(FWD1_CONFIG, "opp_goal", 0.0, 0, support_x_factor=0.3),
    4: _at(FWD2_CONFIG, "opp_goal", 0.55, 0),
}

THREE_ONE = {  # 3-1
    0: GK_CONFIG,
    1: _at(DEF_CONFIG, "my_goal", 0.7, 0, mark_threshold=35.0),
    2: _at(MID_CONFIG, "my_goal", 0.55, -14, possession_action="PASS", press_distance=15.0),
    3: _at(FWD1_CONFIG, "my_goal", 0.55, 14, possession_action="PASS", shoot_threshold=20.0,
           press_distance=15.0),
    4: _at(FWD2_CONFIG, "opp_goal", 0.45, 0),
}

FORMATIONS = {"box": BOX, "diamond": DIAMOND, "pyramid": PYRAMID, "3-1": THREE_ONE}

OPPONENTS = {name: (lambda n, c: lambda: RuleBased(n, c))(name, cfgs)
             for name, cfgs in FORMATIONS.items()}
