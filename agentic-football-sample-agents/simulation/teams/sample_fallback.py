"""The AWS sample teams' rule-based fallback (shared ../../lib/fallback.py).
Balanced, aggressive and defensive share it, so without the model they play the same.
Team spec: py:../teams/sample_fallback.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from fallback import build_fallback, GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG  # noqa: E402

_FB = {i: build_fallback(c) for i, c in enumerate((GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG))}


def commands(game_state, team_id, player_id):
    return _FB[player_id](game_state, team_id, player_id)
