"""Local simulator adapter (not deployed). From the simulator repo:
    python simulator/run_match.py --team kak1=py:<path>/ai-team-kak1/sim_team.py --matches 6
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "lib"))
sys.path.insert(0, os.path.join(_here, "kaklib"))
import tactics  # noqa: E402


def commands(game_state, team_id, player_id):
    return tactics.decide(game_state, team_id, player_id)
