"""A starter team for the simulator: copy this file and put your own logic in commands().

    python3 simulator/run_match.py --home mine=py:examples/my_team.py --away aggressive --replay m.html
    python3 simulator/run_match.py --team mine=py:examples/my_team.py --matches 20

commands() is called once per player per turn, exactly as the game server calls each of
your five agents: the same gameState your deployed agent would receive, your teamId (0 =
HOME, attacks +x; 1 = AWAY, attacks -x) and the player you are deciding for (0 GK, 1 DEF,
2 MID, 3 FWD1, 4 FWD2). Return a list of commands; the server plays only the first one.

As it stands this plays exactly like the built-in "sample" team (the AWS sample agents'
rule-based fallback), so you have a working baseline to improve on. If your deployed agent
already has a pure function from game state to commands, point the spec at that instead:
py:path/to/your_code.py:your_function
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from fallback import (DEF_CONFIG, FWD1_CONFIG, FWD2_CONFIG, GK_CONFIG, MID_CONFIG,  # noqa: E402
                      build_fallback)

_PLAYERS = {pid: build_fallback(cfg)
            for pid, cfg in enumerate((GK_CONFIG, DEF_CONFIG, MID_CONFIG, FWD1_CONFIG, FWD2_CONFIG))}


def command(command_type: str, team_id: int, player_id: int, **parameters) -> dict:
    """One command, tagged the way the server checks it (a wrong teamId is rejected)."""
    return {"commandType": command_type, "teamId": team_id, "playerId": player_id,
            "parameters": parameters}


def commands(game_state: dict, team_id: int, player_id: int) -> list:
    # Your logic goes here, for example:
    #   if player_id == 3 and <I have the ball near goal>:
    #       return [command("SHOOT", team_id, player_id, aim_location="BL", power=0.9)]
    return _PLAYERS[player_id](game_state, team_id, player_id)
