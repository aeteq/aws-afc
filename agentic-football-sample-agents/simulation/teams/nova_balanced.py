"""Stand-in for "The Benchmark FC": the balanced Strands sample team driven by Nova
(Micro/Lite/Pro) on the tiny live pitch. Team spec for the simulator:

    py:<this file>              faithful: perceives possession through lib/state.py, so it
                                has the live agentId bug (HOME player N is picked whenever
                                agentId_N is ambiguous)
    py:<this file>:commands_fixed   same team with possession read correctly (worst case:
                                a workshop team that fixed the bug)

Why it differs from the simulator's "sample" (the lib/fallback.py rules on the doc pitch):
  * Live, every prompt distance is meaningless. Positions arrive in live units (x -7..7)
    but the summary says "Opponent goal at x=55", so the model cannot apply "shoot within
    ~25 units" by the numbers; small Nova models copy the prompt's example instead, and
    the FWD example IS a SHOOT. Live evidence: Benchmark-style goals came from 27-45 doc
    units, 1-2 s after winning the ball. So: forwards shoot from anywhere in the
    attacking half (<50 doc), the MID from <45, otherwise MID plays THROUGH to FWD1
    (its example).
  * "PRESS_BALL high up the pitch" for both forwards and "PRESS_BALL in the middle third"
    for the MID: on a 14-unit pitch everyone is in every third, so MID+FWD1+FWD2 press
    the ball whenever the opponent has it or it is loose. DEF presses in its own half,
    else MARKs (its example: MARK TIGHT).
  * The ball's y in the live payload is height (~0.14), so any MOVE_TO "toward the ball"
    or "between ball and goal" uses y ~ 0: the GK and DEF always sit in the centre lane.
  * Support runs given in doc coordinates (x 30-40) land on the live byline: forwards
    camp near the opponent's goal line when their team has the ball.
  * ~10% of answers are late (Nova Pro MID especially): no command, previous order stays.
"""
import math
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_here, "..", "..", "lib"),):                # agentic-football-sample-agents/lib
    if os.path.isdir(p):
        sys.path.insert(0, p)
from state import _is_my_team, _player_idx, _resolve_holder  # noqa: E402

HALF_X = 55.0
LATE = 0.10          # share of answers that miss the tick
FWD_RANGE = 50.0     # doc units: "shoot on sight" in the attacking half
MID_RANGE = 45.0
BLIND_COPY = 0.5     # carrier that does not know it has the ball still copies its example


def _cmd(kind, pid, tid, params, duration=0):
    return [{"commandType": kind, "playerId": pid, "teamId": tid,
             "parameters": params, "duration": duration}]


def _pos(p):
    q = p.get("position", {})
    return q.get("x", 0.0), q.get("y", 0.0)


def _true_holder(ball, players):
    aid = ball.get("possessionAgentId")
    if not isinstance(aid, str):
        return None
    bx, by = ball.get("position", {}).get("x", 0), ball.get("position", {}).get("y", 0)
    cands = [p for p in players if p.get("agentId") == aid]
    return min(cands, key=lambda p: math.hypot(_pos(p)[0] - bx, _pos(p)[1] - by), default=None)


def _decide(gs, tid, pid, faithful):
    rng = random.Random(hash((gs.get("tick", 0), tid, pid, 7919)))
    if rng.random() < LATE:
        return []
    ball = gs.get("ball", {})
    bx, by = ball.get("position", {}).get("x", 0.0), ball.get("position", {}).get("y", 0.0)
    players = gs.get("players", [])
    s = 1.0 if tid == 0 else -1.0
    my_goal, opp_goal = -s * HALF_X, s * HALF_X
    me = next((p for p in players if _player_idx(p) == pid and _is_my_team(p, tid)), None)
    if me is None:
        return _cmd("SET_STANCE", pid, tid, {"stance": 0})
    mx, my = _pos(me)
    mates = [p for p in players if _is_my_team(p, tid) and _player_idx(p) != pid]
    opps = [p for p in players if not _is_my_team(p, tid)]

    truth = _true_holder(ball, players)
    seen = _resolve_holder(ball, players) if faithful else truth
    if ball.get("isFree") is True:
        seen = truth = None
    i_have = seen is me
    we_have = seen is not None and _is_my_team(seen, tid)
    really_mine = truth is me

    # Carrier the bug blinds (AWAY team): copies its prompt example half the time.
    if really_mine and not i_have:
        if rng.random() < BLIND_COPY:
            if pid in (3, 4):
                return _cmd("SHOOT", pid, tid, {"aim_location": "TR" if pid == 3 else "BL", "power": 0.9})
            if pid == 2:
                return _cmd("PASS", pid, tid, {"target_player_id": 3, "type": "THROUGH"})
            if pid == 0:
                return _cmd("GK_DISTRIBUTE", pid, tid, {"target_player_id": 1, "method": "THROW"})
            return _cmd("MARK", pid, tid, {"target_player_id": 3, "tightness": "TIGHT"}, 5)
        return _cmd("PRESS_BALL", pid, tid, {"intensity": 0.7}, 3)

    d_goal = math.hypot(opp_goal - mx, my)
    if i_have:
        if pid == 0:
            near = min(mates, key=lambda p: math.hypot(_pos(p)[0] - mx, _pos(p)[1] - my))
            return _cmd("GK_DISTRIBUTE", pid, tid, {"target_player_id": _player_idx(near), "method": "THROW"})
        if pid == 1:
            cands = [p for p in mates if _player_idx(p) in (2, 3, 4)]
            near = min(cands, key=lambda p: math.hypot(_pos(p)[0] - mx, _pos(p)[1] - my))
            return _cmd("PASS", pid, tid, {"target_player_id": _player_idx(near), "type": "GROUND"})
        aim = rng.choice(["TL", "TR", "BL", "BR", "CENTER"])
        if pid == 2:
            if d_goal < MID_RANGE:
                return _cmd("SHOOT", pid, tid, {"aim_location": aim, "power": 0.8})
            fw = [p for p in mates if _player_idx(p) in (3, 4)]
            t = min(fw, key=lambda p: abs(_pos(p)[0] - opp_goal))
            return _cmd("PASS", pid, tid, {"target_player_id": _player_idx(t), "type": "THROUGH"})
        if d_goal < FWD_RANGE:
            return _cmd("SHOOT", pid, tid, {"aim_location": aim, "power": 0.9})
        press = min((math.hypot(_pos(o)[0] - mx, _pos(o)[1] - my) for o in opps), default=99)
        if pid == 4 and press < 3.0:
            return _cmd("PASS", pid, tid, {"target_player_id": rng.choice([2, 3]), "type": "GROUND"})
        return _cmd("MOVE_TO", pid, tid, {"target_x": opp_goal * 0.6, "target_y": 0.0, "sprint": True})

    if we_have:  # teammate (as perceived) has it
        if pid in (3, 4):
            return _cmd("MOVE_TO", pid, tid, {"target_x": opp_goal * 0.85,
                                              "target_y": -10.0 if pid == 3 else 10.0, "sprint": True})
        if pid == 2:
            return _cmd("MOVE_TO", pid, tid, {"target_x": max(-50, min(50, bx + s * 12)),
                                              "target_y": 0.0, "sprint": False})
        if pid == 1:
            return _cmd("MOVE_TO", pid, tid, {"target_x": my_goal * 0.5, "target_y": 0.0, "sprint": False})
        return _cmd("MOVE_TO", pid, tid, {"target_x": my_goal * 0.95, "target_y": 0.0, "sprint": False})

    # Opponent has it, or loose: the ball is "in every third" on the live pitch.
    own_half = (bx - 0.0) * s < 0
    if pid in (2, 3, 4):
        return _cmd("PRESS_BALL", pid, tid, {"intensity": 0.8 if pid != 2 else 0.7}, 3)
    if pid == 1:
        if own_half or seen is None:
            if seen is None:
                return _cmd("INTERCEPT", pid, tid, {"aggressive": True}, 3)
            return _cmd("PRESS_BALL", pid, tid, {"intensity": 0.7}, 3)
        danger = min(opps, key=lambda p: abs(_pos(p)[0] - my_goal))
        return _cmd("MARK", pid, tid, {"target_player_id": _player_idx(danger), "tightness": "TIGHT"}, 5)
    # GK: stay on the line, centre lane (live ball y is height); INTERCEPT when it is close
    if seen is None and math.hypot(bx - my_goal, by) < 14.0:
        return _cmd("INTERCEPT", pid, tid, {"aggressive": False}, 2)
    return _cmd("MOVE_TO", pid, tid, {"target_x": my_goal * 0.95, "target_y": 0.0, "sprint": False})


def commands(game_state, team_id, player_id):
    return _decide(game_state, team_id, player_id, faithful=True)


def commands_fixed(game_state, team_id, player_id):
    return _decide(game_state, team_id, player_id, faithful=False)
