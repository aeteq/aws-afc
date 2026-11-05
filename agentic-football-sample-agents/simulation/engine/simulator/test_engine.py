"""Tests for the local match engine — no AWS needed.

    python3 simulator/test_engine.py
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import DT, HALF_X, Match  # noqa: E402
from teams import make  # noqa: E402


def _idle_match(seed=0):
    return Match(make("idle"), make("idle"), seed=seed)


def _hand_ball(m, team, idx, x, y):
    p = m.players[team * 5 + idx]
    p.x, p.y = x, y
    m._gain(p)
    m.ball.protect = 99.0  # no tackles while we test something else
    return p


def _clear_area(m, keep):
    """Move everyone except `keep` out of the way."""
    for q in m.players:
        if not any(q is k for k in keep):
            q.x, q.y = (-20.0 if q.team == 0 else 20.0), 30.0 if q.idx % 2 else -30.0


def _run(m, seconds):
    for _ in range(int(seconds / DT)):
        m._step()
        m.time += DT


def test_payload_matches_the_server_format():
    m = _idle_match()
    gs = m.payload()
    codes = [p["teamCode"] for p in gs["players"]]
    assert codes == ["home"] * 5 + ["away"] * 5, codes
    ids = [p["agentId"] for p in gs["players"]]
    assert ids[:5] == ids[5:] == [f"agentId_{i}" for i in range(5)], ids  # as the server sends them
    assert gs["ball"]["isFree"] is False and gs["ball"]["possessionAgentId"] == "agentId_2"
    assert set(gs) >= {"tick", "gameTime", "playMode", "score", "ball", "players"}
    print("  payload: home first, agentId_N per team, as the server sends it")


def test_goal_counts_and_conceding_team_kicks_off():
    m = _idle_match()
    m.ball.owner = None
    m.ball.x, m.ball.y, m.ball.vx, m.ball.vy = HALF_X - 1.0, 0.5, 20.0, 0.0
    m.ball.last_team = 0
    _run(m, 0.3)
    assert (m.stats[0].goals, m.stats[1].goals) == (1, 0)
    assert m.ball.owner is not None and m.ball.owner.team == 1 and m.ball.owner.idx == 2
    assert abs(m.ball.x) < 1.0
    print("  a ball over the line between the posts scores; the conceding side kicks off")


def test_ground_pass_reaches_an_open_teammate():
    m = _idle_match(seed=3)
    passer = _hand_ball(m, 0, 2, 0.0, 0.0)
    target = m.players[3]
    target.x, target.y = 15.0, 2.0
    _clear_area(m, [passer, target])
    m._apply(0, 2, [{"commandType": "PASS", "playerId": 2, "teamId": 0,
                     "parameters": {"target_player_id": 3, "type": "GROUND"}}])
    _run(m, 3.0)
    assert m.ball.owner is target, m.ball.owner
    assert m.stats[0].passes == 1 and m.stats[0].passes_completed == 1
    print("  an unopposed ground pass arrives and counts as completed")


def test_close_shot_at_an_empty_goal_scores():
    m = _idle_match(seed=5)
    shooter = _hand_ball(m, 0, 3, 47.0, 0.0)
    _clear_area(m, [shooter])
    m._apply(0, 3, [{"commandType": "SHOOT", "playerId": 3, "teamId": 0,
                     "parameters": {"aim_location": "BL", "power": 0.9}}])
    _run(m, 1.0)
    assert m.stats[0].shots == 1 and m.stats[0].goals == 1, m.stats[0]
    print("  an 8 m shot past an absent keeper goes in")


def test_tags_are_enforced_and_offball_commands_wasted():
    m = _idle_match()
    m._apply(0, 3, [{"commandType": "MOVE_TO", "playerId": 3, "teamId": 1,
                     "parameters": {"target_x": 0, "target_y": 0}}])
    m._apply(0, 3, [{"commandType": "SHOOT", "playerId": 3, "teamId": 0, "parameters": {}}])
    m._apply(0, 3, "not a list")
    assert m.stats[0].rejected_commands == 2, m.stats[0]
    assert m.stats[0].wasted_commands == 1, m.stats[0]
    assert m.players[3].order is None
    print("  wrong teamId is rejected; SHOOT without the ball is wasted")


def test_only_the_first_command_per_player_counts():
    m = _idle_match()
    m._apply(0, 3, [{"commandType": "CLEAR_OVERRIDE", "playerId": 3, "teamId": 0, "parameters": {}},
                    {"commandType": "MOVE_TO", "playerId": 3, "teamId": 0,
                     "parameters": {"target_x": 30, "target_y": 0, "sprint": True}}])
    assert m.players[3].order is None and m.stats[0].ignored_commands == 1, m.players[3].order
    print("  [CLEAR_OVERRIDE, MOVE_TO] plays only the CLEAR_OVERRIDE, as the platform does")


def test_pressing_order_ends_when_the_presser_wins_the_ball():
    m = _idle_match()
    p = m.players[6]  # away defender
    m._apply(1, 1, [{"commandType": "PRESS_BALL", "playerId": 1, "teamId": 1,
                     "parameters": {"intensity": 1.0}}])
    assert p.order["type"] == "PRESS_BALL"
    m._gain(p)
    assert p.order is None
    print("  a presser who wins the ball stops pressing and carries it")


def test_keeper_with_the_ball_in_hands_cannot_be_tackled():
    m = _idle_match(seed=9)
    gk = m.players[0]
    gk.x, gk.y = -50.0, 0.0
    m.ball.owner = None
    m.ball.x, m.ball.y, m.ball.vx, m.ball.vy = -50.3, 0.0, 0.0, 0.0
    m._gain(gk)  # picked up in his own box: in his hands
    m.ball.protect = 0.0
    for idx in range(1, 5):
        o = m.players[5 + idx]
        o.x, o.y, o.order = -49.0, (idx - 2.5) * 0.9, {"type": "PRESS_BALL", "intensity": 1.0}
    _run(m, 3.0)
    assert m.ball.owner is gk and m.ball.in_hands, m.ball.owner
    assert -HALF_X <= gk.x < -38.5, gk  # jostled maybe, but holding it in his box, not over the line
    assert m.stats[1].goals == 0

    m2 = _idle_match(seed=9)  # a back-pass must be played with the feet
    gk2, mate = m2.players[0], m2.players[1]
    gk2.x, gk2.y = -50.0, 0.0
    m2.ball.pass_from = mate
    m2._gain(gk2)
    assert not m2.ball.in_hands
    print("  a keeper holding it in his box is safe; from a back-pass he is not")


class _Chaser:
    """A simple team that plays the same as HOME and AWAY and scores, so a mirror match
    measures the engine and nothing else. It reads the ball's owner from the engine itself
    rather than from the payload."""

    name = "chaser"
    match = None

    def commands(self, gs, team, pid):
        me, owner = self.match.players[team * 5 + pid], self.match.ball.owner
        mates = self.match.players[team * 5 + 1:team * 5 + 5]
        ball = self.match.ball
        goal_x = HALF_X if team == 0 else -HALF_X
        tag = {"teamId": team, "playerId": pid}
        if owner is me:
            if abs(goal_x - me.x) < 22:
                return [{"commandType": "SHOOT", **tag, "parameters": {"aim_location": "BL", "power": 0.9}}]
            return [{"commandType": "MOVE_TO", **tag,
                     "parameters": {"target_x": goal_x, "target_y": 0, "sprint": True}}]
        if pid and me is min(mates, key=lambda q: math.hypot(q.x - ball.x, q.y - ball.y)):
            return [{"commandType": "MOVE_TO", **tag,
                     "parameters": {"target_x": ball.x, "target_y": ball.y, "sprint": True}}]
        return []


def test_engine_has_no_home_advantage():
    hw = aw = hg = ag = 0
    for s in range(120):
        home, away = _Chaser(), _Chaser()
        m = Match(home, away, seed=2000 + s)
        home.match = away.match = m
        r = m.run()
        hg, ag = hg + r.score[0], ag + r.score[1]
        hw, aw = hw + (r.score[0] > r.score[1]), aw + (r.score[0] < r.score[1])
    # Goals as a z-score: a fixed ratio is too strict when many goals are scored,
    # too loose when few. |z| < 2.6 is chance 99% of the time on a fair engine.
    z = (hg - ag) / max(1.0, (hg + ag) ** 0.5)
    assert abs(z) < 2.6, (hw, aw, hg, ag, z)
    print(f"  identical teams, 120 matches: home won {hw}, away won {aw}, goals {hg}-{ag} (z = {z:+.2f})")


def test_a_full_match_is_fast():
    t0 = time.perf_counter()
    r = Match(make("aggressive"), make("sample"), seed=1).run()
    dt = time.perf_counter() - t0
    assert dt < 3.0, dt
    assert r.stats[0].rejected_commands == 0 and r.stats[1].rejected_commands == 0
    print(f"  a full 5-minute match simulates in {dt * 1000:.0f} ms; no rejected commands")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"=== {t.__name__} ===")
        t()
    print(f"\nAll {len(tests)} engine tests passed")
