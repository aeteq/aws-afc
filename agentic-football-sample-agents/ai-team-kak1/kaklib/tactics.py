"""Deterministic role tactics. Used as the offline team in the simulator AND as the
fallback when the LLM is late or returns something invalid.

decide(game_state, team_id, player_id) -> [command]  (one command, tagged correctly)

Player ids and roles: see FORMATION below (default 1-2-1: 0 GK, 1 DEF, 2+3 MID, 4 FWD).
Pitch: x -55..55, y -35..35. HOME (team 0) attacks +x, AWAY (team 1) attacks -x.
"""

import math
import os
import re

from state import _is_my_team, _player_idx

HALF_X, HALF_Y = 55.0, 35.0     # the pitch our rules are tuned for (the documented one)

# The live pitch is ~8x smaller than documented in BOTH directions. 132 live turns at 1-decimal
# precision (memory team's stored summaries, 2026-11-05): every player and the ball stayed within
# x -6.9..6.6 and y -3.5..3.6. We scale positions up to the documented pitch on the way in and
# scale MOVE_TO targets back down on the way out, so every distance in this file keeps its
# meaning. PITCH_HALF_X=55 (simulator/tests) = no scaling; PITCH_HALF_Y follows the same ratio.
PITCH_HALF_X = float(os.getenv("PITCH_HALF_X", "7.0"))
PITCH_HALF_Y = float(os.getenv("PITCH_HALF_Y", str(HALF_Y * PITCH_HALF_X / HALF_X)))   # ~4.45 live
SCALE_X = HALF_X / PITCH_HALF_X
SCALE_Y = HALF_Y / PITCH_HALF_Y


# The BALL's sideways axis is z live (y is its height: ~0.14 at rest, up to ~1.7 when lofted).
# The local simulator uses y sideways and z = 0, so sim adapters set BALL_LATERAL=y.
BALL_LATERAL = os.getenv("BALL_LATERAL", "z")


def _ball_flat(d):
    """Ball position/velocity as {x, y} with y = sideways (z when live and present)."""
    if not isinstance(d, dict):
        return d
    if BALL_LATERAL == "z" and "z" in d:
        return dict(d, y=d.get("z", 0.0), height=d.get("y", 0.0))
    return d


def _doc_scale_payload(gs):
    """Kick-off payloads come on the documented scale (GK at x -50) while open play is ~8x
    smaller (GK at ~-6 two seconds later). Nothing on the live pitch is beyond |x| 10."""
    pts = [p.get("position") or {} for p in gs.get("players") or []]
    return any(abs(q.get("x", 0.0)) > 10.0 for q in pts)


def _scale_state(gs):
    """Copy of the game state with the ball made flat (y = sideways) and every x and y
    (positions, velocities) on the documented scale."""
    if not gs:
        return gs
    gs = dict(gs)
    ball = dict(gs.get("ball") or {})
    for k in ("position", "velocity"):
        if k in ball:
            ball[k] = _ball_flat(ball[k])
    gs["ball"] = ball
    if (SCALE_X == 1.0 and SCALE_Y == 1.0) or _doc_scale_payload(gs):
        return gs
    def sx(d):
        if not isinstance(d, dict):
            return d
        return dict(d, x=d.get("x", 0.0) * SCALE_X, y=d.get("y", 0.0) * SCALE_Y)
    out = dict(gs)
    ball = dict(gs.get("ball") or {})
    for k in ("position", "velocity"):
        if k in ball:
            ball[k] = sx(ball[k])
    out["ball"] = ball
    out["players"] = [dict(p, position=sx(p.get("position") or {}), velocity=sx(p.get("velocity") or {}))
                      for p in gs.get("players") or []]
    return out


def _real_x(x):
    return x / SCALE_X


def _real_y(y):
    return y / SCALE_Y
# Formation. Roles come from the player id the server sends, not the agent folder name.
#   1-2-1: 0 GK, 1 DEF, 2 MID (left), 3 MID (right), 4 FWD (centre)
#   1-1-2: 0 GK, 1 DEF, 2 MID (centre), 3 FWD (left), 4 FWD (right)
FORMATION = os.getenv("FORMATION", "1-2-1")
if FORMATION == "1-1-2":
    ROLE = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD", 4: "FWD"}
    SIDE = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: -1.0}     # +1 = left (attacking view)
else:
    ROLE = {0: "GK", 1: "DEF", 2: "MID", 3: "MID", 4: "FWD"}
    SIDE = {0: 0.0, 1: 0.0, 2: 1.0, 3: -1.0, 4: 0.0}

# Tunables. Distances are to the opponent goal centre.
# Live: our 7 shots from 33-44 doc out scored 0, while The Benchmark scored from 27-35 doc
# (Pan's Claude live_report, 3 matches). So: not from 40, but 30ish is fine. Bench: no regression.
SHOOT_RANGE = {"DEF": 20.0, "MID": 30.0, "FWD": 32.0}
# Shoot on sight. In the sim this broke the 0-0 mirror (10W 6D 0L vs the plain plan).
# Sim shot physics are guesses: if long shots just give the ball away live, set LONG_SHOTS=0.
LONG_SHOTS = os.getenv("LONG_SHOTS", "0") == "1"        # off: long shots never scored live
LONG_SHOT_RANGE = {"DEF": 25.0, "MID": 38.0, "FWD": 40.0}
OWN_BOX = 18.0           # "our penalty area": within this of our goal centre (documented scale)
DEF_THIRD = 37.0         # depth from our goal line that counts as our defensive third
PRESSURE_DIST = 4.0      # an opponent this close = under pressure
SPRINT_MIN_STAMINA = 0.35

# Score/clock awareness. Set MATCH_SECONDS=120 if matches turn out to be 2 minutes.
MATCH_SECONDS = float(os.getenv("MATCH_SECONDS", "120"))   # live matches end at ~120 s (logs)
LATE_FRACTION = 0.2      # last 20% of the match counts as "late"
CHASE_SHOOT_BONUS = 4.0  # shoot from further when chasing a goal

# Adaptive style: read the opponent from each snapshot and switch.
# All 5 agents see the same snapshot and run this code, so they agree without talking.
KEEPER_OUT_ON, KEEPER_OUT_OFF = 8.0, 5.0   # their GK this far off his line = shoot from range
KEEPER_OUT_SHOOT_BONUS = 4.0
COMMITTED_ON, COMMITTED_OFF = 3, 1         # their outfielders in our half -> counter
PRESSED_DIST, PRESSED_COUNT = 8.0, 2       # this many opponents near our carrier -> release long
STYLES = ("auto", "normal", "counter", "direct", "patient")
# All 10 players (opponents too) hear every coach message, so a STYLE order only counts
# if the message also contains our tag, e.g. "kak1 STYLE counter". Empty tag = no check.
COACH_TAG = os.getenv("COACH_TAG", "kak1").lower()

# Last flags per (team, player). Lets thresholds be sticky (no flip-flopping).
# May not survive between calls on AgentCore; the code works the same without it.
_memory = {}


def coach_messages(chat, team_id):
    """Our coach's messages, oldest first. All 10 players hear every message, so a message
    only counts if it carries COACH_TAG (and, when the payload says, our team)."""
    my_codes = {str(team_id), "home" if team_id == 0 else "away"}
    out = []
    for m in chat or []:
        if isinstance(m, dict):
            team = m.get("teamId", m.get("teamCode"))
            if team is not None and str(team).lower() not in my_codes:
                continue
            text = str(m.get("message") or m.get("text") or m.get("content") or "")
        else:
            text = str(m)
        if COACH_TAG and COACH_TAG not in text.lower():
            continue
        out.append(text)
    return out


def _coach_style(chat, team_id):
    """Coach override: our message containing 'STYLE counter' (or auto/normal/direct/patient)."""
    for text in reversed(coach_messages(chat, team_id)):
        hit = re.search(r"\bSTYLE\s+(" + "|".join(STYLES) + r")\b", text, re.I)
        if hit:
            return hit.group(1).lower()
    return None


def cmd(kind, team_id, pid, **params):
    return {"commandType": kind, "teamId": team_id, "playerId": pid,
            "parameters": params, "duration": 0}


def _pos(p):
    q = p.get("position", {}) if p else {}
    return q.get("x", 0.0), q.get("y", 0.0)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _stamina(p):
    """Server docs say 0-100, the sample lib says 0-1. Accept both."""
    s = p.get("stamina", 100.0) if p else 100.0
    return s / 100.0 if s > 1.0 else s


def _lane_clear(a, b, opps):
    """Smallest distance from any opponent to the segment a->b (bigger = safer pass)."""
    ax, ay = a
    bx, by = b
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    best = 99.0
    for o in opps:
        ox, oy = o
        t = _clamp(((ox - ax) * (bx - ax) + (oy - ay) * (by - ay)) / L2, 0.0, 1.0)
        best = min(best, math.hypot(ox - (ax + t * (bx - ax)), oy - (ay + t * (by - ay))))
    return best


class View:
    """The game state from one player's point of view."""

    def __init__(self, gs, team_id, pid, adaptive=True, force=None):
        gs = _scale_state(gs)                                       # live pitch -> documented scale
        self.team_id, self.pid, self.role = team_id, pid, ROLE.get(pid, "MID")
        self.side = SIDE.get(pid, 0.0)                              # +1 left, -1 right, 0 centre
        self.s = 1.0 if team_id == 0 else -1.0                    # attack direction
        self.opp_goal = (HALF_X * self.s, 0.0)
        self.my_goal = (-HALF_X * self.s, 0.0)
        players = gs.get("players", [])
        self.mates = {_player_idx(p): p for p in players if _is_my_team(p, team_id)}
        self.opps = {_player_idx(p): p for p in players if not _is_my_team(p, team_id)}
        self.me = self.mates.get(pid)
        ball = gs.get("ball", {})
        self.ball = _pos(ball)
        v = ball.get("velocity", {})
        self.ball_v = (v.get("x", 0.0), v.get("y", 0.0))
        self.holder = self._holder(ball, players)
        self.holder_mine = self.holder is not None and _is_my_team(self.holder, team_id)
        self.i_have_ball = self.holder_mine and _player_idx(self.holder) == pid
        self.chat = gs.get("teamChat") or []
        self.mode = self._mode(gs)
        self.keeper_out = self.pressed = False
        self.style = "normal"
        if adaptive and self.me is not None:
            self._read_opponent(gs, force)

    def _read_opponent(self, gs, force=None):
        """Set self.style (normal|counter|direct|patient), self.keeper_out, self.pressed."""
        key = (self.team_id, self.pid)
        t = gs.get("gameTime", 0) or 0
        prev = _memory.get(key)
        if prev is None or t < prev["t"]:          # first call, or a new match started
            prev = {"t": t, "flags": {}}
        was = prev["flags"]

        def sticky(name, value, on, off):
            return value >= (off if was.get(name) else on)

        outfield = [_pos(p) for i, p in self.opps.items() if i != 0]
        in_our_half = sum(1 for o in outfield if o[0] * self.s < 0)
        in_their_half = len(outfield) - in_our_half
        gk = self.opps.get(0)
        gk_off_line = abs(_pos(gk)[0] - self.opp_goal[0]) if gk else 0.0

        committed = sticky("committed", in_our_half, COMMITTED_ON, COMMITTED_OFF)
        self.keeper_out = sticky("keeper_out", gk_off_line, KEEPER_OUT_ON, KEEPER_OUT_OFF)
        # Deep block can only be judged while we have the ball; otherwise keep the last reading.
        deep = (in_their_half >= len(outfield) and len(outfield) > 0) if self.holder_mine \
            else bool(was.get("deep"))
        if self.i_have_ball:
            me = _pos(self.me)
            self.pressed = sum(1 for o in self.opp_positions() if _d(o, me) < PRESSED_DIST) >= PRESSED_COUNT

        forced = force or _coach_style(self.chat, self.team_id)
        if forced and forced != "auto":
            self.style = forced
        elif committed:
            self.style = "counter"
        elif deep:
            self.style = "patient"
        if self.style == "direct":
            self.pressed = self.i_have_ball            # forced direct: always release early

        _memory[key] = {"t": t, "flags": {"committed": committed, "keeper_out": self.keeper_out,
                                          "deep": deep}}

    def _mode(self, gs):
        """'chase' (losing late, or 2+ down after half time), 'protect' (winning late), else 'normal'."""
        score = gs.get("score") or {}
        mine, theirs = score.get("home", 0), score.get("away", 0)
        if self.team_id == 1:
            mine, theirs = theirs, mine
        t = gs.get("gameTime", 0) or 0
        late = t >= MATCH_SECONDS * (1 - LATE_FRACTION)
        if mine < theirs and (late or (theirs - mine >= 2 and t >= MATCH_SECONDS / 2)):
            return "chase"
        if mine > theirs and late:
            return "protect"
        return "normal"

    def _holder(self, ball, players):
        """Who holds the ball. agentId repeats across teams (agentId_3 exists twice), and the
        stock lib then always picks HOME. We pick the candidate nearest the ball instead."""
        if ball.get("isFree") is True:
            return None
        aid = ball.get("possessionAgentId")
        if isinstance(aid, str):
            cands = [p for p in players if p.get("agentId") == aid]
        else:
            pid = ball.get("possessionPlayerId")
            if pid is None:
                # Older documented format: possession only as players[].hasBall.
                cands = [p for p in players if p.get("hasBall") is True]
                return min(cands, key=lambda p: _d(_pos(p), self.ball)) if cands else None
            cands = [p for p in players if p.get("playerId") == pid]
            if ball.get("possessionTeamId") is not None:
                cands = [p for p in cands if p.get("teamId") == ball["possessionTeamId"]]
        if not cands:
            return None
        return min(cands, key=lambda p: _d(_pos(p), self.ball))

    # --- small helpers ---
    def c(self, kind, **params):
        if kind == "MOVE_TO" and "target_x" in params:            # documented scale -> live pitch
            params["target_x"] = round(_real_x(params["target_x"]), 2)
            params["target_y"] = round(_real_y(params["target_y"]), 2)
        return [cmd(kind, self.team_id, self.pid, **params)]

    def fwd(self, x):
        """x measured from my own goal line toward the opponent (0 = my goal, 110 = theirs)."""
        return self.my_goal[0] + self.s * x

    def nearest_mate_to(self, pt, outfield=True):
        pool = [(i, p) for i, p in self.mates.items() if not (outfield and i == 0)]
        return min(pool, key=lambda ip: _d(_pos(ip[1]), pt))[0] if pool else None

    def opp_positions(self):
        return [_pos(p) for p in self.opps.values()]

    def pressure(self):
        me = _pos(self.me)
        return min((_d(me, o) for o in self.opp_positions()), default=99.0)

    def sprint(self):
        # Chasing: spend the stamina, there is nothing to save it for.
        floor = 0.15 if self.mode == "chase" else SPRINT_MIN_STAMINA
        return _stamina(self.me) > floor

    def shoot_range(self):
        long_ok = LONG_SHOTS and self.mode != "protect"   # winning late: keep the ball
        r = (LONG_SHOT_RANGE if long_ok else SHOOT_RANGE).get(self.role, 20.0)
        if self.mode == "chase":
            r += CHASE_SHOOT_BONUS
        if self.keeper_out and self.role != "DEF" and self.mode != "protect":
            r += KEEPER_OUT_SHOOT_BONUS
        return min(r, 36.0)                # past this, shots are just giveaways (live data)

    # --- the single press/shoot policy: rules and model are both held to it ---
    def press_allowed(self):
        """Press only when the opponent has the ball. Never the GK, never the DEF in their half.
        At most the 2 nearest eligible outfielders (3 when chasing): avoids the 'ball magnet'
        (practice match 1: 152 model presses out of ~390 outfield commands)."""
        if self.pid == 0 or self.holder is None or self.holder_mine:
            return False
        their_half = (self.ball[0] * self.s) > 0

        def eligible(i):
            return i != 0 and not (ROLE.get(i) == "DEF" and their_half)
        if not eligible(self.pid):
            return False
        order = sorted((i for i in self.mates if eligible(i)), key=lambda i: _d(_pos(self.mates[i]), self.ball))
        rank = order.index(self.pid) if self.pid in order else 99
        return rank < (3 if self.mode == "chase" else 2)

    def shot_allowed(self):
        if not self.i_have_ball:
            return False
        me = _pos(self.me)
        # in range of their goal, or a clearance out of our own box
        return _d(me, self.opp_goal) < self.shoot_range() or _d(me, self.my_goal) < OWN_BOX

    def allows(self, cmd):
        kind = cmd.get("commandType")
        if kind == "PRESS_BALL":
            return self.press_allowed()
        if kind == "SHOOT":
            return self.shot_allowed()
        return True

    def most_advanced_mate(self, min_lane=2.0):
        """Furthest-forward outfield teammate with a clear-enough lane, or None."""
        me, opps = _pos(self.me), self.opp_positions()
        best = None
        for i, p in self.mates.items():
            if i in (self.pid, 0):
                continue
            tp = _pos(p)
            if (tp[0] - me[0]) * self.s < 5 or _lane_clear(me, tp, opps) < min_lane:
                continue
            if best is None or tp[0] * self.s > _pos(self.mates[best])[0] * self.s:
                best = i
        return best


# --- on the ball -------------------------------------------------------------------------

def _shot(v):
    """Aim low at the corner away from the opponent keeper. L/R is from the shooter's view:
    attacking +x, 'L' is +y."""
    gk = v.opps.get(0)
    gk_y = _pos(gk)[1] if gk else 0.0
    want_y = -1.0 if gk_y > 0 else 1.0          # far side from the keeper
    lateral = want_y * v.s
    return v.c("SHOOT", aim_location="BL" if lateral > 0 else "BR", power=1.0)


GROUND_LANE_MIN = 6.0    # ground pass only with this much lane (doc units) ...
GROUND_DIST_MAX = 15.0   # ... and this short. Live: 21 of 27 ground passes lost within 2 ticks.


def _pass_type(v, bp):
    """THROUGH to a forward with a clear lane; GROUND only when short and very clear; else AERIAL
    (live: AERIAL 5 of 5 not lost immediately; GROUND 21 of 27 lost)."""
    i, _, lane = bp
    dist = _d(_pos(v.me), _pos(v.mates[i]))
    if ROLE.get(i) == "FWD" and lane > 5:
        return "THROUGH"
    if lane >= GROUND_LANE_MIN and dist <= GROUND_DIST_MAX:
        return "GROUND"
    return "AERIAL"


def _best_pass(v, min_lane=2.5):
    """Teammate that is open and ahead. Returns (pid, score, lane) or None."""
    me = _pos(v.me)
    opps = v.opp_positions()
    best = None
    for i, p in v.mates.items():
        if i in (v.pid, 0):
            continue
        tp = _pos(p)
        dist = _d(me, tp)
        if dist < 5 or dist > 35:
            continue
        lane = _lane_clear(me, tp, opps)
        if lane < min_lane:
            continue
        progress = (tp[0] - me[0]) * v.s                    # metres gained toward goal
        space = min((_d(tp, o) for o in opps), default=20.0)
        score = progress + 1.5 * min(space, 10) + lane
        if best is None or score > best[1]:
            best = (i, score, lane)
    return best


def _clear_own_box(v):
    """Ball in our box: get it out. Lofted pass to a free teammate outside the box, otherwise
    a full-power SHOOT upfield as a clearance (how we were conceding: ball stuck near our goal)."""
    me, opps = _pos(v.me), v.opp_positions()
    outs = [i for i in v.mates if i not in (0, v.pid)
            and _d(_pos(v.mates[i]), v.my_goal) >= OWN_BOX
            and _lane_clear(me, _pos(v.mates[i]), opps) >= 2.0]
    if outs:
        best = max(outs, key=lambda i: min((_d(_pos(v.mates[i]), o) for o in opps), default=20))
        return v.c("PASS", target_player_id=best, type="AERIAL")
    return v.c("SHOOT", aim_location="CENTER", power=1.0)


def _on_ball(v):
    me = _pos(v.me)
    if _d(me, v.my_goal) < OWN_BOX:
        return _clear_own_box(v)
    to_goal = _d(me, v.opp_goal)
    if to_goal < v.shoot_range():
        return _shot(v)
    # Direct: 2+ opponents on us, release long. Counter: won it in our half, hit the
    # furthest forward before they get back.
    own_half = me[0] * v.s < 0
    if v.pressed or (v.style == "counter" and own_half):
        t = v.most_advanced_mate()
        if t is not None:
            far = _d(me, _pos(v.mates[t])) > 25
            return v.c("PASS", target_player_id=t, type="AERIAL" if far else "THROUGH")
    pressured = v.pressure() < PRESSURE_DIST
    bp = _best_pass(v)
    # Defenders move it on quickly (except when patient); others pass when pressured or a
    # mate is clearly better placed.
    quick_def = v.role == "DEF" and v.style != "patient"
    if bp and (quick_def or pressured or bp[1] > 18):
        kind = _pass_type(v, bp)
        return v.c("PASS", target_player_id=bp[0], type=kind)
    if pressured and v.role == "DEF":
        # Under pressure, nobody clearly open: first a nearby midfielder with a half-clear
        # lane (lofted over the presser), only then hoof it to the furthest forward.
        me = _pos(v.me)
        near = [i for i in v.mates if ROLE.get(i) == "MID" and _d(me, _pos(v.mates[i])) < 40
                and _lane_clear(me, _pos(v.mates[i]), v.opp_positions()) >= 1.5]
        if near:
            best = max(near, key=lambda i: min((_d(_pos(v.mates[i]), o) for o in v.opp_positions()), default=20))
            return v.c("PASS", target_player_id=best, type="AERIAL")
        far = max((i for i in v.mates if i != v.pid and i != 0),
                  key=lambda i: _pos(v.mates[i])[0] * v.s)
        return v.c("PASS", target_player_id=far, type="AERIAL")
    # Dribble toward a shooting spot, drifting away from the nearest defender.
    near_opp = min(v.opp_positions(), key=lambda o: _d(o, me), default=(0, 0))
    ty = _clamp(me[1] + (3.0 if near_opp[1] < me[1] else -3.0), -12, 12)
    tx = v.opp_goal[0] - v.s * 14.0
    return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(ty, 1), sprint=v.sprint())


# --- goalkeeper --------------------------------------------------------------------------

def _gk(v):
    if v.i_have_ball:
        if v.style == "counter":                  # long kick to the forward left high
            t = v.most_advanced_mate()
            if t is not None:
                return v.c("GK_DISTRIBUTE", target_player_id=t, method="KICK")
        me = _pos(v.me)
        opps = v.opp_positions()
        cands = [(i, p) for i, p in v.mates.items() if i != 0]
        def openness(ip):
            tp = _pos(ip[1])
            return min((_d(tp, o) for o in opps), default=20) + _lane_clear(me, tp, opps)
        target, p = max(cands, key=openness)
        method = "THROW" if _d(me, _pos(p)) < 25 else "KICK"
        if v.pressure() < 2 * PRESSURE_DIST:            # opponent close: kick it long, out of danger
            far = v.most_advanced_mate(min_lane=1.0)
            if far is not None:
                target, method = far, "KICK"
        return v.c("GK_DISTRIBUTE", target_player_id=target, method=method)
    # Loose ball close to goal and we'd get there first: go and claim it.
    bd = _d(v.ball, v.my_goal)
    if v.holder is None and bd < 14:
        mine = _d(_pos(v.me), v.ball)
        theirs = min((_d(o, v.ball) for o in v.opp_positions()), default=99)
        if mine < theirs:
            return v.c("INTERCEPT", aggressive=True)
    # Stay on the line, shading toward the ball; step out a little when it is close.
    step = 4.0 if bd < 25 else 2.0
    ty = _clamp(v.ball[1] * 0.15, -3.0, 3.0)
    return v.c("MOVE_TO", target_x=round(v.fwd(step), 1), target_y=round(ty, 1), sprint=bd < 25)


# --- off the ball ------------------------------------------------------------------------

def _support(v):
    """A teammate has the ball: get open."""
    bx, by = v.ball
    if v.role == "FWD":
        side = v.side                      # 0 = lone central striker
        if v.mode == "protect":
            # Winning late: show for short passes near halfway instead of running in behind.
            return v.c("MOVE_TO", target_x=round(v.fwd(65), 1), target_y=12.0 * side * v.s, sprint=False)
        return v.c("MOVE_TO", target_x=round(v.fwd(94), 1), target_y=10.0 * side * v.s, sprint=v.sprint())
    if v.role == "MID":
        tx = _clamp(bx - v.s * 6.0, -45, 45)
        if v.mode == "chase":
            tx = v.fwd(85)                 # arrive at the edge of the box for a shot
        elif v.style == "patient":
            tx = v.fwd(80)                 # they sit deep: join the attack
        ty = -by * 0.5
        if v.side:
            # Two mids: each holds a channel (y = +-12). Ball-side mid sits just behind the ball
            # (safe option), far-side mid moves ahead of it (forward option).
            ty = 12.0 * v.side * v.s
            ball_my_side = (by * v.s) * v.side > 0
            if v.mode not in ("chase",) and v.style != "patient" and not ball_my_side:
                tx = _clamp(bx + v.s * 4.0, -45, 45)
        # Corner trap fix: ball in our defensive third -> mids stay AHEAD of it as outlets
        # (practice 2: both mids dropped to x -45 next to our DEF, so he could only hoof it).
        ball_depth = (bx - v.my_goal[0]) * v.s
        if ball_depth < DEF_THIRD:
            ahead = 12.0 if (v.side == 0 or (by * v.s) * v.side > 0) else 20.0   # ball-side mid closer
            tx = v.fwd(max((tx - v.my_goal[0]) * v.s, ball_depth + ahead, 25.0))
            if v.side and (by * v.s) * v.side > 0:
                ty = _clamp(by * 0.6, -30, 30)       # ball-side mid: short diagonal option
        return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(ty, 1), sprint=v.mode == "chase")
    # DEF: hold a safe line behind the ball. Chasing: join the attack as an extra midfielder.
    if v.mode == "chase":
        gap, cap = 10.0, 75.0
    elif v.style == "patient":
        gap, cap = 15.0, 55.0              # they sit deep: hold the halfway line
    else:
        gap, cap = 25.0, 45.0
    tx = v.fwd(_clamp((bx - v.my_goal[0]) * v.s - gap, 18.0, cap))
    return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(by * 0.3, 1), sprint=False)


def _defend(v):
    """Opponent has the ball, or it is loose."""
    me = _pos(v.me)
    chaser = v.nearest_mate_to(v.ball)
    loose = v.holder is None

    if loose:
        if chaser == v.pid:
            # Run to where the ball will be in half a second.
            tx = v.ball[0] + v.ball_v[0] * 0.5
            ty = v.ball[1] + v.ball_v[1] * 0.5
            return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(ty, 1), sprint=True)
        if v.role == "FWD":
            return _hold_high(v)
        if v.role == "MID" and _d(me, v.ball) < 15:
            return v.c("INTERCEPT", aggressive=True)
        return _shape(v)

    # Opponent has the ball. Every press below also needs v.press_allowed() (one policy).
    if chaser == v.pid and v.press_allowed():
        return v.c("PRESS_BALL", intensity=1.0)
    ball_in_our_half = (v.ball[0] - 0.0) * v.s < 0
    if v.role == "DEF":
        threat = _most_dangerous_free_opp(v)
        if threat is not None:
            return v.c("MARK", target_player_id=threat, tightness="TIGHT")
        return _shape(v)
    if v.role == "MID":
        # Chasing: press anywhere. Otherwise only in our half.
        if (ball_in_our_half or v.mode == "chase") and _d(me, v.ball) < 15 and v.press_allowed():
            return v.c("PRESS_BALL", intensity=0.8)
        return _shape(v)
    # Forwards: second presser high up the pitch, otherwise wait for the counter.
    if v.mode == "protect":
        return _shape(v)                   # winning late: drop back and block passes
    if v.style == "counter" and v.mode != "chase":
        return _hold_high(v, depth=72)     # don't press: stay up as the outlet
    radius = 20.0 if v.mode == "chase" else 12.0
    if not ball_in_our_half and _d(me, v.ball) < radius and v.press_allowed():
        return v.c("PRESS_BALL", intensity=1.0 if v.mode == "chase" else 0.8)
    return _hold_high(v)


def _most_dangerous_free_opp(v):
    """Opponent (not the ball holder) closest to our goal, if he is in our half."""
    holder_idx = _player_idx(v.holder) if v.holder is not None and not v.holder_mine else None
    cands = [(i, p) for i, p in v.opps.items() if i != 0 and i != holder_idx]
    if not cands:
        return None
    i, p = min(cands, key=lambda ip: _d(_pos(ip[1]), v.my_goal))
    return i if _d(_pos(p), v.my_goal) < 45 else None


def _shape(v):
    """Default defensive position between ball and own goal."""
    bx, by = v.ball
    depth = {"DEF": 0.35, "MID": 0.6}.get(v.role, 0.8)
    if v.style == "counter" and v.role == "MID":
        depth = 0.45                       # sit deeper, absorb, then break
    tx = v.my_goal[0] + (bx - v.my_goal[0]) * depth
    ty = by * (0.4 if v.role == "DEF" else 0.6)
    if v.role == "MID":
        ty = _clamp(ty + 8.0 * v.side * v.s, -30, 30)   # two mids: one each side of the ball
    return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(ty, 1), sprint=False)


def _hold_high(v, depth=60):
    """Forwards stay near halfway (or higher), split wide, ready for a counter."""
    y = 8.0 * v.side * v.s                 # lone striker (side 0) stays central
    if depth > 60:
        y *= 1.5                           # wider when staying high, harder to mark both
    return v.c("MOVE_TO", target_x=round(v.fwd(depth), 1), target_y=y, sprint=False)


# --- entry -------------------------------------------------------------------------------

# --- options for the model (USE_MODEL=choice) ---------------------------------------------

def candidates(game_state, team_id, player_id):
    """2-4 safe options for an outfield player WITH the ball, when no hard rule decides
    (not in our box, no shot in range). Each: {"label", "cmd", "info"}; info is in LIVE units.
    The first option is always the rule's own choice. Empty list = don't ask the model."""
    v = View(game_state, team_id, player_id)
    if v.me is None or v.pid == 0 or not v.i_have_ball:
        return []
    me = _pos(v.me)
    if _d(me, v.my_goal) < OWN_BOX or _d(me, v.opp_goal) < v.shoot_range():
        return []
    opps = v.opp_positions()

    def free(pt):
        return min((_d(pt, o) for o in opps), default=20.0)

    def live(d):                      # documented distance -> live units (x scale; y is close)
        return round(d / SCALE_X, 2)

    out, seen = [], set()

    def add(label, cmd):
        c = cmd[0]
        p = c["parameters"]
        key = (c["commandType"], p.get("target_player_id"), p.get("type"),
               round(p.get("target_x", 0), 0), round(p.get("target_y", 0), 0))
        if key in seen:
            return
        seen.add(key)
        p = c["parameters"]
        info = {}
        if c["commandType"] == "PASS":
            tp = _pos(v.mates[p["target_player_id"]])
            info = {"to": ROLE.get(p["target_player_id"]) + str(p["target_player_id"]),
                    "dist": live(_d(me, tp)), "lane": live(_lane_clear(me, tp, opps)),
                    "receiverFree": live(free(tp)), "gain": live((tp[0] - me[0]) * v.s)}
        elif c["commandType"] == "MOVE_TO":
            info = {"gain": round(((p["target_x"] - _real_x(me[0])) * v.s), 2), "nearestOpp": live(free(me))}
        out.append({"label": label, "cmd": cmd, "info": info})

    add("rule", _on_ball(v))
    bp = _best_pass(v)
    if bp:
        add("best pass", v.c("PASS", target_player_id=bp[0], type=_pass_type(v, bp)))
    mates = [i for i in v.mates if i not in (0, v.pid)]
    if mates:
        safe = max(mates, key=lambda i: _lane_clear(me, _pos(v.mates[i]), opps) + free(_pos(v.mates[i])))
        add("safe pass", v.c("PASS", target_player_id=safe, type="AERIAL"))
        fwd = [i for i in mates if ROLE.get(i) == "FWD"]
        if fwd and _lane_clear(me, _pos(v.mates[fwd[0]]), opps) > 3.0:
            add("through ball", v.c("PASS", target_player_id=fwd[0], type="THROUGH"))
    # Dribble only with room: live, dribbles were lost 6 of 8 times (review note).
    if free(me) > PRESSURE_DIST:
        near_opp = min(opps, key=lambda o: _d(o, me), default=(0, 0))
        ty = _clamp(me[1] + (3.0 if near_opp[1] < me[1] else -3.0), -12, 12)
        add("dribble", v.c("MOVE_TO", target_x=round(me[0] + v.s * 12.0, 1), target_y=round(ty, 1), sprint=v.sprint()))
    return out[:4]


def decide(game_state, team_id, player_id, adaptive=True, force=None):
    """adaptive=False: the plain plan (no opponent read). force: fix a style (sim A/B tests)."""
    v = View(game_state, team_id, player_id, adaptive=adaptive, force=force)
    if v.me is None:
        return v.c("CLEAR_OVERRIDE")
    if v.pid == 0:
        return _gk(v)
    if v.i_have_ball:
        return _on_ball(v)
    if v.holder_mine:
        return _support(v)
    return _defend(v)


def allowed(game_state, team_id, player_id, cmd):
    """The press/shoot policy applied to any command (model or rules)."""
    v = View(game_state, team_id, player_id)
    return v.me is not None and v.allows(cmd)


def may_press(game_state, team_id, player_id):
    return allowed(game_state, team_id, player_id, {"commandType": "PRESS_BALL"})


def facts(game_state, team_id, player_id):
    """Pre-computed facts for the LLM prompt (code does the maths, the model judges)."""
    v = View(game_state, team_id, player_id)
    if v.me is None:
        return {}
    me = _pos(v.me)
    bp = _best_pass(v) if v.i_have_ball else None
    return {
        "role": v.role,
        "side": v.side,
        "hasBall": v.i_have_ball,
        "ballOwner": ("me" if v.i_have_ball else "team" if v.holder_mine
                      else "loose" if v.holder is None else "opponent"),
        "distToOppGoal": round(_d(me, v.opp_goal), 1),
        "distToBall": round(_d(me, v.ball), 1),
        "nearestOppDist": round(v.pressure(), 1),
        "iAmNearestToBall": v.nearest_mate_to(v.ball) == v.pid,
        "bestPassTarget": bp[0] if bp else None,
        "stamina": round(_stamina(v.me), 2),
        "attackDirection": "+x" if v.s > 0 else "-x",
        "style": v.style,                  # normal | counter | direct | patient (read from opponent)
        "oppKeeperOut": v.keeper_out,
        "mode": v.mode,                    # normal | chase (need a goal) | protect (keep the lead)
        # Positions so the model can judge where to move and whom to mark. Rounded to keep it short.
        # Positions on the LIVE pitch (what the model must answer in).
        # 2 decimals: 1 live x unit is ~8 m on the documented pitch, so whole numbers hide a lot.
        "me": [round(_real_x(me[0]), 2), round(_real_y(me[1]), 2)],
        "ball": [round(_real_x(v.ball[0]), 2), round(_real_y(v.ball[1]), 2)],
        "mates": {i: [round(_real_x(_pos(p)[0]), 2), round(_real_y(_pos(p)[1]), 2)] for i, p in sorted(v.mates.items()) if i != v.pid},
        "opps": {i: [round(_real_x(_pos(p)[0]), 2), round(_real_y(_pos(p)[1]), 2)] for i, p in sorted(v.opps.items())},
        "pitchHalfLengthX": PITCH_HALF_X,
        "pitchHalfWidthY": round(PITCH_HALF_Y, 2),
        "inOwnBox": _d(me, v.my_goal) < OWN_BOX,
        "score": game_state.get("score", {}),
        "gameTime": game_state.get("gameTime", 0),
    }


# --- between-match coach overrides ----------------------------------------------------------
# tools/coach.py may write kaklib/coach_config.json:
#   {"version": ..., "params": {"NAME" or "NAME.KEY": number, ...}, "prompt_rules": [str, ...]}
# Only existing numeric module constants (or keys of dict constants) are overridden, each within
# +-50% of its default. A missing or invalid file = today's behaviour.
COACH_VERSION = None
COACH_PROMPT_RULES = []
COACH_APPLIED = {}
# Only these may be overridden (= tools/coach.py BOUNDS). Never scaling, match length etc.
COACH_ALLOWED = {
    "SHOOT_RANGE.FWD", "SHOOT_RANGE.MID", "SHOOT_RANGE.DEF", "CHASE_SHOOT_BONUS",
    "KEEPER_OUT_SHOOT_BONUS", "OWN_BOX", "DEF_THIRD", "PRESSURE_DIST", "PRESSED_DIST",
    "GROUND_LANE_MIN", "GROUND_DIST_MAX", "SPRINT_MIN_STAMINA",
}


def _load_coach_config(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "coach_config.json")):
    global COACH_VERSION, COACH_PROMPT_RULES
    import json
    import logging
    log = logging.getLogger("tactics")
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except FileNotFoundError:
        return
    except Exception as e:  # bad JSON: keep defaults
        log.warning(f"coach_config ignored: {e}")
        return
    g = globals()
    for name, value in (cfg.get("params") or {}).items():
        if name not in COACH_ALLOWED or not isinstance(value, (int, float)) or isinstance(value, bool):
            log.warning(f"coach_config: skipped {name}={value!r} (not allowed or not a number)")
            continue
        base, _, key = str(name).partition(".")
        cur = g.get(base)
        if key:
            if not isinstance(cur, dict) or not isinstance(cur.get(key), (int, float)):
                continue
            default = cur[key]
        elif isinstance(cur, (int, float)) and not isinstance(cur, bool):
            default = cur
        else:
            continue
        lo, hi = sorted((default * 0.5, default * 1.5))
        v = max(lo, min(hi, float(value)))
        if key:
            cur[key] = v
        else:
            g[base] = v
        COACH_APPLIED[name] = v
    rules = cfg.get("prompt_rules") or []
    COACH_PROMPT_RULES = [str(r)[:200] for r in rules if isinstance(r, str)][:5]
    COACH_VERSION = cfg.get("version")


_load_coach_config()
