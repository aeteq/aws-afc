"""Deterministic role tactics. Used as the offline team in the simulator AND as the
fallback when the LLM is late or returns something invalid.

decide(game_state, team_id, player_id) -> [command]  (one command, tagged correctly)

Player ids: 0 GK, 1 DEF, 2 MID, 3 FWD1 (left), 4 FWD2 (right).
Pitch: x -55..55, y -35..35. HOME (team 0) attacks +x, AWAY (team 1) attacks -x.
"""

import math
import os
import re

from state import _is_my_team, _player_idx

HALF_X, HALF_Y = 55.0, 35.0
ROLE = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}

# Tunables. Distances are to the opponent goal centre.
SHOOT_RANGE = {"DEF": 18.0, "MID": 23.0, "FWD1": 25.0, "FWD2": 25.0}
# Shoot on sight. In the sim this broke the 0-0 mirror (10W 6D 0L vs the plain plan).
# Sim shot physics are guesses: if long shots just give the ball away live, set LONG_SHOTS=0.
LONG_SHOTS = os.getenv("LONG_SHOTS", "1") == "1"
LONG_SHOT_RANGE = {"DEF": 25.0, "MID": 38.0, "FWD1": 40.0, "FWD2": 40.0}
PRESSURE_DIST = 4.0      # an opponent this close = under pressure
SPRINT_MIN_STAMINA = 0.35

# Score/clock awareness. Set MATCH_SECONDS=120 if matches turn out to be 2 minutes.
MATCH_SECONDS = float(os.getenv("MATCH_SECONDS", "300"))
LATE_FRACTION = 0.2      # last 20% of the match counts as "late"
CHASE_SHOOT_BONUS = 8.0  # shoot from further when chasing a goal

# Adaptive style: read the opponent from each snapshot and switch.
# All 5 agents see the same snapshot and run this code, so they agree without talking.
KEEPER_OUT_ON, KEEPER_OUT_OFF = 8.0, 5.0   # their GK this far off his line = shoot from range
KEEPER_OUT_SHOOT_BONUS = 10.0
COMMITTED_ON, COMMITTED_OFF = 3, 1         # their outfielders in our half -> counter
PRESSED_DIST, PRESSED_COUNT = 8.0, 2       # this many opponents near our carrier -> release long
STYLES = ("auto", "normal", "counter", "direct", "patient")
# All 10 players (opponents too) hear every coach message, so a STYLE order only counts
# if the message also contains our tag, e.g. "kak1 STYLE counter". Empty tag = no check.
COACH_TAG = os.getenv("COACH_TAG", "kak1").lower()

# Last flags per (team, player). Lets thresholds be sticky (no flip-flopping).
# May not survive between calls on AgentCore; the code works the same without it.
_memory = {}


def _coach_style(chat, team_id):
    """Coach override: a teamChat message containing 'STYLE counter' (or auto/normal/direct/patient)."""
    my_codes = {str(team_id), "home" if team_id == 0 else "away"}
    for m in reversed(chat or []):
        if isinstance(m, dict):
            team = m.get("teamId", m.get("teamCode"))
            if team is not None and str(team).lower() not in my_codes:
                continue
            text = m.get("message") or m.get("text") or m.get("content") or ""
        else:
            text = str(m)
        if COACH_TAG and COACH_TAG not in str(text).lower():
            continue
        hit = re.search(r"\bSTYLE\s+(" + "|".join(STYLES) + r")\b", str(text), re.I)
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
        self.team_id, self.pid, self.role = team_id, pid, ROLE.get(pid, "MID")
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
        if self.keeper_out and self.role != "DEF":
            r += KEEPER_OUT_SHOOT_BONUS
        return min(r, 45.0)                # past this, shots are just giveaways

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


def _on_ball(v):
    me = _pos(v.me)
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
        kind = "THROUGH" if v.mates[bp[0]] and bp[0] in (3, 4) and bp[2] > 5 else "GROUND"
        return v.c("PASS", target_player_id=bp[0], type=kind)
    if pressured and v.role == "DEF":
        # Nobody open and under pressure deep: hoof it to the furthest forward.
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
    if v.role in ("FWD1", "FWD2"):
        side = 1.0 if v.role == "FWD1" else -1.0
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
        return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(-by * 0.5, 1), sprint=v.mode == "chase")
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
        if v.role in ("FWD1", "FWD2"):
            return _hold_high(v)
        if v.role == "MID" and _d(me, v.ball) < 15:
            return v.c("INTERCEPT", aggressive=True)
        return _shape(v)

    # Opponent has the ball.
    if chaser == v.pid:
        return v.c("PRESS_BALL", intensity=1.0)
    ball_in_our_half = (v.ball[0] - 0.0) * v.s < 0
    if v.role == "DEF":
        threat = _most_dangerous_free_opp(v)
        if threat is not None:
            return v.c("MARK", target_player_id=threat, tightness="TIGHT")
        return _shape(v)
    if v.role == "MID":
        # Chasing: press anywhere. Otherwise only in our half.
        if (ball_in_our_half or v.mode == "chase") and _d(me, v.ball) < 15:
            return v.c("PRESS_BALL", intensity=0.8)
        return _shape(v)
    # Forwards: second presser high up the pitch, otherwise wait for the counter.
    if v.mode == "protect":
        return _shape(v)                   # winning late: drop back and block passes
    if v.style == "counter" and v.mode != "chase":
        return _hold_high(v, depth=72)     # don't press: stay up as the outlet
    radius = 20.0 if v.mode == "chase" else 12.0
    if not ball_in_our_half and _d(me, v.ball) < radius:
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
    return v.c("MOVE_TO", target_x=round(tx, 1), target_y=round(ty, 1), sprint=False)


def _hold_high(v, depth=60):
    """Forwards stay near halfway (or higher), split wide, ready for a counter."""
    y = (8.0 if v.role == "FWD1" else -8.0) * v.s
    if depth > 60:
        y *= 1.5                           # wider when staying high, harder to mark both
    return v.c("MOVE_TO", target_x=round(v.fwd(depth), 1), target_y=y, sprint=False)


# --- entry -------------------------------------------------------------------------------

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


def facts(game_state, team_id, player_id):
    """Pre-computed facts for the LLM prompt (code does the maths, the model judges)."""
    v = View(game_state, team_id, player_id)
    if v.me is None:
        return {}
    me = _pos(v.me)
    bp = _best_pass(v) if v.i_have_ball else None
    return {
        "role": v.role,
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
        "score": game_state.get("score", {}),
        "gameTime": game_state.get("gameTime", 0),
    }
