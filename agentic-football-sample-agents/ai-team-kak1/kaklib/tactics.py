"""Deterministic role tactics, written from our coach's instructions (the same rules the
model reads in fast_agent.py). Used as the offline team in the simulator, as the SUGGESTION
the model sees, and as the fallback when the model is late or invalid.

decide(game_state, team_id, player_id) -> [command]  (one command, tagged correctly)

Formation 1-2-1. Player ids: 0 GK, 1 DEF, 2 MID1 (left), 3 MID2 (right), 4 FWD (single forward).
Roles come from the player id the server sends, not the agent folder name (ai-fwd1 = MID2).
Pitch: x -55..55, y -35..35. HOME (team 0) attacks +x, AWAY (team 1) attacks -x.
"Depth" below = metres from OUR goal line toward theirs (0 = our goal, 110 = theirs), so the
same rule works for both teams.
"""

import math
import os

from state import _is_my_team, _player_idx

HALF_X, HALF_Y = 55.0, 35.0
FIELD = 2 * HALF_X
HALFWAY = HALF_X                 # depth of the halfway line
THIRD = FIELD / 3                # middle third = depth 36.7 .. 73.3
ROLE = {0: "GK", 1: "DEF", 2: "MID1", 3: "MID2", 4: "FWD"}
MIDS = (2, 3)                    # MID1 takes the left channel, MID2 the right
FWD_ID = 4
CHANNEL_Y = 12.0                 # how wide each midfielder sits in his channel

# Zones. The server documents no penalty area; this is a real one scaled to the pitch.
BOX_DEPTH, BOX_HALF_W = 16.5, 20.0
ZONE_MARGIN = 1.0                # stay this far inside a hard boundary

# Shooting: clear shots only (close, centred, no defender blocking).
CLEAR_SHOT_DIST = 24.0           # to the opponent goal centre
CLEAR_SHOT_MAX_Y = 14.0          # "centred"
SHOT_LANE_MIN = 2.0              # no opponent outfielder this close to the shot line
CHASE_SHOOT_BONUS = 6.0          # losing late: shoot from a little further

PRESSURE_DIST = 4.0              # an opponent this close to me = under pressure
MARKED_DIST = 5.0                # an opponent this close to a teammate = marked
SAFE_LANE = 2.5                  # pass lane clearance for a "safe" pass
TACKLE_REACH = 3.5               # DEF tackles only within this distance
GOAL_SIDE_GAP = 3.0              # DEF stands this far from his man, toward our goal
INSIDE_SHADE = 1.5               # ...and this far to his inside, to show him wide
THROUGH_LEAD = 5.0               # "space ahead" of a forward for a through ball
SPRINT_MIN_STAMINA = 0.35

# Score/clock awareness. Set MATCH_SECONDS=120 if matches turn out to be 2 minutes.
MATCH_SECONDS = float(os.getenv("MATCH_SECONDS", "300"))
LATE_FRACTION = 0.2              # last 20% of the match counts as "late"

# All 10 players (opponents too) hear every coach message, so a message only reaches our
# model if it contains our tag, e.g. "kak1 forwards push higher". Empty tag = no check.
COACH_TAG = os.getenv("COACH_TAG", "kak1").lower()


def coach_messages(chat, team_id):
    """Our coach's messages, oldest first. A message only counts if it carries COACH_TAG
    (and, when the payload says, our team)."""
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


def _toward(a, b, dist):
    """The point `dist` metres from a toward b (b itself if closer)."""
    L = _d(a, b)
    if L <= dist or L == 0:
        return b
    return a[0] + (b[0] - a[0]) * dist / L, a[1] + (b[1] - a[1]) * dist / L


def _stamina(p):
    """Server docs say 0-100, the sample lib says 0-1. Accept both."""
    s = p.get("stamina", 100.0) if p else 100.0
    return s / 100.0 if s > 1.0 else s


def _lane_clear(a, b, opps):
    """Smallest distance from any opponent to the segment a->b (bigger = safer)."""
    ax, ay = a
    bx, by = b
    L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1e-9
    best = 99.0
    for o in opps:
        ox, oy = o
        t = _clamp(((ox - ax) * (bx - ax) + (oy - ay) * (by - ay)) / L2, 0.0, 1.0)
        best = min(best, math.hypot(ox - (ax + t * (bx - ax)), oy - (ay + t * (by - ay))))
    return best


# --- zones -------------------------------------------------------------------------------

def zone(role):
    """(min_depth, max_depth, max_abs_y) for roles with a HARD zone rule; None = no hard limit."""
    if role == "GK":
        return 0.0, BOX_DEPTH - ZONE_MARGIN, BOX_HALF_W - ZONE_MARGIN
    if role == "DEF":
        return 0.0, HALFWAY - ZONE_MARGIN, HALF_Y
    if role == "FWD":
        return HALFWAY + ZONE_MARGIN, FIELD, HALF_Y
    return None


class View:
    """The game state from one player's point of view."""

    def __init__(self, gs, team_id, pid):
        self.team_id, self.pid, self.role = team_id, pid, ROLE.get(pid, "MID1")
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
        self.mode = self._mode(gs)

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

    # --- geometry ---
    def depth(self, pt):
        return (pt[0] - self.my_goal[0]) * self.s

    def fwd(self, depth):
        return self.my_goal[0] + self.s * depth

    def in_zone(self, pt):
        z = zone(self.role)
        if z is None:
            return True
        return z[0] <= self.depth(pt) <= z[1] and abs(pt[1]) <= z[2]

    def clamp_zone(self, pt):
        z = zone(self.role) or (0.0, FIELD, HALF_Y)
        return self.fwd(_clamp(self.depth(pt), z[0], z[1])), _clamp(pt[1], -z[2], z[2])

    # --- commands ---
    def c(self, kind, **params):
        return [cmd(kind, self.team_id, self.pid, **params)]

    def move(self, pt, sprint=False):
        """MOVE_TO, always clamped to my zone (the hard rules can't be broken by rounding)."""
        x, y = self.clamp_zone(pt)
        return self.c("MOVE_TO", target_x=round(x, 1), target_y=round(y, 1), sprint=bool(sprint))

    def pass_to(self, i, kind="GROUND"):
        return self.c("PASS", target_player_id=i, type=kind)

    # --- reading the game ---
    def opp_positions(self):
        return [_pos(p) for p in self.opps.values()]

    def opp_outfield(self):
        return {i: p for i, p in self.opps.items() if i != 0}

    def pressure(self):
        me = _pos(self.me)
        return min((_d(me, o) for o in self.opp_positions()), default=99.0)

    def pressured(self):
        return self.pressure() < PRESSURE_DIST

    def marked(self, i):
        tp = _pos(self.mates[i])
        return min((_d(tp, o) for o in self.opp_positions()), default=99.0) < MARKED_DIST

    def lane_to(self, pt):
        return _lane_clear(_pos(self.me), pt, self.opp_positions())

    def safe_pass(self, i):
        """Teammate i is unmarked and the lane to his feet is clear."""
        return i in self.mates and i != self.pid and not self.marked(i) \
            and self.lane_to(_pos(self.mates[i])) >= SAFE_LANE

    def space_ahead(self, i):
        tx, ty = _pos(self.mates[i])
        return tx + self.s * THROUGH_LEAD, ty

    def safe_through(self, i):
        """The space ahead of teammate i is free and the lane to it is clear."""
        if i not in self.mates or i == self.pid:
            return False
        sp = self.space_ahead(i)
        free = min((_d(sp, o) for o in self.opp_positions()), default=99.0) >= MARKED_DIST - 1
        return free and self.lane_to(sp) >= SAFE_LANE

    def channel_y(self):
        """Midfielders: centre of my channel (MID1 left, MID2 right, from our attacking view)."""
        return (CHANNEL_Y if self.pid == MIDS[0] else -CHANNEL_Y) * self.s

    def other_mid(self):
        o = MIDS[1] if self.pid == MIDS[0] else MIDS[0]
        return o if o in self.mates else None

    def nearest_mate_to(self, pt):
        pool = [(i, p) for i, p in self.mates.items() if i != 0]
        return min(pool, key=lambda ip: _d(_pos(ip[1]), pt))[0] if pool else None

    def sprint(self):
        floor = 0.15 if self.mode == "chase" else SPRINT_MIN_STAMINA
        return _stamina(self.me) > floor

    def clear_shot(self):
        """Close, centred, and no opponent outfielder on the line to goal (keeper ignored)."""
        me = _pos(self.me)
        rng = CLEAR_SHOT_DIST + (CHASE_SHOOT_BONUS if self.mode == "chase" else 0.0)
        if _d(me, self.opp_goal) > rng or abs(me[1]) > CLEAR_SHOT_MAX_Y:
            return False
        blockers = [_pos(p) for p in self.opp_outfield().values()]
        return _lane_clear(me, self.opp_goal, blockers) >= SHOT_LANE_MIN

    # --- the single press/shoot policy: rules and model are both held to it ---
    def press_allowed(self):
        """Press only when the opponent has the ball. Never the GK, never the DEF in their half.
        At most the 2 nearest eligible outfielders (3 when chasing): avoids the 'ball magnet'
        (practice match 1: 152 model presses out of ~390 outfield commands)."""
        if self.pid == 0 or self.holder is None or self.holder_mine:
            return False
        their_half = self.depth(self.ball) > HALFWAY

        def eligible(i):
            return i != 0 and not (ROLE.get(i) == "DEF" and their_half)
        if not eligible(self.pid):
            return False
        order = sorted((i for i in self.mates if eligible(i)), key=lambda i: _d(_pos(self.mates[i]), self.ball))
        rank = order.index(self.pid) if self.pid in order else 99
        return rank < (3 if self.mode == "chase" else 2)

    def shot_allowed(self):
        """Same shot rule as the coach's: clear shots only."""
        return self.i_have_ball and self.clear_shot()

    def allows(self, command):
        kind = command.get("commandType")
        if kind == "PRESS_BALL":
            return self.press_allowed()
        if kind == "SHOOT":
            return self.shot_allowed()
        return True

    def threats(self):
        """Opponent outfielders, most dangerous (closest to our goal) first."""
        return sorted(self.opp_outfield(), key=lambda i: _d(_pos(self.opps[i]), self.my_goal))


def _goal_side(v, me, opp):
    """I'm between this opponent and the centre of our goal."""
    return _d(me, v.my_goal) < _d(opp, v.my_goal) and _lane_clear(opp, v.my_goal, [me]) < 3.0


def _shot(v):
    """Aim low at the corner away from the opponent keeper. L/R is from the shooter's view:
    attacking +x, 'L' is +y."""
    gk = v.opps.get(0)
    gk_y = _pos(gk)[1] if gk else 0.0
    want_y = -1.0 if gk_y > 0 else 1.0          # far side from the keeper
    lateral = want_y * v.s
    return v.c("SHOOT", aim_location="BL" if lateral > 0 else "BR", power=1.0)


def _chase_loose(v):
    """Loose ball: only the nearest outfielder goes, and only if it is in his zone."""
    if v.nearest_mate_to(v.ball) == v.pid and v.in_zone(v.ball):
        tgt = (v.ball[0] + v.ball_v[0] * 0.5, v.ball[1] + v.ball_v[1] * 0.5)
        return v.move(tgt, sprint=v.sprint())
    return None


# --- GK: our penalty area, never leaves it -----------------------------------------------

def _gk(v):
    me = _pos(v.me)
    if v.i_have_ball:
        return _gk_distribute(v)
    bd = _d(v.ball, v.my_goal)
    # Loose ball inside our area and we get there first: claim it (still inside the area).
    if v.holder is None and v.in_zone(v.ball):
        if _d(me, v.ball) < min((_d(o, v.ball) for o in v.opp_positions()), default=99.0):
            return v.move(v.ball, sprint=True)
    # Opponent clear through on goal: step toward him to narrow the angle, then hold.
    if _through_on_goal(v):
        return v.move(_toward(v.my_goal, v.ball, min(6.0, bd / 2)), sprint=True)
    # Default: on the line between the ball and the centre of our goal.
    return v.move(_toward(v.my_goal, v.ball, _clamp(bd * 0.08, 1.0, 3.0)))


def _through_on_goal(v):
    """An opponent carries the ball in our half and none of our outfielders is goal-side."""
    if v.holder is None or v.holder_mine or v.depth(v.ball) > HALFWAY:
        return False
    hp = _pos(v.holder)
    return not any(_goal_side(v, _pos(p), hp) for i, p in v.mates.items() if i != 0)


def _gk_distribute(v):
    """Pass immediately to the nearest unmarked teammate, preferring the defender."""
    me = _pos(v.me)
    cands = [i for i in v.mates if i != 0]
    if not cands:
        return v.c("CLEAR_OVERRIDE")
    free = [i for i in cands if v.safe_pass(i)]
    if 1 in free:
        target = 1
    elif free:
        target = min(free, key=lambda i: _d(me, _pos(v.mates[i])))
    else:   # everyone marked: the most open one
        target = max(cands, key=lambda i: min((_d(_pos(v.mates[i]), o) for o in v.opp_positions()),
                                              default=20.0) + v.lane_to(_pos(v.mates[i])))
    method = "THROW" if _d(me, _pos(v.mates[target])) < 25 else "KICK"
    return v.c("GK_DISTRIBUTE", target_player_id=target, method=method)


# --- DEF: our half, never crosses halfway ------------------------------------------------

def _def(v):
    if v.i_have_ball:
        return _def_on_ball(v)
    if v.holder_mine:
        # Hold a safe line behind the ball, inside our half.
        d = _clamp(v.depth(v.ball) - 25.0, 15.0, HALFWAY - 5.0)
        return v.move((v.fwd(d), v.ball[1] * 0.3))
    if v.holder is None:
        return _chase_loose(v) or _def_guard(v)
    return _def_guard(v)


def _def_guard(v):
    me = _pos(v.me)
    # Tackle only when the carrier is within reach AND we are between him and our goal.
    if v.holder is not None and not v.holder_mine:
        hp = _pos(v.holder)
        if _d(me, hp) <= TACKLE_REACH and _goal_side(v, me, hp) and v.in_zone(hp) and v.press_allowed():
            return v.c("PRESS_BALL", intensity=0.9)
    threats = v.threats()
    if not threats:
        return v.move(_toward(v.my_goal, v.ball, 20.0))
    # Goal-side of the most dangerous opponent, on his inside shoulder: show him wide.
    t = v.opps[threats[0]]
    tp = _pos(t)
    return v.move(_show_wide_spot(v, tp, t.get("velocity") or {}),
                  sprint=_d(me, tp) > 10 and v.sprint())


def _show_wide_spot(v, tp, vel):
    """Where to stand to push an attacker toward the touchline: GOAL_SIDE_GAP toward our goal
    centre, shifted INSIDE_SHADE toward the middle of the pitch. That leaves him the outside
    as his open route, where the angle to goal keeps shrinking. If he's dead centre, push him
    to the side he's drifting toward (or the side he's on)."""
    gx, gy = _toward(tp, v.my_goal, GOAL_SIDE_GAP)
    if abs(tp[1]) >= 1.0:
        out = 1.0 if tp[1] > 0 else -1.0             # his outside: away from the centre
    else:
        vy = vel.get("y", 0.0) or 0.0
        out = (1.0 if vy > 0 else -1.0) if abs(vy) > 0.3 else (1.0 if tp[1] >= 0 else -1.0)
    return gx, gy - out * INSIDE_SHADE


def _def_on_ball(v):
    """Forward to a playmaker (the unmarked midfielder, MID1 first); if both are marked, back
    to the keeper. Never hold it."""
    for i in MIDS:
        if v.safe_pass(i):
            return v.pass_to(i)
    if v.safe_pass(0):
        return v.pass_to(0)
    # Nobody safe: clear it long to the most advanced teammate rather than hold it.
    far = max((i for i in v.mates if i not in (v.pid, 0)),
              key=lambda i: v.depth(_pos(v.mates[i])), default=None)
    return v.pass_to(far, "AERIAL") if far is not None else v.pass_to(0)


# --- MID1 / MID2: middle third, free to drift wide, one channel each ---------------------

def _mid(v):
    if v.i_have_ball:
        return _mid_on_ball(v)
    if v.holder_mine:
        # Offer an option in the middle third, in my channel, drifting a little with the ball.
        d = _clamp(v.depth(v.ball) - 6.0, THIRD, 2 * THIRD)
        if v.holder is not None and _player_idx(v.holder) == v.other_mid():
            d = _clamp(v.depth(v.ball) + 8.0, THIRD, 2 * THIRD)   # partner has it: get ahead
        return v.move((v.fwd(d), v.channel_y() + v.ball[1] * 0.3))
    if v.holder is None:
        return _chase_loose(v) or _mid_cover(v)
    return _mid_defend(v)


def _mid_on_ball(v):
    if v.clear_shot():
        return _shot(v)
    # Striker first (our single forward), winger second (the other midfielder, if ahead of me).
    me_depth = v.depth(_pos(v.me))
    order = [FWD_ID]
    o = v.other_mid()
    if o is not None and v.depth(_pos(v.mates[o])) > me_depth + 3:
        order.append(o)
    for i in order:
        if v.safe_through(i):
            return v.pass_to(i, "THROUGH")       # into space ahead of him, not to his feet
    if v.safe_pass(1):
        return v.pass_to(1)                      # recycle to the defender
    if not v.pressured():
        # Nothing on and nobody near me: carry it forward through the middle.
        me = _pos(v.me)
        return v.move((v.fwd(_clamp(v.depth(me) + 10.0, 0.0, FIELD - 18.0)), me[1] * 0.7),
                      sprint=v.sprint())
    # Under pressure and nothing safe: the most open teammate, never a risky dribble.
    i = max((i for i in v.mates if i != v.pid), key=lambda i: v.lane_to(_pos(v.mates[i])))
    return v.pass_to(i)


def _mid_defend(v):
    if _our_half_outnumbered(v):
        return _mid_cover(v)
    if THIRD <= v.depth(v.ball) <= 2 * THIRD and _i_am_pressing_mid(v) and v.press_allowed():
        return v.c("PRESS_BALL", intensity=0.8)  # their carrier is in the middle third
    return _mid_cover(v)


def _i_am_pressing_mid(v):
    """Only one midfielder presses: the one nearer the ball. The other covers."""
    o = v.other_mid()
    if o is None:
        return True
    mine, theirs = _d(_pos(v.me), v.ball), _d(_pos(v.mates[o]), v.ball)
    return mine < theirs or (mine == theirs and v.pid == MIDS[0])


def _our_half_outnumbered(v):
    theirs = sum(1 for p in v.opp_outfield().values() if v.depth(_pos(p)) < HALFWAY)
    ours = sum(1 for i, p in v.mates.items() if i != 0 and v.depth(_pos(p)) < HALFWAY)
    return theirs > ours


def _mid_cover(v):
    """Cover the defender: pick up an opponent in our half the DEF isn't on (the two
    midfielders split them, nearest first), otherwise hold my channel between ball and goal."""
    in_our_half = [i for i in v.threats() if v.depth(_pos(v.opps[i])) < HALFWAY][1:]  # DEF has [0]
    o = v.other_mid()
    if in_our_half:
        me = _pos(v.me)
        if o is None or len(in_our_half) == 1:
            # One spare attacker: the nearer midfielder takes him, the other holds shape.
            t = in_our_half[0]
            tp = _pos(v.opps[t])
            if o is None or _d(me, tp) <= _d(_pos(v.mates[o]), tp):
                return _mark_wide(v, t)
        else:
            a, b = in_our_half[0], in_our_half[1]
            pa, pb = _pos(v.opps[a]), _pos(v.opps[b])
            op = _pos(v.mates[o])
            # Pick the split with the shorter total run.
            mine_a = _d(me, pa) + _d(op, pb) <= _d(me, pb) + _d(op, pa)
            return _mark_wide(v, a if mine_a else b)
    d = _clamp(v.depth(v.ball) * 0.6, 20.0, 2 * THIRD)
    return v.move((v.fwd(d), v.channel_y() * 0.7 + v.ball[1] * 0.3))


def _mark_wide(v, opp_id):
    """Midfielder covering a spare attacker: goal-side on his inside, like the DEF."""
    t = v.opps[opp_id]
    return v.move(_show_wide_spot(v, _pos(t), t.get("velocity") or {}), sprint=v.sprint())


# --- FWD: opponent half, never drops behind halfway ---------------------------------------

def _fwd(v):
    if v.i_have_ball:
        return _fwd_on_ball(v)
    if v.holder is None:
        got = _chase_loose(v)
        if got:
            return got
    return _fwd_position(v)


def _fwd_on_ball(v):
    if v.clear_shot():
        return _shot(v)
    me = _pos(v.me)
    blocked = v.pressured() or _lane_clear(me, v.opp_goal, [_pos(p) for p in v.opp_outfield().values()]) < 3.0
    if blocked:
        # Lay off to a midfielder, whichever is safer.
        opts = [i for i in MIDS if v.safe_pass(i)]
        if opts:
            return v.pass_to(max(opts, key=lambda i: v.lane_to(_pos(v.mates[i]))))
    # Carry it toward a central shooting spot, away from the nearest defender.
    near = min(v.opp_positions(), key=lambda o: _d(o, me), default=(0.0, 0.0))
    ty = _clamp(me[1] + (3.0 if near[1] < me[1] else -3.0), -10.0, 10.0)
    return v.move((v.fwd(FIELD - 16.0), ty), sprint=v.sprint())


def _fwd_position(v):
    """Between their two deepest defenders, on their line, ready for a pass in behind."""
    outfield = sorted(v.opp_outfield().values(), key=lambda p: -v.depth(_pos(p)))[:2]
    if not outfield:
        return v.move((v.fwd(FIELD - 20.0), 0.0))
    line = sum(v.depth(_pos(p)) for p in outfield) / len(outfield)
    gap_y = sum(_pos(p)[1] for p in outfield) / len(outfield)
    if len(outfield) == 1:
        # One deep defender: stand off his shoulder, on the side nearer the centre.
        gap_y += -6.0 if gap_y > 0 else 6.0
    d = _clamp(line, HALFWAY + 5.0, FIELD - 10.0)
    ty = _clamp(gap_y, -HALF_Y + 3, HALF_Y - 3)
    return v.move((v.fwd(d), ty), sprint=v.holder_mine and v.sprint())


# --- entry -------------------------------------------------------------------------------

def decide(game_state, team_id, player_id):
    v = View(game_state, team_id, player_id)
    if v.me is None:
        return v.c("CLEAR_OVERRIDE")
    return {"GK": _gk, "DEF": _def, "FWD": _fwd}.get(v.role, _mid)(v)


def allowed(game_state, team_id, player_id, command):
    """The press/shoot policy applied to any command (model or rules)."""
    v = View(game_state, team_id, player_id)
    return v.me is not None and v.allows(command)


def may_press(game_state, team_id, player_id):
    return allowed(game_state, team_id, player_id, {"commandType": "PRESS_BALL"})


def respects_zone(game_state, team_id, player_id, command):
    """False if a command (e.g. from the model) would break the role's hard zone rule."""
    v = View(game_state, team_id, player_id)
    if v.me is None or zone(v.role) is None:
        return True
    kind, p = command.get("commandType"), command.get("parameters") or {}
    if kind == "MOVE_TO":
        return v.in_zone((p.get("target_x", 0.0), p.get("target_y", 0.0)))
    if kind in ("PRESS_BALL", "INTERCEPT"):
        return v.in_zone(v.ball)
    if kind == "MARK":
        target = v.opps.get(p.get("target_player_id"))
        return target is not None and v.in_zone(_pos(target))
    return True


def facts(game_state, team_id, player_id):
    """Pre-computed facts for the model prompt (code does the maths, the model judges)."""
    v = View(game_state, team_id, player_id)
    if v.me is None:
        return {}
    me = _pos(v.me)
    z = zone(v.role)
    return {
        "role": v.role,
        "hasBall": v.i_have_ball,
        "ballOwner": ("me" if v.i_have_ball else "team" if v.holder_mine
                      else "loose" if v.holder is None else "opponent"),
        "clearShot": v.i_have_ball and v.clear_shot(),
        "underPressure": v.pressured(),
        "iAmNearestToBall": v.nearest_mate_to(v.ball) == v.pid,
        "unmarkedMates": [i for i in sorted(v.mates) if v.safe_pass(i)] if v.i_have_ball else None,
        "zone": None if z is None else {
            "x": sorted([round(v.fwd(z[0]), 1), round(v.fwd(z[1]), 1)]), "maxAbsY": z[2]},
        "distToOppGoal": round(_d(me, v.opp_goal), 1),
        "distToBall": round(_d(me, v.ball), 1),
        "stamina": round(_stamina(v.me), 2),
        "attackDirection": "+x" if v.s > 0 else "-x",
        "mode": v.mode,                    # normal | chase (need a goal) | protect (keep the lead)
        "me": [round(me[0]), round(me[1])],
        "ball": [round(v.ball[0]), round(v.ball[1])],
        "mates": {i: [round(_pos(p)[0]), round(_pos(p)[1])] for i, p in sorted(v.mates.items()) if i != v.pid},
        "opps": {i: [round(_pos(p)[0]), round(_pos(p)[1])] for i, p in sorted(v.opps.items())},
        "score": game_state.get("score", {}),
        "gameTime": game_state.get("gameTime", 0),
    }
