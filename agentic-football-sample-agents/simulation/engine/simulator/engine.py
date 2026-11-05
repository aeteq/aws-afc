"""A small 2D match engine for playing Agentic Football teams against each other locally.

The real game server is not public, so this is a stand-in, built from the published rules:
5v5 on a 110 x 70 pitch, a 5-minute match, and every agent called every 2 s on the same
snapshot, in the server's payload format. The commands are the ones the sample teams use.

THE PHYSICS ARE A GUESS. Speeds, friction, tackle and save odds are plausible numbers,
not the server's. Use this to catch tactical mistakes and to compare two versions of a
team, never as proof of a result on match day.

Coordinates are world coordinates: HOME (team 0) attacks +x, AWAY (team 1) attacks -x.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

DT = 0.1
DECISION_INTERVAL = 2.0
MATCH_SECONDS = 300.0
HALF_X, HALF_Y = 55.0, 35.0
GOAL_HALF_WIDTH = 3.66
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.0

WALK, RUN, SPRINT = 2.5, 5.0, 7.5          # m/s at full stamina
DRIBBLE_FACTOR = 0.85
CONTROL_RADIUS = 1.0
TACKLE_RADIUS = 1.4
BODY = 0.8                                 # minimum distance between two players
GROUND_FRICTION = 1.0                      # per second
AIR_FRICTION = 0.2
KEEPER_MAX_HOLD = 6.0                      # seconds before an undirected keeper kicks long

# Kick-off spots for HOME; AWAY uses the same spots rotated 180 degrees.
FORMATION = {0: (-52.0, 0.0), 1: (-30.0, 0.0), 2: (-12.0, 0.0), 3: (-4.0, -10.0), 4: (-4.0, 10.0)}

# The platform keeps the first command per player per reply (the organisers' cars
# edition says so, and a CLEAR_OVERRIDE alongside a command cancels it), so a reply
# of [CLEAR_OVERRIDE, PASS] plays only the CLEAR_OVERRIDE.
FIRST_COMMAND_ONLY = True

# The workshop's game rules: the ball never goes out of play and there are no
# penalties; after a goal everyone resets and the team that conceded kicks off.
# False brings back throw-ins, corners and goal kicks.
BALL_STAYS_IN = True

MAINTAINED = {"PRESS_BALL", "MARK", "INTERCEPT", "FOLLOW_PLAYER", "SLIDE_TACKLE"}
ON_BALL = {"PASS", "SHOOT", "GK_DISTRIBUTE"}


def _num(v, default=0.0) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    v = float(v)
    return v if math.isfinite(v) else default


def _side(team: int) -> float:
    """+1 for HOME, -1 for AWAY: the direction a team attacks, and its rotation."""
    return 1.0 if team == 0 else -1.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class Player:
    team: int
    idx: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    stamina: float = 1.0
    order: dict | None = None
    cooldown: float = 0.0      # seconds before this player may touch the ball again

    @property
    def agent_id(self) -> str:
        return f"agentId_{self.idx}"


@dataclass
class Ball:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    owner: Player | None = None
    air: float = 0.0           # seconds of flight left; nobody can touch it until 0
    shot_by: int | None = None
    shot_low: bool = True
    save_tried: bool = False
    shot_from: tuple = (0.0, 0.0)
    shooter: int | None = None
    in_hands: bool = False     # a keeper holding it in his own box: cannot be tackled
    protect: float = 0.0       # seconds the owner cannot be tackled
    hold: float = 0.0          # seconds the current owner has held it
    last_team: int = 0
    pass_from: Player | None = None
    pass_to: Player | None = None  # the intended receiver of the pass in flight


@dataclass
class TeamStats:
    goals: int = 0
    shots: int = 0
    on_target: int = 0
    saves_made: int = 0
    passes: int = 0
    passes_completed: int = 0
    tackles_won: int = 0
    possession: float = 0.0
    wasted_commands: int = 0   # PASS/SHOOT/GK_DISTRIBUTE sent by a player without the ball
    rejected_commands: int = 0  # wrong playerId/teamId, or not a command at all
    ignored_commands: int = 0   # a second command for the same player in one reply


@dataclass
class MatchResult:
    home: str
    away: str
    score: tuple
    stats: tuple
    events: list = field(default_factory=list)
    frames: list | None = None


class Match:
    """One match. Controllers expose commands(game_state, team_id, player_id) -> list[dict]."""

    def __init__(self, home, away, seed: int = 0, record: bool = False,
                 seconds: float = MATCH_SECONDS, realtime: bool = False):
        self.rng = random.Random(seed)
        self.controllers = (home, away)
        self.seconds = seconds
        # As live: both teams asked at the same moment, one turn per DECISION_INTERVAL of
        # wall clock (a slow turn is not caught up). Off, the match runs as fast as it can.
        self.realtime = realtime
        self.players = [Player(t, i, *self._spot(t, i)) for t in (0, 1) for i in range(5)]
        self.ball = Ball()
        self.time = 0.0
        self.tick = 0
        self.stats = (TeamStats(), TeamStats())
        self.events: list = []
        self.frames: list | None = [] if record else None
        self._kickoff(self.rng.randrange(2))  # coin toss: first possession is a real edge

    # ------------------------------------------------------------------ loop

    def run(self) -> MatchResult:
        next_decision = 0.0
        turn_started = None
        while self.time < self.seconds - 1e-9:
            if self.time >= next_decision - 1e-9:
                if self.realtime:
                    if turn_started is not None:
                        time.sleep(max(0.0, DECISION_INTERVAL - (time.monotonic() - turn_started)))
                    turn_started = time.monotonic()
                self._decide()
                next_decision += DECISION_INTERVAL
            self._step()
            self.time += DT
            self.tick += 1
            if self.frames is not None and self.tick % 2 == 0:
                self._record()
        return MatchResult(getattr(self.controllers[0], "name", "home"),
                           getattr(self.controllers[1], "name", "away"),
                           (self.stats[0].goals, self.stats[1].goals), self.stats,
                           self.events, self.frames)

    def _decide(self) -> None:
        state = self.payload()

        def ask(team):
            asked = []
            for pid in range(5):
                try:
                    cmds = self.controllers[team].commands(state, team, pid)
                except Exception as e:  # a crashing agent stands still, as live
                    self.events.append((round(self.time, 1), "crash", team, pid, str(e)[:80]))
                    cmds = []
                asked.append((team, pid, cmds))
            return asked

        if self.realtime:
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = [a for team in pool.map(ask, (0, 1)) for a in team]
        else:
            pending = ask(0) + ask(1)
        # Both teams decided on the same snapshot; apply in a random team order so
        # neither side systematically acts first.
        if self.rng.random() < 0.5:
            pending.reverse()
        for team, pid, cmds in pending:
            self._apply(team, pid, cmds)

    # ------------------------------------------------------------- payload

    def payload(self) -> dict:
        """The game state in the server's format (home listed first, ids per team)."""
        b = self.ball
        ball = {"position": {"x": round(b.x, 2), "y": round(b.y, 2), "z": 0.0},
                "velocity": {"x": round(b.vx, 2), "y": round(b.vy, 2), "z": 0.0},
                "isFree": b.owner is None}
        if b.owner is not None:
            ball["possessionAgentId"] = b.owner.agent_id
        players = [{"agentId": p.agent_id, "teamCode": "home" if p.team == 0 else "away",
                    "position": {"x": round(p.x, 2), "y": round(p.y, 2)},
                    "velocity": {"x": round(p.vx, 2), "y": round(p.vy, 2)},
                    "stamina": round(100.0 * p.stamina, 1),  # 0-100, as the server sends it
                    "isSprinting": math.hypot(p.vx, p.vy) > RUN + 0.1}
                   for p in self.players]
        return {"tick": self.tick, "gameTime": round(self.time, 1), "playMode": "OPEN_PLAY",
                "score": {"home": self.stats[0].goals, "away": self.stats[1].goals},
                "ball": ball, "players": players, "teamChat": []}

    # ------------------------------------------------------------ commands

    def _player(self, team: int, idx) -> Player | None:
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < 5:
            return None
        return self.players[team * 5 + idx]

    def _apply(self, team: int, pid: int, cmds) -> None:
        p = self.players[team * 5 + pid]
        st = self.stats[team]
        if not isinstance(cmds, list):
            st.rejected_commands += 1
            return
        applied = False
        for c in cmds:
            if not isinstance(c, dict) or c.get("playerId") != pid or c.get("teamId") != team:
                st.rejected_commands += 1
                continue
            if applied and FIRST_COMMAND_ONLY:
                st.ignored_commands += 1
                continue
            applied = True
            kind = c.get("commandType")
            params = c.get("parameters") if isinstance(c.get("parameters"), dict) else {}
            if kind in ("CLEAR_OVERRIDE", "RESET", "SET_STANCE"):
                p.order = None
            elif kind == "MOVE_TO":
                p.order = {"type": kind, "x": _clamp(_num(params.get("target_x"), p.x), -HALF_X, HALF_X),
                           "y": _clamp(_num(params.get("target_y"), p.y), -HALF_Y, HALF_Y),
                           "sprint": params.get("sprint") is True}
            elif kind in MAINTAINED:
                p.order = {"type": kind, **params}
            elif kind in ON_BALL:
                if self.ball.owner is not p or (kind == "GK_DISTRIBUTE" and p.idx != 0):
                    st.wasted_commands += 1
                elif kind == "SHOOT":
                    self._shoot(p, params)
                else:
                    self._pass(p, params, keeper=(kind == "GK_DISTRIBUTE"))
            else:
                st.rejected_commands += 1

    def _release(self, p: Player, tx: float, ty: float, speed: float, sigma_deg: float,
                 air: float = 0.0) -> None:
        b = self.ball
        angle = math.atan2(ty - p.y, tx - p.x) + math.radians(self.rng.gauss(0.0, sigma_deg))
        b.vx, b.vy = speed * math.cos(angle), speed * math.sin(angle)
        b.x, b.y = p.x, p.y
        b.owner, b.air, b.hold, b.in_hands = None, air, 0.0, False
        b.last_team = p.team
        p.cooldown = 0.4

    def _pressure(self, p: Player) -> float:
        return min((math.hypot(o.x - p.x, o.y - p.y) for o in self.players if o.team != p.team),
                   default=99.0)

    def _pass(self, p: Player, params: dict, keeper: bool) -> None:
        t = self._player(p.team, params.get("target_player_id"))
        if t is None or t is p:
            self.stats[p.team].wasted_commands += 1
            return
        kind = params.get("type", "GROUND")
        if keeper:
            kind = "GROUND" if params.get("method") == "THROW" else "AERIAL"
        d = math.hypot(t.x - p.x, t.y - p.y)
        aerial = kind == "AERIAL"
        if aerial:
            speed = _clamp(8.0 + 0.5 * d, 12.0, 25.0)
            flight = d / speed
        else:
            # Kicked to arrive at ~5 m/s. The old 6 + 0.7 d ignored friction: the
            # ball could roll only speed / friction metres in total, so a 20 m pass
            # died at the receiver's feet and was cut out there.
            speed = _clamp(5.0 + GROUND_FRICTION * d, 8.0, 28.0)
            reach = min(GROUND_FRICTION * d / speed, 0.95)
            flight = -math.log(1.0 - reach) / GROUND_FRICTION
        tx, ty = t.x + t.vx * flight, t.y + t.vy * flight            # lead the receiver
        if kind == "THROUGH":
            tx += _side(p.team) * 6.0
        sigma = (2.0 + 0.08 * d + (2.0 if aerial else 0.0)) * (1.5 if self._pressure(p) < 2.0 else 1.0)
        self._release(p, tx, ty, speed, sigma, air=0.7 * flight if aerial else 0.0)
        self.ball.pass_from, self.ball.pass_to = p, t
        self.stats[p.team].passes += 1

    def _shoot(self, p: Player, params: dict) -> None:
        aim = params.get("aim_location", "CENTER")
        aim = aim if isinstance(aim, str) else "CENTER"
        s = _side(p.team)
        gx = s * HALF_X
        # L/R from the shooter's view: attacking +x, the left post is at +y.
        lateral = 1.0 if aim.endswith("L") else -1.0 if aim.endswith("R") else 0.0
        gy = s * lateral * (GOAL_HALF_WIDTH - 0.6)
        d = math.hypot(gx - p.x, gy - p.y)
        power = _clamp(_num(params.get("power"), 0.8), 0.0, 1.0)
        high = aim.startswith("T")
        sigma = (2.5 + 0.2 * d + (1.0 if high else 0.0)) * (1.4 if self._pressure(p) < 2.0 else 1.0)
        self._release(p, gx, gy, 12.0 + 18.0 * power, sigma)
        b = self.ball
        b.shot_by, b.shot_low, b.save_tried, b.pass_from, b.pass_to = p.team, not high, False, None, None
        b.shot_from, b.shooter = (p.x, p.y), p.idx
        st = self.stats[p.team]
        st.shots += 1
        if b.vx * s > 0:
            y_at_line = b.y + b.vy * (gx - b.x) / b.vx
            if abs(y_at_line) < GOAL_HALF_WIDTH:
                st.on_target += 1
        self.events.append((round(self.time, 1), "shot", p.team, p.idx, round(d, 1)))

    # ------------------------------------------------------------- physics

    def _step(self) -> None:
        # Every target is computed from the same snapshot, then everyone moves.
        # Moving one by one, home first, let away players react to where home
        # players were this step while home reacted to last step: a one-step
        # edge for away in every chase, mark and press.
        targets = [self._target(p) for p in self.players]
        for p, target in zip(self.players, targets):
            self._move(p, target)
        self._separate()
        self._move_ball()
        if self.ball.owner is not None:
            self.stats[self.ball.owner.team].possession += DT

    def _move(self, p: Player, target: tuple) -> None:
        tx, ty, pace = target
        speed = {"stand": 0.0, "walk": WALK, "run": RUN, "sprint": SPRINT}[pace]
        speed *= 0.6 + 0.4 * p.stamina
        if self.ball.owner is p:
            speed *= DRIBBLE_FACTOR
        dx, dy = tx - p.x, ty - p.y
        d = math.hypot(dx, dy)
        step = min(speed * DT, d)
        if d > 1e-6 and step > 0:
            p.vx, p.vy = dx / d * step / DT, dy / d * step / DT
            p.x = _clamp(p.x + p.vx * DT, -HALF_X, HALF_X)
            p.y = _clamp(p.y + p.vy * DT, -HALF_Y, HALF_Y)
        else:
            p.vx = p.vy = 0.0
        moving = math.hypot(p.vx, p.vy)
        if moving > RUN * 0.6 + 0.4 and pace == "sprint":
            p.stamina -= 0.010 * DT
        elif moving > WALK + 0.1:
            p.stamina -= 0.0015 * DT
        else:
            p.stamina += 0.006 * DT
        p.stamina = _clamp(p.stamina, 0.0, 1.0)
        p.cooldown = max(0.0, p.cooldown - DT)

    def _separate(self) -> None:
        """Bodies collide: no two players closer than BODY metres (a presser cannot stand on the ball)."""
        ps = self.players
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if d >= BODY:
                    continue
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0
                push = (BODY - d) / 2
                # Clamped: a crowd must not shove a keeper (and the ball he holds)
                # over his own goal line.
                a.x = _clamp(a.x - dx / d * push, -HALF_X, HALF_X)
                a.y = _clamp(a.y - dy / d * push, -HALF_Y, HALF_Y)
                b.x = _clamp(b.x + dx / d * push, -HALF_X, HALF_X)
                b.y = _clamp(b.y + dy / d * push, -HALF_Y, HALF_Y)

    def _target(self, p: Player) -> tuple:
        """(x, y, pace) the player heads for this step, from its order or the default AI."""
        b = self.ball
        o = p.order
        s = _side(p.team)
        if o is not None:
            kind = o["type"]
            if kind == "MOVE_TO":
                arrived = math.hypot(o["x"] - p.x, o["y"] - p.y) < 0.5
                return o["x"], o["y"], "stand" if arrived else ("sprint" if o["sprint"] else "run")
            if kind in ("PRESS_BALL", "SLIDE_TACKLE"):
                if b.owner is not None and b.owner.team == p.team:
                    return p.x, p.y, "stand"  # nothing to press: our own man has it
                intensity = 1.0 if kind == "SLIDE_TACKLE" else _num(o.get("intensity"), 0.7)
                return b.x, b.y, "sprint" if intensity >= 0.6 else "run"
            if kind == "MARK":
                m = self._player(1 - p.team, o.get("target_player_id"))
                if m is not None:
                    gap = 1.5 if o.get("tightness") == "TIGHT" else 3.5
                    gx = -s * HALF_X
                    d = math.hypot(gx - m.x, m.y) or 1.0
                    x, y = m.x + (gx - m.x) / d * gap, m.y + (0 - m.y) / d * gap
                    return x, y, "sprint" if math.hypot(x - p.x, y - p.y) > 5 else "run"
            if kind == "INTERCEPT":
                if b.owner is not None:
                    tx, ty = b.owner.x, b.owner.y
                else:
                    tx, ty = b.x + b.vx * 0.6, b.y + b.vy * 0.6
                return tx, ty, "sprint" if o.get("aggressive") is True else "run"
            if kind == "FOLLOW_PLAYER":
                team = 1 if str(o.get("target_team", "")).upper() == "AWAY" else 0
                m = self._player(team, o.get("target_player_id"))
                if m is not None:
                    d = math.hypot(m.x - p.x, m.y - p.y) or 1.0
                    gap = _num(o.get("distance"), 2.0)
                    return m.x - (m.x - p.x) / d * gap, m.y - (m.y - p.y) / d * gap, "run"
        return self._default(p)

    def _default(self, p: Player) -> tuple:
        """What a player does with no order: the engine's stand-in for the server's own AI."""
        b = self.ball
        s = _side(p.team)
        if b.owner is p:
            if b.in_hands:  # a keeper with it in his hands holds it and distributes
                return p.x, p.y, "stand"
            return s * HALF_X, p.y * 0.8, "run"  # carry it toward goal
        if p.idx == 0:
            gx = -s * (HALF_X - 2.0)
            return gx, _clamp(b.y * 0.1, -3.0, 3.0), "walk"
        if b.owner is None:
            mates = [q for q in self.players if q.team == p.team and q.idx != 0]
            nearest = min(mates, key=lambda q: math.hypot(q.x - b.x, q.y - b.y))
            if nearest is p and math.hypot(p.x - b.x, p.y - b.y) < 15.0:
                return b.x, b.y, "run"
        fx, fy = FORMATION[p.idx]
        return _clamp(s * fx * 0.5 + b.x * 0.5, -50, 50), s * fy + 0.3 * b.y, "walk"

    def _move_ball(self) -> None:
        b = self.ball
        if b.owner is not None:
            p = b.owner
            sp = math.hypot(p.vx, p.vy)
            ux, uy = (p.vx / sp, p.vy / sp) if sp > 0.1 else (_side(p.team), 0.0)
            b.x, b.y, b.vx, b.vy = p.x + 0.5 * ux, p.y + 0.5 * uy, p.vx, p.vy
            b.protect = max(0.0, b.protect - DT)
            b.hold += DT
            if p.idx == 0 and b.hold > KEEPER_MAX_HOLD:
                self._keeper_hoof(p)
            else:
                self._tackles(p)
        else:
            b.x += b.vx * DT
            b.y += b.vy * DT
            k = AIR_FRICTION if b.air > 0 else GROUND_FRICTION
            f = math.exp(-k * DT)
            b.vx, b.vy = b.vx * f, b.vy * f
            b.air = max(0.0, b.air - DT)
            if b.shot_by is not None:
                self._shot_stoppers()
        if self._out_of_play():
            return
        if b.owner is None and b.air <= 0:
            self._pickup()

    def _keeper_hoof(self, gk: Player) -> None:
        mates = [q for q in self.players if q.team == gk.team and q is not gk]
        t = max(mates, key=lambda q: _side(gk.team) * q.x)
        self._pass(gk, {"target_player_id": t.idx}, keeper=True)

    def _tackles(self, carrier: Player) -> None:
        b = self.ball
        if b.protect > 0 or (b.in_hands and self._in_own_box(carrier)):
            return
        for o in self.players:
            if o.team == carrier.team or o.cooldown > 0:
                continue
            if math.hypot(o.x - carrier.x, o.y - carrier.y) > TACKLE_RADIUS:
                continue
            chance = 0.03
            if o.order and o.order["type"] == "PRESS_BALL":
                chance += 0.09 * _clamp(_num(o.order.get("intensity"), 0.7), 0.0, 1.0)
            elif o.order and o.order["type"] == "SLIDE_TACKLE":
                chance = 0.15
            elif o.order and o.order["type"] == "MARK" and o.order.get("target_player_id") == carrier.idx \
                    and o.order.get("tightness") == "TIGHT":
                chance += 0.04
            if self.rng.random() < chance:
                self.stats[o.team].tackles_won += 1
                carrier.cooldown = 0.5
                if self.rng.random() < 0.5:
                    self._gain(o)
                else:
                    # A tackler pokes the ball away from his own goal, not into it.
                    forward = 0.0 if o.team == 0 else math.pi
                    a = forward + math.radians(self.rng.uniform(-70.0, 70.0))
                    b.owner, b.vx, b.vy, b.last_team = None, 5 * math.cos(a), 5 * math.sin(a), o.team
                return

    def _shot_stoppers(self) -> None:
        b = self.ball
        defending = 1 - b.shot_by
        speed = math.hypot(b.vx, b.vy)
        if speed < 8.0:
            b.shot_by = None
            return
        gk = self.players[defending * 5]
        # The keeper's chance depends on reaction time: how long the shot takes to
        # reach him. A 35 m drive gives him over a second; a 10 m finish, a third.
        react = math.hypot(gk.x - b.shot_from[0], gk.y - b.shot_from[1]) / max(speed, 1.0)
        reach = 1.5 + 3.0 * react
        if not b.save_tried and math.hypot(gk.x - b.x, gk.y - b.y) < reach:
            b.save_tried = True
            odds = 0.4 + 0.75 * react - (0.05 if not b.shot_low else 0.0)
            if self.rng.random() < _clamp(odds, 0.2, 0.92):
                self.stats[defending].saves_made += 1
                self.events.append((round(self.time, 1), "save", defending, 0, round(speed, 1)))
                if self.rng.random() < 0.6:
                    self._gain(gk)
                else:
                    b.vx, b.vy = -0.3 * b.vx + self.rng.uniform(-3, 3), self.rng.uniform(-6, 6)
                    b.shot_by, b.last_team = None, defending
                return
        if b.shot_low:
            for o in self.players[defending * 5 + 1: defending * 5 + 5]:
                if math.hypot(o.x - b.x, o.y - b.y) < 0.8 and self.rng.random() < 0.4:
                    b.vx, b.vy = -0.2 * b.vx + self.rng.uniform(-4, 4), self.rng.uniform(-5, 5)
                    b.shot_by, b.last_team = None, defending
                    return

    def _pickup(self) -> None:
        b = self.ball
        speed = math.hypot(b.vx, b.vy)
        near = [p for p in self.players if p.cooldown <= 0
                and math.hypot(p.x - b.x, p.y - b.y) < CONTROL_RADIUS]
        if not near:
            return
        p = min(near, key=lambda q: math.hypot(q.x - b.x, q.y - b.y))
        if p is b.pass_to:
            # The intended receiver is expecting it: a pass is played to be
            # controlled. Without this, a 10 m pass (~13 m/s) was fumbled half the
            # time and short passes completed 31%, where real ones complete ~85%.
            if speed > 24.0 or (speed > 16.0 and self.rng.random() < 0.1):
                return
        else:
            limit = 25.0 if p.idx == 0 else 20.0
            if speed > limit or (speed > 12.0 and self.rng.random() < 0.5):
                return
        self._gain(p)

    def _in_own_box(self, p: Player) -> bool:
        return _side(p.team) * p.x < -(HALF_X - BOX_DEPTH) and abs(p.y) < BOX_HALF_WIDTH

    def _gain(self, p: Player) -> None:
        b = self.ball
        back_pass = b.pass_from is not None and b.pass_from.team == p.team and b.pass_from is not p
        if back_pass:
            self.stats[p.team].passes_completed += 1
        # Only the keeper may handle, and only in his own box. A deliberate pass
        # from a teammate must be played with the feet (the back-pass law), so
        # that keeper can still be tackled.
        b.in_hands = p.idx == 0 and self._in_own_box(p) and not back_pass
        b.owner, b.pass_from, b.pass_to, b.shot_by, b.air = p, None, None, None, 0.0
        b.vx = b.vy = 0.0
        b.protect, b.hold, b.last_team = 0.5, 0.0, p.team
        # ASSUMPTION: an order aimed at the opposing carrier ends once you are the
        # carrier, and the default AI carries the ball until your next turn.
        # Otherwise a presser who wins the ball would go on "pressing" himself.
        if p.order is not None and p.order["type"] in MAINTAINED:
            p.order = None

    def _out_of_play(self) -> bool:
        b = self.ball
        if abs(b.x) >= HALF_X and abs(b.y) < GOAL_HALF_WIDTH:
            self._goal(0 if b.x > 0 else 1)
            return True
        if BALL_STAYS_IN:
            # Official rules: "the ball never goes out of bounds" - no throw-ins,
            # corners or goal kicks. It comes back off the boundary, slower.
            if abs(b.x) >= HALF_X:
                b.x = math.copysign(HALF_X - 0.1, b.x)
                b.vx = -0.5 * b.vx
            if abs(b.y) >= HALF_Y:
                b.y = math.copysign(HALF_Y - 0.1, b.y)
                b.vy = -0.5 * b.vy
            return False
        if abs(b.x) >= HALF_X:
            scorer = 0 if b.x > 0 else 1
            if b.last_team == scorer:  # attacker put it out: goal kick
                self._restart(self.players[(1 - scorer) * 5], -_side(1 - scorer) * 52.0, 0.0)
            else:                        # defender put it out: corner
                taker = self._nearest(scorer, b.x, b.y)
                self._restart(taker, _side(scorer) * 54.0, math.copysign(34.0, b.y))
            return True
        if abs(b.y) >= HALF_Y:
            team = 1 - b.last_team
            taker = self._nearest(team, b.x, b.y, outfield=True)
            self._restart(taker, _clamp(b.x, -53.0, 53.0), math.copysign(34.5, b.y))
            return True
        return False

    def _nearest(self, team: int, x: float, y: float, outfield: bool = False) -> Player:
        pool = [p for p in self.players if p.team == team and (p.idx != 0 or not outfield)]
        return min(pool, key=lambda p: math.hypot(p.x - x, p.y - y))

    def _restart(self, taker: Player, x: float, y: float) -> None:
        taker.x, taker.y = x, y
        for o in self.players:  # opponents give the taker 4 m
            if o.team != taker.team and math.hypot(o.x - x, o.y - y) < 4.0:
                d = math.hypot(o.x - x, o.y - y) or 1.0
                o.x, o.y = x + (o.x - x) / d * 4.0, y + (o.y - y) / d * 4.0
        self._gain(taker)
        self.ball.x, self.ball.y = x, y
        self.ball.protect = 1.0

    def _goal(self, scorer: int) -> None:
        self.stats[scorer].goals += 1
        b = self.ball
        if b.owner is not None and b.owner.team == scorer:
            credit = b.owner.idx          # walked it in
        elif b.shot_by == scorer:
            credit = b.shooter            # the shot that went in
        else:
            credit = None                 # own goal or deflection
        self.events.append((round(self.time, 1), "goal", scorer, credit,
                            f"{self.stats[0].goals}-{self.stats[1].goals}"))
        self._kickoff(1 - scorer)

    def _spot(self, team: int, idx: int) -> tuple:
        x, y = FORMATION[idx]
        s = _side(team)
        return s * x, s * y

    def _kickoff(self, team: int) -> None:
        for p in self.players:
            p.x, p.y = self._spot(p.team, p.idx)
            p.vx = p.vy = 0.0
            p.order = None
        taker = self.players[team * 5 + 2]
        taker.x, taker.y = -_side(team) * 0.5, 0.0
        self._gain(taker)
        self.ball.x, self.ball.y = 0.0, 0.0
        self.ball.protect = 1.0

    def _record(self) -> None:
        b = self.ball
        self.frames.append([round(self.time, 1), self.stats[0].goals, self.stats[1].goals,
                            round(b.x, 1), round(b.y, 1),
                            [[round(p.x, 1), round(p.y, 1)] for p in self.players],
                            None if b.owner is None else b.owner.team * 5 + b.owner.idx])
