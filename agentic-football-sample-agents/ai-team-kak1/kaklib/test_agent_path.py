"""Checks for the active decision path: tactics possession, command validation,
model timeouts, and player selection. Run: python test_agent_path.py (no AWS needed)."""
import asyncio, json, os, sys, threading, time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "lib"))   # shared workshop lib
sys.path.insert(0, _here)

import fast_agent as fa  # noqa: E402
import tactics as t

passed = 0


def check(name, cond):
    global passed
    assert cond, name
    passed += 1
    print("ok  ", name)


def state(fmt="new", holder=3, gk_line=True):
    """FWD1 (id 3, home) at x=20: 35 from goal, outside normal shoot range, inside long range."""
    players = []
    for team, code in ((0, "home"), (1, "away")):
        for i in range(5):
            x = (-50 + i * 17.5) * (1 if team == 0 else -1)
            p = {"position": {"x": x, "y": i * 4 - 8}, "velocity": {"x": 0, "y": 0}, "stamina": 0.8}
            if fmt == "new":
                p.update(agentId=f"agentId_{i}", teamCode=code)
            else:
                p.update(playerId=i, teamId=team, hasBall=(team == 0 and i == holder))
            players.append(p)
    ball = {"position": {"x": 20.5, "y": 4}, "velocity": {"x": 0, "y": 0}}
    if fmt == "new":
        ball.update(isFree=False, possessionAgentId=f"agentId_{holder}")
    return {"tick": 1, "gameTime": 30, "score": {"home": 0, "away": 0}, "ball": ball,
            "players": players, "teamChat": []}


# 1. hasBall-only format
old = state(fmt="old")
check("hasBall format: FWD1 sees it has the ball", t.facts(old, 0, 3)["hasBall"] is True)
check("hasBall format: FWD1 makes an on-ball decision",
      t.decide(old, 0, 3)[0]["commandType"] == t.decide(state(), 0, 3)[0]["commandType"])
check("hasBall format: MID knows a teammate has it", t.facts(old, 0, 2)["ballOwner"] == "team")

# 2. command validation
v = lambda c, has_ball=False, pid=2: fa._validate([c], 0, pid, has_ball)   # noqa: E731
base = {"playerId": 2, "teamId": 0}
check("MOVE_TO with {} rejected", v({**base, "commandType": "MOVE_TO", "parameters": {}}) is None)
check("MOVE_TO valid accepted", v({**base, "commandType": "MOVE_TO",
                                   "parameters": {"target_x": 10, "target_y": 2, "sprint": True}}))
check("MOVE_TO off the pitch rejected", v({**base, "commandType": "MOVE_TO",
                                           "parameters": {"target_x": 500, "target_y": 2}}) is None)
check("SHOOT bad aim rejected", v({**base, "commandType": "SHOOT",
                                   "parameters": {"aim_location": "TOP", "power": 1}}, True) is None)
check("unknown param rejected", v({**base, "commandType": "PRESS_BALL",
                                   "parameters": {"intensity": 0.5, "foo": 1}}) is None)
check("PASS to self rejected", v({**base, "commandType": "PASS",
                                  "parameters": {"target_player_id": 2}}, True) is None)
check("wrong playerId rejected", v({"playerId": 4, "teamId": 0, "commandType": "INTERCEPT",
                                    "parameters": {}}) is None)


# 3. timeouts stop the call, and don't pile up
class SlowAgent:
    """Fake model call: takes 1 s unless cancelled."""
    def __init__(self):
        self.cancelled = threading.Event()

    def __call__(self, msg):
        self.cancelled.wait(1.0)
        return "[]"

    def cancel(self):
        self.cancelled.set()


made = []
fa.TIMEOUT_S = 0.4            # the fake call takes 1 s; test the cut-off, not the default
fa._new_agent = lambda role, pid: made.append(SlowAgent()) or made[-1]
gs = state(holder=2)          # MID (2) has the ball: no hard rule, so the model is asked
logs = []
fa.log.info = lambda m: logs.append(json.loads(m))
t0 = time.perf_counter()
out = fa.decide_sync(gs, 0, 3)
took = time.perf_counter() - t0
check(f"timeout returns near the deadline ({took*1000:.0f} ms)", took < fa.TIMEOUT_S + 0.15)
check("timeout falls back to tactics", logs[-1]["src"] == "fallback_timeout" and out)
check("timed-out call was cancelled", made[-1].cancelled.wait(0.2))
time.sleep(0.05)
check("cancelled call released its slot", fa._inflight._value == fa.MAX_INFLIGHT)
for _ in range(fa.MAX_INFLIGHT):
    fa._inflight.acquire()
fa.decide_sync(gs, 0, 3)
check("no free slot -> fallback_busy, no new call", logs[-1]["src"] == "fallback_busy")
for _ in range(fa.MAX_INFLIGHT):
    fa._inflight.release()


# 4. player selection from myPlayers
class FakeApp:
    class logger:
        error = warning = info = staticmethod(lambda *a, **k: None)

    def entrypoint(self, fn):
        self.fn = fn
        return fn


fa.decide = lambda gs, team, pid: asyncio.sleep(0, result=[t.cmd("INTERCEPT", team, pid)])


def run(default_pid, my_players):
    app = FakeApp()
    fa.create_handler(app, default_pid)
    prompt = json.dumps({"gameState": state(), "teamId": 0, "myPlayers": my_players})

    async def go():
        return [x async for x in app.fn({"prompt": prompt}, None)]
    return json.loads(asyncio.run(go())[0])[0]["playerId"]


check("myPlayers [0..4]: FWD1 agent controls player 3", run(3, [0, 1, 2, 3, 4]) == 3)
check("myPlayers [3]: controls 3", run(3, [3]) == 3)
check("myPlayers missing: controls its own id", run(1, []) == 1)

print(f"\n{passed} checks passed")
