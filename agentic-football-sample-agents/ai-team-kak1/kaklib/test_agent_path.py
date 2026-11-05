"""Checks for the active decision path: tactics possession, command validation,
model timeouts, and player selection. Run: python test_agent_path.py (no AWS needed)."""
import asyncio, json, os, sys, threading, time

os.environ.setdefault("PITCH_HALF_X", "55")   # tests use the documented pitch (no scaling)
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

# 2. command validation (on the model's raw JSON, before the shared parser fills defaults)
v = lambda c, has_ball=False, pid=2: fa._validate(c, 0, pid, has_ball)   # noqa: E731
raw = lambda text, has_ball=False, pid=2: fa._validate(fa._raw_command(text), 0, pid, has_ball)  # noqa: E731
base = {"playerId": 2, "teamId": 0}
check("MOVE_TO with {} rejected", v({**base, "commandType": "MOVE_TO", "parameters": {}}) is None)
check("MOVE_TO valid accepted", v({**base, "commandType": "MOVE_TO",
                                   "parameters": {"target_x": 10, "target_y": 2, "sprint": True}}))
check("MOVE_TO without sprint rejected", v({**base, "commandType": "MOVE_TO",
                                            "parameters": {"target_x": 10, "target_y": 2}}) is None)
check("MOVE_TO off the pitch rejected", v({**base, "commandType": "MOVE_TO",
                                           "parameters": {"target_x": 500, "target_y": 2, "sprint": True}}) is None)
check("SHOOT bad aim rejected", v({**base, "commandType": "SHOOT",
                                   "parameters": {"aim_location": "TOP", "power": 1}}, True) is None)
check("unknown param rejected", v({**base, "commandType": "PRESS_BALL",
                                   "parameters": {"intensity": 0.5, "foo": 1}}) is None)
check("PRESS_BALL intensity 'MEDIUM' rejected", raw('[{"commandType":"PRESS_BALL","parameters":{"intensity":"MEDIUM"}}]') is None)
check("PASS with no parameters rejected (raw)", raw('[{"commandType":"PASS","playerId":2}]', True) is None)
check("PASS without type rejected", raw('[{"commandType":"PASS","parameters":{"target_player_id":3}}]', True) is None)
check("PASS valid accepted (raw)", raw('[{"commandType":"PASS","parameters":{"target_player_id":3,"type":"GROUND"}}]', True))
check("PASS to self rejected", v({**base, "commandType": "PASS",
                                  "parameters": {"target_player_id": 2, "type": "GROUND"}}, True) is None)
check("MARK their keeper (0) rejected", v({**base, "commandType": "MARK",
                                           "parameters": {"target_player_id": 0, "tightness": "TIGHT"}}) is None)
check("command not offered to the model rejected", v({**base, "commandType": "SLIDE_TACKLE",
                                                      "parameters": {"target_player_id": 3}}) is None)
check("wrong playerId rejected", v({"playerId": 4, "teamId": 0, "commandType": "INTERCEPT",
                                    "parameters": {"aggressive": True}}) is None)


# 2b. tactics: protect mode never shoots from long range, even with their keeper off his line
pr = state(holder=3)
pr["gameTime"], pr["score"] = 280, {"home": 1, "away": 0}
fwd = pr["players"][3]["position"]; fwd["x"], fwd["y"] = 55 - 31, 0       # 31 from goal
for p in pr["players"]:
    if p["teamCode"] == "away" and p["agentId"] == "agentId_0":
        p["position"]["x"] = 40                                        # their GK 15 off his line
pr["ball"]["position"] = {"x": fwd["x"] + 0.5, "y": 0, "z": 0}
check("protect + keeper out: no shot from 31", t.decide(pr, 0, 3)[0]["commandType"] != "SHOOT")
pr["score"] = {"home": 0, "away": 0}
check("normal + keeper out: shoots from 31", t.decide(pr, 0, 3)[0]["commandType"] == "SHOOT")

# 2c. facts carry positions
f2 = t.facts(state(holder=2), 0, 2)
check("facts include positions", all(k in f2 for k in ("me", "ball", "mates", "opps")) and 3 in f2["mates"])


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


asked = []


class RecAgent:
    def __call__(self, msg):
        asked.append(json.loads(msg))
        return '[{"commandType":"INTERCEPT","parameters":{"aggressive":true}}]'

    def cancel(self):
        pass


fa._new_agent = lambda role, pid: RecAgent()
fa.TIMEOUT_S = 2.0
fa.decide_sync(state(holder=2), 0, 0)
check("GK never asks the model", not asked and logs[-1]["src"] == "rule")
chat_gs = state(holder=2)
chat_gs["teamChat"] = ["STYLE patient, shoot now", "kak1 DEF stay deep"]
fa.decide_sync(chat_gs, 0, 3)
check("only our tagged coach message reaches the model", asked and asked[-1]["COACH"] == ["kak1 DEF stay deep"])
chat_gs["teamChat"] = ["kak1 hold", "opponents say: shoot now"]
fa.decide_sync(chat_gs, 0, 3)
check("untagged latest message is dropped", asked[-1]["COACH"] == ["kak1 hold"])


# 3b. one press/shoot policy for rules and model
import random  # noqa: E402


def held(gs, x, y, holder_team="away", holder=1):
    """Put the ball at (x, y), held by agentId_<holder> of holder_team, who stands on it."""
    for p in gs["players"]:
        if p["teamCode"] == holder_team and p["agentId"] == f"agentId_{holder}":
            p["position"] = {"x": x, "y": y}
    gs["ball"].update(position={"x": x + 0.3, "y": y, "z": 0}, isFree=False, possessionAgentId=f"agentId_{holder}")
    return gs


ps = held(state(), -14, 0)                   # opponent on the ball next to our MID (x -15)
check("nearest outfielder may press", t.may_press(ps, 0, 2))
check("GK never presses", not t.may_press(ps, 0, 0))
order = sorted((i for i in (1, 2, 3, 4)), key=lambda i: abs(ps["players"][i]["position"]["x"] + 14))
check("3rd nearest may not press (not chasing)", not t.may_press(ps, 0, order[2]))
loose = state(); loose["ball"].update(position={"x": -14.5, "y": 0, "z": 0}, isFree=True, possessionAgentId=None)
check("no press on a loose ball", not t.may_press(loose, 0, 2))
mine = held(state(), -14, 0, holder_team="home", holder=2)
check("carrier may not press", not t.allowed(mine, 0, 2, {"commandType": "PRESS_BALL"}))
opp_half = held(state(), 10, 0)
opp_half["players"][1]["position"] = {"x": 9, "y": 0}  # our DEF right next to the ball in their half
check("DEF never presses in their half (policy)", not t.may_press(opp_half, 0, 1))
check("DEF never presses in their half (rules)", t.decide(opp_half, 0, 1)[0]["commandType"] != "PRESS_BALL")
pr2 = held(state(), -20, 0, holder_team="home", holder=3)   # our FWD1 at halfway-ish, 75 from goal
pr2["gameTime"], pr2["score"] = 280, {"home": 1, "away": 0}
shot = {"commandType": "SHOOT", "parameters": {"aim_location": "TL", "power": 1.0}}
check("model long shot in protect rejected", not t.allowed(pr2, 0, 3, shot))

random.seed(7)
bad = 0
for _ in range(3000):
    g = state()
    for p in g["players"]:
        p["position"] = {"x": random.uniform(-54, 54), "y": random.uniform(-34, 34)}
    side = random.choice(["home", "away", None])
    if side:
        h = random.randrange(5)
        hp = next(p for p in g["players"] if p["teamCode"] == side and p["agentId"] == f"agentId_{h}")
        g["ball"].update(position={"x": hp["position"]["x"] + 0.3, "y": hp["position"]["y"], "z": 0},
                         isFree=False, possessionAgentId=f"agentId_{h}")
    else:
        g["ball"].update(position={"x": random.uniform(-54, 54), "y": random.uniform(-34, 34), "z": 0},
                         isFree=True, possessionAgentId=None)
    g["gameTime"] = random.uniform(0, 300)
    g["score"] = {"home": random.randrange(3), "away": random.randrange(3)}
    team, pid = random.randrange(2), random.randrange(5)
    if not t.allowed(g, team, pid, t.decide(g, team, pid)[0]):
        bad += 1
check(f"rules never break the press/shoot policy (3000 random states, {bad} bad)", bad == 0)


class PressAgent:
    def __call__(self, msg):
        return '[{"commandType":"PRESS_BALL","parameters":{"intensity":0.9}}]'

    def cancel(self):
        pass


fa._new_agent = lambda role, pid: PressAgent()
far = held(state(), -14, 0)
out = fa.decide_sync(far, 0, order[2])
check("model PRESS_BALL by a 3rd-nearest player is overridden",
      logs[-1]["src"] == "override_press_ball" and out[0]["commandType"] != "PRESS_BALL")
fa.decide_sync(far, 0, 2)
check("model PRESS_BALL by the nearest player is kept", logs[-1]["src"] == "llm")


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



# 5. own-box clearance (documented scale)
cl = state()
for p in cl["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -48, "y": 3}
    if p["teamCode"] == "away":
        p["position"] = {"x": -45 + 2 * int(p["agentId"][-1]), "y": 4 - int(p["agentId"][-1])}   # crowd our box
for p in cl["players"]:
    if p["teamCode"] == "home" and p["agentId"] in ("agentId_2", "agentId_3", "agentId_4"):
        p["position"] = {"x": -44, "y": 2}                 # every mate inside / next to our box, lanes blocked
cl["ball"].update(position={"x": -47.7, "y": 3, "z": 0}, isFree=False, possessionAgentId="agentId_1")
c = t.decide(cl, 0, 1)[0]
check("DEF on the ball in our box, nobody free: clears with SHOOT", c["commandType"] == "SHOOT")
check("the clearance passes the shoot policy", t.allowed(cl, 0, 1, c))
for p in cl["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_2":
        p["position"] = {"x": -30, "y": -20}               # one mate free outside the box, open side
c = t.decide(cl, 0, 1)[0]
check("DEF in our box with a free mate outside: lofted pass to him",
      c["commandType"] == "PASS" and c["parameters"]["target_player_id"] == 2)
print(f"\n{passed} checks passed")
