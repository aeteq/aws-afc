"""kak1 FWD1 check. Offline by default (model call disabled); --llm calls Bedrock.
Run from ai-team-kak1/:  python3 ai-fwd1/test_local.py [--llm]"""
import asyncio, json, os, sys, time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))
import main  # noqa: E402  (builds the app + handler exactly as deployed)
import fast_agent as fa  # noqa: E402

PID, LLM = 3, "--llm" in sys.argv


def state(holder):
    players = []
    for team, code in ((0, "home"), (1, "away")):
        for i in range(5):
            x = (-50 + i * 17.5) * (1 if team == 0 else -1)
            players.append({"agentId": f"agentId_{i}", "teamCode": code, "position": {"x": x, "y": i * 4 - 8},
                            "velocity": {"x": 0, "y": 0}, "stamina": 80})
    bx = players[holder]["position"]["x"] + 0.5
    return {"tick": 1, "gameTime": 30, "playMode": "OPEN_PLAY", "score": {"home": 0, "away": 0},
            "ball": {"position": {"x": bx, "y": 0, "z": 0}, "velocity": {"x": 0, "y": 0, "z": 0}, "isFree": False,
                     "possessionAgentId": f"agentId_{holder}"}, "players": players, "teamChat": []}


if not LLM:
    def _off(role, pid):
        raise RuntimeError("model disabled offline")
    fa._new_agent = _off

ok = True
for team in (0, 1):
    for holder in (PID, (PID + 1) % 5, 2):
        payload = {"prompt": json.dumps({"gameState": state(holder), "teamId": team, "myPlayers": [PID]})}

        t0 = time.perf_counter()
        out = asyncio.run(fa.decide(json.loads(payload["prompt"])["gameState"], team, PID))
        ms = (time.perf_counter() - t0) * 1000
        c = out[0]
        good = c["playerId"] == PID and c["teamId"] == team and fa._params_ok(c, PID) if c["commandType"] in fa.PARAMS \
            else c["playerId"] == PID
        ok &= bool(good)
        print(f"{'OK ' if good else 'BAD'} team {team} holder {holder}: {c['commandType']:14s} {c['parameters']}  ({ms:.0f} ms)")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
