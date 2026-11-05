"""Fast LLM decision layer: "the LLM reads the game, the code pulls the trigger".

Per call:
  1. Hard rules in code (no LLM): shoot in range, keeper distributes. Instant.
  2. Otherwise ask the LLM, with the tactics suggestion included, under a hard timeout.
  3. Validate the answer (Pydantic + "PASS/SHOOT only with the ball"). Invalid or late ->
     tactics.decide(). One JSON log line per decision: source + latency.

Env vars: MODEL_ID, LLM_TIMEOUT_MS (default 1500; tick ~2 s, server limit 5 s per the Actions page),
          MAX_TOKENS (default 120), TEMPERATURE (default 0.2), AWS_REGION.
"""

import asyncio
import copy
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

import tactics
from parsing import _iter_array_candidates, _loads

log = logging.getLogger("fast_agent")
if not log.handlers:  # one JSON line per decision on stdout (CloudWatch when deployed)
    _h = logging.StreamHandler(); _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h); log.setLevel(logging.INFO); log.propagate = False

MODEL_ID = os.getenv("MODEL_ID", "us.amazon.nova-micro-v1:0")
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_MS", "1500")) / 1000.0
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "2"))   # model calls still running in this process

# Own pool, not asyncio's default: asyncio.run() would otherwise wait for late calls on exit.
_POOL = ThreadPoolExecutor(max_workers=MAX_INFLIGHT + 1, thread_name_prefix="llm")
_inflight = threading.BoundedSemaphore(MAX_INFLIGHT)

ON_BALL = {"PASS", "SHOOT", "GK_DISTRIBUTE"}


class Command(BaseModel):
    """Shape the server accepts. Anything else falls back to tactics."""
    commandType: Literal["MOVE_TO", "PASS", "SHOOT", "PRESS_BALL", "MARK", "INTERCEPT",
                         "FOLLOW_PLAYER", "SLIDE_TACKLE", "GK_DISTRIBUTE", "CLEAR_OVERRIDE"]
    playerId: int
    teamId: int
    parameters: dict = {}
    duration: int = 0
    reason: Optional[str] = None   # ignored by the server, handy in logs


BASE_PROMPT = """You control ONE player ({role}, id {pid}) in a 5v5 football match.
Each turn you get FACTS (pre-computed) and SUGGESTION (a safe default command).
Return the SUGGESTION unless you see a clearly better option. Rules:
- PASS/SHOOT only if hasBall is true. Code already shoots when in range; you don't need to.
- If iAmNearestToBall and ballOwner is opponent: PRESS_BALL.
- Use the whole command set: MOVE_TO, PASS, SHOOT, PRESS_BALL, MARK, INTERCEPT.
- FACTS.mode: chase = we need a goal, take risks; protect = keep the lead, play safe.
- FACTS.style: counter = they push up, hit our forwards fast; patient = they sit deep, keep the ball.
{role_notes}
Commands (ALL parameters required; numbers are numbers, not words):
MOVE_TO{{target_x:number,target_y:number,sprint:true|false}}
PASS{{target_player_id:teammate id 1-4,type:"GROUND"|"AERIAL"|"THROUGH"}}
SHOOT{{aim_location:"TL"|"TR"|"BL"|"BR",power:0.0-1.0}}
PRESS_BALL{{intensity:0.0-1.0}}
MARK{{target_player_id:opponent id 1-4,tightness:"TIGHT"|"LOOSE"}}
INTERCEPT{{aggressive:true|false}}
Pitch x -{half}..{half} (goals at the two ends), y -{half_y}..{half_y}. FACTS has positions: me, ball, mates and opps as id: [x, y].
You attack toward FACTS.attackDirection.
Reply with ONLY a JSON array of one command, e.g.
[{{"commandType":"MOVE_TO","playerId":{pid},"parameters":{{"target_x":20,"target_y":5,"sprint":true}}}}]"""

ROLE_NOTES = {
    "GK": "- Stay near your goal. With the ball, GK_DISTRIBUTE to the most open teammate.",
    "DEF": "- Protect the goal. MARK the most dangerous attacker. With the ball, pass forward fast.",
    "MID": "- Link play. Keep your side channel (FACTS.side: +1 left, -1 right, 0 centre). Win loose balls (INTERCEPT). Pass to the forward in space.",
    "FWD": "- Score. Stay high (central if FACTS.side is 0). Run in behind when a teammate has the ball.",
}

_model = None


def _get_model():
    global _model
    if _model is None:
        import botocore.config
        from strands.models import BedrockModel
        _model = BedrockModel(
            model_id=MODEL_ID, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            boto_client_config=botocore.config.Config(
                read_timeout=2, connect_timeout=1, retries={"max_attempts": 0}),
        )
    return _model


def _new_agent(role, pid):
    from strands import Agent
    # New Agent per call = no history (stateless, small prompt, safe under parallel calls).
    return Agent(model=_get_model(), callback_handler=None, retry_strategy=None,
                 system_prompt=BASE_PROMPT.format(role=role, pid=pid, half=tactics.PITCH_HALF_X, half_y=round(tactics.PITCH_HALF_Y, 2),
                                                  role_notes=ROLE_NOTES.get(role, "")))


def _ask_llm(agent, user_msg):
    try:
        return str(agent(user_msg))
    finally:
        _inflight.release()


def _num(lo, hi):
    return lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def _one_of(*opts):
    return lambda v: v in opts


_MATE = lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 4   # noqa: E731
_BOOL = lambda v: isinstance(v, bool)                                               # noqa: E731

# commandType -> (required params, optional params). Unknown params are rejected too.
_OPP = lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 4    # noqa: E731

# Commands the MODEL may send: the ones the prompt offers, every parameter required.
# (The GK never asks the model; tactics code sends GK_DISTRIBUTE etc. directly.)
PARAMS = {
    "MOVE_TO": ({"target_x": _num(-55, 55), "target_y": _num(-35, 35), "sprint": _BOOL}, {}),   # +-documented (covers live)
    "PASS": ({"target_player_id": _MATE, "type": _one_of("GROUND", "AERIAL", "THROUGH")}, {}),
    "SHOOT": ({"aim_location": _one_of("TL", "TR", "BL", "BR", "CENTER"), "power": _num(0, 1)}, {}),
    "PRESS_BALL": ({"intensity": _num(0, 1)}, {}),
    "MARK": ({"target_player_id": _OPP, "tightness": _one_of("TIGHT", "LOOSE")}, {}),
    "INTERCEPT": ({"aggressive": _BOOL}, {}),
}


def _params_ok(c, pid):
    req, opt = PARAMS[c["commandType"]]
    p = c.get("parameters") or {}
    if any(k not in p or not ok(p[k]) for k, ok in req.items()):
        return False
    if any(k not in req and (k not in opt or not opt[k](v)) for k, v in p.items()):
        return False
    if c["commandType"] in ("PASS", "GK_DISTRIBUTE") and p["target_player_id"] in (pid, 0):
        return False           # can't pass to yourself or back to the keeper
    return True


def _raw_command(text):
    """First command object exactly as the model wrote it. The shared parse_commands fills in
    defaults (e.g. a PASS target) before we could see they were missing, so check the raw one."""
    for candidate in _iter_array_candidates(text):
        parsed = _loads(candidate, None)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return copy.deepcopy(parsed[0])
    parsed = _loads(text, None)
    if isinstance(parsed, dict) and "commandType" in parsed:
        return copy.deepcopy(parsed)
    return None


def _validate(raw_cmd, team_id, pid, has_ball):
    if not isinstance(raw_cmd, dict) or raw_cmd.get("commandType") not in PARAMS:
        return None
    if raw_cmd.get("playerId", pid) != pid:
        return None                                      # the model wrote someone else's id
    c = dict(raw_cmd, playerId=pid, teamId=team_id)
    params = c.get("parameters")
    if isinstance(params, dict) and "duration" in params:
        # Harmless slip seen live (15 of 85 rejected replies): "duration" written inside parameters.
        params = dict(params)
        d = params.pop("duration")
        c["parameters"] = params
        c.setdefault("duration", d if isinstance(d, int) and not isinstance(d, bool) else 0)
    try:
        c = Command(**c).model_dump(exclude_none=True)
    except ValidationError:
        return None
    if not _params_ok(c, pid):
        return None
    if c["commandType"] in ON_BALL and not has_ball:
        return None            # would be wasted on the server
    if c["commandType"] == "GK_DISTRIBUTE" and pid != 0:
        return None
    c.pop("reason", None)
    return [c]


_seen_payload = False


def _where(game_state, team_id, pid):
    """Live-pitch positions for the decision log, so a bad match can be read back."""
    try:
        f = tactics.facts(game_state, team_id, pid) or {}
        return {"t": round(game_state.get("gameTime") or 0), "me": f.get("me"), "ball": f.get("ball"),
                "owner": f.get("ballOwner"), "team": team_id, "score": game_state.get("score")}
    except Exception:  # logging must never break a decision
        return {}


class _Busy(Exception):
    pass


async def decide(game_state, team_id, pid):
    t0 = time.perf_counter()
    fallback = tactics.decide(game_state, team_id, pid)
    f = tactics.facts(game_state, team_id, pid)
    source, out, raw = "rule", fallback, None

    # Final without the model: the goalkeeper (fully coded), and shots.
    # Final without the model: the goalkeeper, shots, and clearing our own box.
    hard = f and (pid == 0 or fallback[0]["commandType"] == "SHOOT" or (f["hasBall"] and f.get("inOwnBox")))
    if f and not hard:
        msg = json.dumps({"FACTS": f, "SUGGESTION": fallback[0],
                          "COACH": tactics.coach_messages(game_state.get("teamChat"), team_id)[-1:]})
        agent = None
        try:
            if not _inflight.acquire(blocking=False):
                raise _Busy()                       # earlier calls still running: don't pile up
            try:
                agent = _new_agent(f["role"], pid)
                fut = _POOL.submit(_ask_llm, agent, msg)
            except BaseException:
                _inflight.release()
                raise
            raw = await asyncio.wait_for(asyncio.wrap_future(fut), TIMEOUT_S)
            got = _validate(_raw_command(raw), team_id, pid, f["hasBall"])
            if not got:
                source = "fallback_invalid"
            elif not tactics.allowed(game_state, team_id, pid, got[0]):
                source = "override_" + got[0]["commandType"].lower()   # e.g. override_press_ball
            else:
                source, out = "llm", got
        except asyncio.TimeoutError:
            source = "fallback_timeout"
            try:
                agent.cancel()                      # aborts the Bedrock stream at the next chunk
            except Exception:
                pass
        except _Busy:
            source = "fallback_busy"
        except Exception as e:  # throttling, auth, network
            source = f"fallback_error:{type(e).__name__}"
    log.info(json.dumps({"pid": pid, "src": source, "ms": round((time.perf_counter() - t0) * 1000),
                         "cmd": out[0]["commandType"], "tick": game_state.get("tick"),
                         "params": out[0].get("parameters"), **_where(game_state, team_id, pid),
                         "raw": (raw or "")[:120]}))
    return out


def decide_sync(game_state, team_id, pid):
    return asyncio.run(decide(game_state, team_id, pid))


def create_handler(app, default_pid):
    """Register the AgentCore entrypoint. Payload: {"prompt": "<json with gameState, teamId, myPlayers>"}."""
    from agent_base import parse_prompt, team_id_from

    @app.entrypoint
    async def invoke(payload, context):
        data = parse_prompt(payload)
        team_id = team_id_from(data, app.logger, f"P{default_pid}")
        # One agent per player. If the server lists the whole team, control our own id,
        # not myPlayers[0] (the stock sample's bug: every agent would drive player 0).
        mine = data.get("myPlayers") or []
        global _seen_payload
        if not _seen_payload:                       # once per process: confirm the live contract
            _seen_payload = True
            gs = data.get("gameState", {})
            log.info(json.dumps({"payload_check": {
                "myPlayers": mine, "teamId": data.get("teamId"),
                "ball_keys": sorted((gs.get("ball") or {}).keys()),
                "player_keys": sorted(((gs.get("players") or [{}])[0]).keys()),
                "playMode": gs.get("playMode"), "gameTime": gs.get("gameTime")}}))
        pid = mine[0] if len(mine) == 1 else default_pid
        if len(mine) > 1:
            app.logger.warning(f"myPlayers has {len(mine)} ids; controlling {pid}")
        try:
            cmds = await decide(data.get("gameState", {}), team_id, pid)
        except Exception as e:
            app.logger.error(f"decide failed: {e}")
            cmds = [tactics.cmd("CLEAR_OVERRIDE", team_id, pid)]
        yield json.dumps(cmds)

    return invoke
