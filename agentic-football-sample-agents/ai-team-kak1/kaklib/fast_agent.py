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
import random
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
# Model off by default: live, it overrode our shape (ball-magnet presses), ~10% of replies were
# invalid, and each call adds ~0.4 s model + ~0.6 s platform time on a 2 s tick. Turn it back on
# (USE_MODEL=1) only if a live A/B shows it helps.
#   USE_MODEL=0       rules only (default)
#   USE_MODEL=1       free-form model reply checked against the rules (old mode)
#   USE_MODEL=choice  on the ball, the model picks one of 2-4 options built by tactics.candidates()
MODEL_MODE = os.getenv("USE_MODEL", "0")
USE_MODEL = MODEL_MODE == "1"
CHOICE = MODEL_MODE == "choice"
CHOICE_MAX_TOKENS = 4

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


# What worked in our own matches, written by make_lessons.py from tools/mine_examples.py output.
def _load_lessons(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")):
    try:
        with open(path) as fh:
            d = json.load(fh)
        return [str(x)[:200] for x in d.get("lines", [])][:8], d.get("version")
    except Exception:
        return [], None


LESSONS, LESSONS_VERSION = _load_lessons()

CHOICE_PROMPT = """You are the {role} (player {pid}) in a 5-a-side AI football match, and you have the ball.
Pick the best of the options you are given. Reply with ONLY the option letter, nothing else.
Distances are in pitch units (the pitch is about {length} long). "lane" = how far the nearest
opponent is from the pass line; "receiverFree" = how far the nearest opponent is from the receiver.
Bigger lane and receiverFree = safer; bigger gain = more progress toward their goal.
{lessons}{rules}"""

_choice_model = None


def _get_choice_model():
    global _choice_model
    if _choice_model is None:
        import botocore.config
        from strands.models import BedrockModel
        _choice_model = BedrockModel(
            model_id=MODEL_ID, temperature=TEMPERATURE, max_tokens=CHOICE_MAX_TOKENS,
            region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            boto_client_config=botocore.config.Config(
                read_timeout=2, connect_timeout=1, retries={"max_attempts": 0}),
        )
    return _choice_model


def _new_choice_agent(role, pid):
    from strands import Agent
    lessons = ("What has worked in our matches:\n" + "".join(f"- {x}\n" for x in LESSONS)) if LESSONS else ""
    rules = ("Coach rules:\n" + "".join(f"- {r}\n" for r in tactics.COACH_PROMPT_RULES)) if tactics.COACH_PROMPT_RULES else ""
    return Agent(model=_get_choice_model(), callback_handler=None, retry_strategy=None,
                 system_prompt=CHOICE_PROMPT.format(role=role, pid=pid, length=round(2 * tactics.PITCH_HALF_X, 1),
                                                    lessons=lessons, rules=rules))


def _choice_message(f, opts):
    """Situation + lettered options, short."""
    score = f.get("score") or {}
    lines = [f"Score {score.get('home', 0)}-{score.get('away', 0)} (we are {'home' if f['attackDirection'] == '+x' else 'away'}), "
             f"time {round(f.get('gameTime') or 0)}s of {int(tactics.MATCH_SECONDS)}, mode {f.get('mode')}. "
             f"You are {f['distToOppGoal'] / tactics.SCALE_X:.1f} from their goal, nearest opponent "
             f"{f['nearestOppDist'] / tactics.SCALE_X:.1f} away.", "Options:"]
    for letter, o in zip("ABCD", opts):
        c, i = o["cmd"][0], o["info"]
        if c["commandType"] == "PASS":
            lines.append(f"{letter}) {c['parameters']['type']} pass to {i['to']}: dist {i['dist']}, lane {i['lane']}, "
                         f"receiverFree {i['receiverFree']}, gain {i['gain']}")
        else:
            lines.append(f"{letter}) dribble forward: gain {i['gain']}, nearest opponent {i['nearestOpp']}")
    return "\n".join(lines)


def _pick(raw, opts):
    """First A-D letter in the reply -> that option, else None."""
    for ch in (raw or "").strip().upper()[:6]:
        if ch in "ABCD"[:len(opts)]:
            return opts["ABCD".index(ch)]
    return None


def _new_agent(role, pid):
    from strands import Agent
    # New Agent per call = no history (stateless, small prompt, safe under parallel calls).
    rules = "".join(f"\n- {r}" for r in tactics.COACH_PROMPT_RULES)
    notes = ROLE_NOTES.get(role, "") + (f"\nCoach rules:{rules}" if rules else "")
    return Agent(model=_get_model(), callback_handler=None, retry_strategy=None,
                 system_prompt=BASE_PROMPT.format(role=role, pid=pid, half=tactics.PITCH_HALF_X, half_y=round(tactics.PITCH_HALF_Y, 2),
                                                  role_notes=notes))


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
_last_raw_t = None       # gameTime of the last payload seen (a drop = new match)
_mid_logged = False      # logged this match's open-play payload yet?


def _log_raw_payload(data):
    """Log one full raw payload per match (when game time restarts), with real numbers.
    Skips payloads without players, e.g. the portal's 'Test your agents' sample."""
    global _last_raw_t, _mid_logged
    gs = data.get("gameState") or {}
    t = gs.get("gameTime") or 0
    if not gs.get("players"):
        return
    new_match = _last_raw_t is None or t < _last_raw_t
    _last_raw_t = t
    if new_match:
        _mid_logged = False
        log.info(json.dumps({"raw_payload": data, "coach_version": tactics.COACH_VERSION})[:20000])
    elif not _mid_logged and t >= 20:      # one open-play payload too (kick-off uses other units)
        _mid_logged = True
        log.info(json.dumps({"raw_payload": data})[:20000])


def _ball_height(game_state):
    """Ball height (live payload y when its sideways axis is z). Its presence marks logs where
    'ball' is [x, sideways] for real; older logs had the height in the second slot."""
    bp = (game_state.get("ball") or {}).get("position") or {}
    if tactics.BALL_LATERAL == "z" and "z" in bp:
        return round(bp.get("y", 0.0), 2)
    return None


def _where(game_state, team_id, pid):
    """Live-pitch positions for the decision log, so a bad match can be read back."""
    try:
        f = tactics.facts(game_state, team_id, pid) or {}
        return {"t": round(game_state.get("gameTime") or 0), "me": f.get("me"), "ball": f.get("ball"),
                "owner": f.get("ballOwner"), "team": team_id, "score": game_state.get("score"),
                "opps": f.get("opps"), "ball_h": _ball_height(game_state),
                "pm": game_state.get("playMode")}
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
    hard = f and (not USE_MODEL or pid == 0 or fallback[0]["commandType"] == "SHOOT"
                  or (f["hasBall"] and f.get("inOwnBox")))
    opts = tactics.candidates(game_state, team_id, pid) if (CHOICE and f) else []
    if len(opts) >= 2:
        # Nova Micro favours letter B (21 of 39 test picks): shuffle so the bias doesn't always
        # land on the same kind of action.
        opts = random.sample(opts, len(opts))
    choice = None
    if len(opts) >= 2:
        agent = None
        try:
            if not _inflight.acquire(blocking=False):
                raise _Busy()
            try:
                agent = _new_choice_agent(f["role"], pid)
                fut = _POOL.submit(_ask_llm, agent, _choice_message(f, opts))
            except BaseException:
                _inflight.release()
                raise
            raw = await asyncio.wait_for(asyncio.wrap_future(fut), TIMEOUT_S)
            opt = _pick(raw, opts)
            if opt is None:
                source = "fallback_invalid"
            elif not tactics.allowed(game_state, team_id, pid, opt["cmd"][0]):
                source = "override_" + opt["cmd"][0]["commandType"].lower()
            else:
                source, out, choice = "llm_choice", opt["cmd"], opt["label"]
        except asyncio.TimeoutError:
            source = "fallback_timeout"
            try:
                agent.cancel()
            except Exception:
                pass
        except _Busy:
            source = "fallback_busy"
        except Exception as e:
            source = f"fallback_error:{type(e).__name__}"
    elif f and not hard:
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
                         **({"options": [o["label"] for o in opts], "choice": choice} if opts else {}),
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
                "playMode": gs.get("playMode"), "gameTime": gs.get("gameTime"),
                "coach_version": tactics.COACH_VERSION, "coach_params": tactics.COACH_APPLIED}}))
        _log_raw_payload(data)
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
