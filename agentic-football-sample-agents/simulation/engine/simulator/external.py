"""Field a team that is not in this repo: another team's code, or their running agents.

A team spec, given wherever the simulator takes a team name (run_match.py --home, --away,
--team, --opponent):

  py:FILE.py[:FUNC]                 a function FUNC(game_state, team_id, player_id) returning
                                    the player's commands (default FUNC: commands). Runs in
                                    this process, so read the code before you run it.
  http://HOST:PORT[,...]            agents served over AgentCore's HTTP protocol: `agentcore
                                    dev`, or src/main.py run with PORT set. One URL serves all
                                    five players, or give five, one per player (GK, DEF, MID,
                                    FWD1, FWD2), or a port range: http://HOST:8080-8084
  arn:aws:bedrock-agentcore:...     deployed AgentCore runtimes, one or five ARNs, invoked with
                                    your AWS credentials (needs boto3 and permission to
                                    InvokeAgentRuntime on them)
  @FILE                             any of the above read from a file, one entry per line
  NAME=SPEC                         the name to show in results

Remote agents are called the way the game server calls them: the request body is
{"prompt": "{\"gameState\": ..., \"teamId\": 0|1, \"myPlayers\": [player]}"}, all five
players of a team are asked in parallel on the same snapshot, and an answer that arrives
after the deadline (default 1 s, 0 for none) is dropped, as live. The engine rejects
commands tagged with the wrong team or player, as the server does. Late and unreadable
answers are counted on the team (calls, late, errors, summary()).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

DEADLINE = 1.0

_NAMED = re.compile(r"^([A-Za-z][\w-]{0,19})=(.+)$", re.S)
_PORT_RANGE = re.compile(r"^(https?://[^/:]+):(\d+)-(\d+)(/.*)?$")

# Shared by every remote team: a call that misses its deadline keeps running, and must not
# hold up the next turn's calls behind it.
_POOL = ThreadPoolExecutor(max_workers=128, thread_name_prefix="agent-call")


def parse_spec(spec: str) -> tuple[str, str, list[str]]:
    """(name, kind, entries) for a team spec; kind is "py", "http" or "agentcore"."""
    name = None
    m = _NAMED.match(spec)
    if m and not spec.startswith(("py:", "http", "arn:")):
        name, spec = m.group(1), m.group(2)
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text().splitlines()
        spec = ",".join(ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))
    entries = [e for e in re.split(r"[,\s]+", spec) if e]

    if len(entries) == 1 and entries[0].startswith("py:"):
        return name or Path(entries[0][3:].split(":")[0]).stem, "py", [entries[0][3:]]
    if entries and all(e.startswith(("http://", "https://")) for e in entries):
        kind = "http"
        if len(entries) == 1 and (r := _PORT_RANGE.match(entries[0])):
            host, lo, hi, path = r.group(1), int(r.group(2)), int(r.group(3)), r.group(4) or ""
            entries = [f"{host}:{port}{path}" for port in range(lo, hi + 1)]
    elif entries and all(e.startswith("arn:") for e in entries):
        kind = "agentcore"
    else:
        raise SystemExit(f"unknown team '{spec}': use a built-in name, py:FILE.py[:FUNC], "
                         "http://HOST:PORT, arn:aws:bedrock-agentcore:..., or @FILE "
                         "(see simulator/external.py)")
    if len(entries) not in (1, 5):
        raise SystemExit(f"{kind} team needs 1 entry for all five players or 5, one per "
                         f"player; got {len(entries)}")
    return name or kind, kind, entries


def make_external(spec: str, deadline: float | None = None):
    name, kind, entries = parse_spec(spec)
    deadline = DEADLINE if deadline is None else deadline
    if kind == "py":
        return ModuleTeam(name, entries[0])
    if kind == "http":
        return HttpTeam(name, entries, deadline)
    return AgentCoreTeam(name, entries, deadline)


def commands_from(answer) -> list:
    """The command list in an agent's answer: server-sent events, JSON, or a JSON string of it."""
    if isinstance(answer, bytes):
        answer = answer.decode()
    if isinstance(answer, str):
        data = [ln[5:].strip() for ln in answer.splitlines() if ln.startswith("data:")]
        answer = json.loads(data[-1] if data else answer)
        if isinstance(answer, str):
            answer = json.loads(answer)
    if isinstance(answer, tuple):  # (commands, reason) style
        answer = answer[0]
    if not isinstance(answer, list):
        raise ValueError(f"expected a list of commands, got {type(answer).__name__}")
    return answer


class ModuleTeam:
    """A Python function, called once per player like any built-in team."""

    def __init__(self, name: str, target: str):
        path, func = (target, "commands") if target.endswith(".py") else target.rsplit(":", 1)
        path = Path(path).resolve()
        if not path.is_file():
            raise SystemExit(f"{name}: no such file {path}")
        sys.path.insert(0, str(path.parent))  # so its own sibling imports work
        spec = importlib.util.spec_from_file_location(f"external_{path.stem}_{id(self)}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses and pickling look modules up here
        spec.loader.exec_module(module)
        self.name = name
        self._fn = getattr(module, func, None)
        if not callable(self._fn):
            raise SystemExit(f"{name}: {path.name} has no function {func}()")

    def commands(self, game_state, team_id, player_id):
        return commands_from(self._fn(game_state, team_id, player_id))


class _RemoteTeam:
    """Asks all five agents at once on a team's first call of a turn, then answers from that."""

    def __init__(self, name: str, targets: list[str], deadline: float):
        self.name = name
        self.targets = targets * 5 if len(targets) == 1 else targets
        self.deadline = deadline
        self.calls = self.late = self.errors = 0
        self.slowest_ms = 0.0
        self.last_error = ""
        self._turn = None
        self._answers: dict = {}

    def commands(self, game_state, team_id, player_id):
        turn = (game_state.get("tick"), game_state.get("gameTime"), team_id)
        if turn != self._turn:
            self._turn, self._answers = turn, self._ask_all(game_state, team_id)
        return self._answers.get(player_id, [])

    def summary(self) -> str:
        limit = f">{self.deadline:g}s" if self.deadline else "no deadline"
        text = (f"{self.name}: {self.calls} calls, {self.late} late ({limit}), "
                f"{self.errors} unreadable, slowest answer {self.slowest_ms:.0f} ms")
        return text + (f"; last error {self.last_error}" if self.last_error else "")

    def _ask_all(self, game_state, team_id) -> dict:
        futures = {}
        for pid in range(5):
            body = json.dumps({"prompt": json.dumps(
                {"gameState": game_state, "teamId": team_id, "myPlayers": [pid]})}).encode()
            futures[pid] = _POOL.submit(self._timed, pid, body)
        done, _ = wait(futures.values(), timeout=self.deadline or None)
        answers = {}
        for pid, f in futures.items():
            self.calls += 1
            if f not in done:
                self.late += 1
                continue
            try:
                answer, ms = f.result()
                self.slowest_ms = max(self.slowest_ms, ms)
                answers[pid] = commands_from(answer)
            except Exception as e:
                self.errors += 1
                self.last_error = f"P{pid}: {type(e).__name__}: {e}"[:160]
        return answers

    def _timed(self, pid: int, body: bytes):
        t0 = time.perf_counter()
        answer = self._call(pid, body)
        return answer, (time.perf_counter() - t0) * 1000

    def _call(self, pid: int, body: bytes):
        raise NotImplementedError


class HttpTeam(_RemoteTeam):
    def _call(self, pid, body):
        url = self.targets[pid].rstrip("/")
        if not url.endswith("/invocations"):
            url += "/invocations"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.deadline + 5 if self.deadline else 120) as r:
            return r.read()


class AgentCoreTeam(_RemoteTeam):
    def __init__(self, name, arns, deadline, client_for=None):
        super().__init__(name, arns, deadline)
        self._client_for = client_for or _boto3_client_factory(deadline)
        self._clients: dict = {}
        # One session per player per match, as each match starts with a fresh agent memory.
        match = uuid.uuid4().hex
        self._sessions = [f"sim-{match}-p{pid}" for pid in range(5)]
        for arn in set(self.targets):
            region = arn.split(":")[3]
            if region not in self._clients:
                self._clients[region] = self._client_for(region)

    def _call(self, pid, body):
        arn = self.targets[pid]
        resp = self._clients[arn.split(":")[3]].invoke_agent_runtime(
            agentRuntimeArn=arn, runtimeSessionId=self._sessions[pid], payload=body,
            contentType="application/json", accept="application/json")
        return resp["response"].read()


def _boto3_client_factory(deadline: float):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise SystemExit("AgentCore teams need boto3: uv run --with boto3 python3 "
                         "simulator/run_match.py ...") from None
    config = Config(read_timeout=deadline + 5 if deadline else 120,
                    retries={"max_attempts": 1}, max_pool_connections=10)
    return lambda region: boto3.client("bedrock-agentcore", region_name=region, config=config)
