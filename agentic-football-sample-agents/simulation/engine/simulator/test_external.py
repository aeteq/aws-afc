"""Tests for fielding outside teams in the simulator (simulator/external.py) — no AWS needed.

    python3 simulator/test_external.py

The HTTP tests serve examples/my_team.py from a server in this process, so an outside team
can be checked against the built-in "sample" it copies: same seed, same match, exactly.
"""

import io
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, os.path.join(SIM_DIR, "..", "lib"))
sys.path.insert(0, os.path.join(SIM_DIR, "..", "examples"))

import my_team  # noqa: E402
from engine import Match  # noqa: E402
from external import AgentCoreTeam, make_external, parse_spec  # noqa: E402
from teams import make  # noqa: E402

MY_TEAM_PY = os.path.abspath(os.path.join(SIM_DIR, "..", "examples", "my_team.py"))


def _decide(prompt_data):
    return my_team.commands(prompt_data["gameState"], prompt_data["teamId"], prompt_data["myPlayers"][0])


def _sse(commands) -> bytes:
    # What a deployed agent sends: an async-generator entrypoint yielding json.dumps(commands)
    # is served as server-sent events, so the list arrives JSON-encoded twice.
    return f"data: {json.dumps(json.dumps(commands))}\n\n".encode()


class _Agent:
    """An agent on localhost. respond(prompt_data) -> (body bytes, delay seconds)."""

    def __init__(self, respond):
        outer = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt_data = json.loads(body["prompt"])
                outer.requests.append((self.path, prompt_data))
                reply, delay = respond(prompt_data)
                time.sleep(delay)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(reply)
                except OSError:
                    pass  # the caller gave up on a late answer

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _same_match(a, b):
    assert (a.score, a.stats, a.events) == (b.score, b.stats, b.events), (a.score, b.score)


def test_spec_parsing():
    name, kind, targets = parse_spec("http://h:8080-8084")
    assert kind == "http" and targets == [f"http://h:{p}" for p in range(8080, 8085)], targets
    assert parse_spec("rivals=http://h:9000") == ("rivals", "http", ["http://h:9000"])
    assert parse_spec(f"py:{MY_TEAM_PY}:commands")[1:] == ("py", [f"{MY_TEAM_PY}:commands"])
    arns = [f"arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/p{i}" for i in range(5)]
    assert parse_spec(",".join(arns)) == ("agentcore", "agentcore", arns)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# their deployed team\n" + "\n".join(arns) + "\n")
    try:
        assert parse_spec(f"them=@{f.name}") == ("them", "agentcore", arns)
    finally:
        os.unlink(f.name)
    for bad in ("http://a:1,http://a:2", "ftp://x", "nonsense", "http://a:1,arn:aws:x"):
        try:
            parse_spec(bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad!r}")
    print("  names, port ranges, @files and ARN lists parse; 2 URLs, mixed or unknown specs are refused")


def test_python_function_plays_exactly_like_builtin():
    ext = Match(make(f"py:{MY_TEAM_PY}"), make("aggressive"), seed=3).run()
    _same_match(ext, Match(make("sample"), make("aggressive"), seed=3).run())
    print(f"  py:examples/my_team.py == sample, {ext.score[0]}-{ext.score[1]} vs aggressive")


def test_http_team_plays_exactly_like_builtin_from_both_ends():
    with _Agent(lambda d: (_sse(_decide(d)), 0.0)) as agent:
        home = Match(make(agent.url), make("aggressive"), seed=5, seconds=90).run()
        away = Match(make("aggressive"), make(agent.url), seed=5, seconds=90).run()
        requests = list(agent.requests)
    _same_match(home, Match(make("sample"), make("aggressive"), seed=5, seconds=90).run())
    _same_match(away, Match(make("aggressive"), make("sample"), seed=5, seconds=90).run())
    path, first = requests[0]
    assert path == "/invocations" and set(first) == {"gameState", "teamId", "myPlayers"}, requests[0]
    assert {d["teamId"] for _, d in requests} == {0, 1}
    print(f"  over HTTP, HOME and AWAY: identical to sample ({len(requests)} requests)")


def test_late_answers_are_dropped():
    def respond(d):
        return _sse(_decide(d)), (0.5 if d["myPlayers"][0] == 2 else 0.0)

    with _Agent(respond) as agent:
        team = make_external(agent.url, deadline=0.2)
        Match(team, make("idle"), seed=1, seconds=4.0).run()  # 2 decisions
    assert (team.calls, team.late, team.errors) == (10, 2, 0), team.summary()
    print(f"  {team.summary()}")


def test_unreadable_answers_do_not_stop_the_match():
    def respond(d):
        pid = d["myPlayers"][0]
        if pid == 1:
            return b"I think the defender should press", 0.0
        if pid == 2:
            return json.dumps(_decide(d)).encode(), 0.0  # plain JSON, not server-sent events
        return _sse(_decide(d)), 0.0

    with _Agent(respond) as agent:
        team = make_external(agent.url)
        r = Match(team, make("idle"), seed=1, seconds=4.0).run()
    assert (team.calls, team.errors) == (10, 2) and r.stats[0].rejected_commands == 0, team.summary()
    assert "P1" in team.last_error, team.last_error
    print(f"  {team.summary()}")


class _FakeAgentCore:
    def __init__(self):
        self.calls = []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        prompt_data = json.loads(json.loads(kwargs["payload"])["prompt"])
        return {"response": io.BytesIO(_sse(_decide(prompt_data))), "contentType": "text/event-stream"}


def test_agentcore_team_invokes_each_runtime():
    arns = [f"arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/rival_p{i}" for i in range(5)]
    fake = _FakeAgentCore()
    team = AgentCoreTeam("rivals", arns, deadline=1.0, client_for=lambda region: fake)
    r = Match(team, make("aggressive"), seed=3, seconds=60).run()
    _same_match(r, Match(make("sample"), make("aggressive"), seed=3, seconds=60).run())
    by_arn = {}
    for c in fake.calls:
        pid = json.loads(json.loads(c["payload"])["prompt"])["myPlayers"][0]
        by_arn.setdefault(c["agentRuntimeArn"], set()).add(pid)
    assert by_arn == {arn: {i} for i, arn in enumerate(arns)}, by_arn
    sessions = {c["runtimeSessionId"] for c in fake.calls}
    assert len(sessions) == 5 and all(len(s) >= 33 for s in sessions), sessions
    print(f"  each player's ARN called for that player only, {len(fake.calls)} calls, 5 sessions")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"=== {t.__name__} ===")
        t()
    print(f"\nAll {len(tests)} external-team tests passed")
