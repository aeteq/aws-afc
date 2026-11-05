# Agentic Football Simulator (unofficial)

A local match engine for the [AWS Agentic Football Cup](https://agenticfootballcup.com).
AWS doesn't provide a game server you can test against before the event, so this one lets
you play full 5v5 matches with **your own team**, as a Python function, as agents running on
localhost, or as your deployed Bedrock AgentCore runtimes. You can also play another team's
deployed agents, if they let you.

> **Not affiliated with AWS or the organisers.** The official engine is closed source. This
> one follows the published rules and the server's payload format, but **the physics are a
> guess**: speeds, friction, and pass, tackle, shot and save odds are plausible numbers, not
> the server's. Use it to catch mistakes (wrong tags, commands that never play, a keeper who
> never distributes, agents too slow for the deadline) and to compare two versions of your
> team. A win here is not a win on the day.

Python 3.10+, standard library only. `boto3` is needed only to call deployed runtimes.

## Quick start

```bash
git clone https://github.com/wormym1011/agentic-football-simulator
cd agentic-football-simulator

python3 simulator/run_match.py --home sample --away aggressive --replay match.html  # open match.html
python3 simulator/run_match.py --team mine=py:examples/my_team.py --matches 20      # a series, home and away
```

`examples/my_team.py` is a starter team: it plays like the AWS sample teams' rule-based
fallback until you put your own logic in `commands()`.

## Play your team

Wherever a team goes (`--home`, `--away`, `--team`, `--opponent`), give a built-in name or a
**team spec**:

| Spec | Plays |
|---|---|
| `py:FILE.py[:FUNC]` | A function `FUNC(game_state, team_id, player_id)` returning the commands (default `commands`). Runs in this process, so read the code first if it isn't yours |
| `http://HOST:PORT` | Agents over AgentCore's HTTP protocol (`agentcore dev`, or your `main.py` started with `PORT` set). One URL for all five players, five comma-separated URLs (GK, DEF, MID, FWD1, FWD2), or a range: `http://127.0.0.1:8080-8084` |
| `arn:aws:bedrock-agentcore:...` | Deployed runtimes, one ARN or five. Needs `boto3` and credentials allowed to `InvokeAgentRuntime` on them |
| `@FILE` | Any of the above, one entry per line (`#` comments allowed) |
| `NAME=SPEC` | Shows NAME in the results |

```bash
# your five agents running locally
python3 simulator/run_match.py --home mine=http://127.0.0.1:8080-8084 --away sample

# your deployed team against another deployed team, as live: a turn every 2 s, both teams
# asked at once, answers later than the deadline dropped. A 5-minute match takes 5 minutes.
pip install boto3
python3 simulator/run_match.py --home mine=@my-arns.txt --away rivals=@rivals-arns.txt \
    --realtime --deadline 1 --replay live.html
```

An ARN file lists the five runtimes in the order GK, DEF, MID, FWD1, FWD2:

```
# my team
arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/my_gk_agent-XXXXXXXXXX
arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/my_def_agent-XXXXXXXXXX
...
```

Each agent is called the way the game server calls it. The body is
`{"prompt": "{\"gameState\": …, \"teamId\": 0|1, \"myPlayers\": [n]}"}`, and all five players
are asked **in parallel** on the same snapshot. Answers may be server-sent events (what an
async-generator entrypoint sends), plain JSON, or a JSON string of the list. Commands aren't
re-tagged, so a command with the wrong `teamId` or `playerId` is **rejected**, as on the server.
Every result shows each remote team's calls, late answers, unreadable answers and slowest answer.

An LLM-driven team is slow and costs money: a match is 150 turns × 5 calls, billed to whichever
account runs the agents.

## Built-in teams

| Name | What it is |
|---|---|
| `sample` | The rule-based fallback (`lib/fallback.py`) every AWS sample team shares, i.e. what they play whenever their model fails |
| `aggressive`, `defensive` | Rule-based **approximations** of the extremely-aggressive and extremely-defensive sample teams' prompts |
| `box`, `diamond`, `pyramid`, `3-1` | Classic 5-a-side shapes (2-2, 1-2-1, 2-1-1, 3-1), built like `sample` |
| `idle` | Sends nothing; the engine's default AI plays |

`lib/fallback.py` and `lib/state.py` are copied unchanged from
[aws-samples/sample-ai-possibilities](https://github.com/aws-samples/sample-ai-possibilities/tree/main/agentic-football-sample-agents/lib).

## What the official sources say

The organisers' sources don't all agree. The engine's defaults are in the last column; change
them in `simulator/engine.py` or on the command line.

| Rule | Workshop guide | Player portal help | Organisers' blog (June 2026) | This engine |
|---|---|---|---|---|
| Turn interval | ~2 s | ~2 s | 2 s | 2 s |
| Reply deadline | 5 s | **500 ms** (one warning says 2 s) | 1 s, end to end | `--deadline`, default 1 s |
| A late or invalid reply | player keeps its last command | "default engine behaviour" | player IDLEs that tick | player keeps its last order |
| Match length | 5 min | 5 min | 2 min | 5 min (`MATCH_SECONDS`) |
| Commands kept per reply | the first | — | one | the first (`FIRST_COMMAND_ONLY`) |
| Ball out of play | never | — | — | never, it bounces (`BALL_STAYS_IN`) |
| Penalties / offside | none / not listed | — | — | none |

Sources: the AWS workshop *Agentic Football Cup* (Workshop Studio), the help text in the
player portal at agentic-football.aws.dev, and [Inside Agentic Football Cup](https://github.com/strands-agents/harness-sdk/blob/main/site/src/content/blog/inside-agentic-football-cup.mdx)
on the Strands blog. The blog also lists `DRIBBLE`, `MARK`, `INTERCEPT`, `TACKLE`, `CLEAR`,
`IDLE` and goalkeeper actions; the portal scores the match MVP as
*commands × success rate ÷ (1 + average latency in seconds)*, so speed counts too.

**Plan for the 500 ms deadline.** A new AgentCore session's first call (a cold start) took 3–4 s
when we measured; warm calls took ~340 ms from outside AWS. A team that calls an LLM every turn
will struggle to answer in time.

## What it models

- 110 × 70 pitch, coin-toss kick-off, and the team that concedes kicks off. All ten agents are
  called every 2 s on the same snapshot, and commands are applied in a random team order.
- The server's payload format: `tick`, `gameTime`, `playMode`, `score`, `ball` (position,
  velocity, `isFree`, `possessionAgentId`), `players` (`agentId`, `teamCode`, position,
  velocity, stamina 0–100, `isSprinting`) and `teamChat`. HOME is listed first.
- HOME (team 0) attacks +x, AWAY (team 1) attacks −x.
- Every sample-team command. `PASS`/`SHOOT` from a player without the ball is **wasted**.
- Stamina (sprinting drains it, tired players are slower), dribbling, passes that lead the
  receiver with noise that grows with distance and pressure, lofted balls, tackles (likelier
  when pressing), shots saved according to the keeper's reaction time, and blocks.
- A keeper who wins the ball in his own box holds it and can't be tackled; from a back-pass he
  must play it with his feet.
- **Not modelled:** fouls, cards, injuries and skill ratings. Pressing costs only stamina here,
  so swarming is probably stronger than it is live.

## Tests

```bash
python3 simulator/test_engine.py
python3 simulator/test_external.py   # local HTTP agents and a fake AgentCore client; no AWS
```

## Files

```
simulator/engine.py      the match engine (physics, commands, payload, stats)
simulator/run_match.py   series / single match / replay CLI
simulator/external.py    team specs: Python functions, HTTP agents, AgentCore ARNs
simulator/teams.py       built-in teams
simulator/opponents.py   formation teams
simulator/replay.py      self-contained HTML replay viewer
examples/my_team.py      starter team
lib/                     the AWS sample agents' shared rule-based code (unchanged)
```

## Contributing

Seen something on the real server that this engine gets wrong (timings, the payload, how a
command behaves)? Open an issue with what you saw. That's the fastest way to make this closer
to the real game for everyone.

## License

MIT No Attribution (MIT-0), see [LICENSE](LICENSE). The files in `lib/` are © Amazon.com,
Inc. or its affiliates, from aws-samples/sample-ai-possibilities, also under MIT-0.
