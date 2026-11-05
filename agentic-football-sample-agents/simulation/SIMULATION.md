# Local simulation

Play matches on your laptop between our teams, the AWS sample teams and built-in opponents. Use it to catch crashes and compare two versions of a team before deploying. **Don't use it to predict live results**: the physics are guesses and the pitch differs from the live one (see Limits).

## Layout

| Path | What |
|---|---|
| `engine/` | Unofficial match engine ([wormym1011/agentic-football-simulator](https://github.com/wormym1011/agentic-football-simulator), MIT-0), copied in. Standard library only. |
| `teams/sample_fallback.py` | The AWS sample teams without the model (their shared `lib/fallback.py`). |
| `live/start_team.sh` | Runs a team's 5 agents as local servers that call real Bedrock. |
| `live/stop_all.sh` | Stops those servers. |
| `live/serve.py` | Used by `start_team.sh` to run one agent's `src/main.py`. |

## Setup (once)

Python 3.10+. From `agentic-football-sample-agents/`:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install strands-agents bedrock-agentcore bedrock-agentcore-starter-toolkit boto3 pydantic
```

The engine itself needs no packages. The venv is for running our agents and the AWS samples.

All commands below run from `simulation/engine/` with the venv active.

## 1. Rules only (free, seconds)

Our team with the model switched off, against every built-in opponent, home and away:

```bash
python simulator/run_match.py --team kak1=py:../../ai-team-kak1/sim_team.py --matches 6
```

The last line is the total, e.g. `vs others: 44W 4D 0L, goals 198-3`. Built-in opponents: `sample`, `aggressive`, `defensive`, `idle`, `box`, `diamond`, `pyramid`, `3-1`.

Against the AWS samples' rules:

```bash
python simulator/run_match.py --team kak1=py:../../ai-team-kak1/sim_team.py \
  --opponent aws=py:../teams/sample_fallback.py --matches 6
```

## 2. A/B test a change

Copy the team folder (or check out the old commit somewhere else), then play the two against each other:

```bash
python simulator/run_match.py --team new=py:../../ai-team-kak1/sim_team.py \
  --opponent old=py:/path/to/old/ai-team-kak1/sim_team.py --matches 8
```

Matches are often 0-0 or 1-0, so judge on 16+ matches, not 2.

Quicker: `bench.py` does this for you against a git ref, with no checkout. From `simulation/`:

```bash
python3 bench.py                      # ai-team-kak1 in the working tree vs origin/main
python3 bench.py --base HEAD~3 --seeds 40
```

It plays both versions on the same seeds, home and away, against every built-in opponent, plus head to head, and prints W/D/L, points and goal difference per match with a 95% interval. Exit code 1 means a regression. Run it before every push that touches `ai-team-kak1/`. Play both sides: the engine moves home players first, which gives the away team an edge in every chase (kak1 vs itself: the away side wins).

## 3. Watch a replay

```bash
python simulator/run_match.py --home kak1=py:../../ai-team-kak1/sim_team.py --away aggressive --replay match.html
open match.html
```

`--seed N` replays a different match. The summary lists every goal with its time and scorer, or "own goal".

## 4. With the real model (costs Bedrock calls)

Each team's 5 agents run as local servers and are asked at once each tick, like live. Needs AWS credentials (e.g. `export AWS_PROFILE=<profile>`).

```bash
cd ../live
./start_team.sh kak1 9100 ../../ai-team-kak1
./start_team.sh balanced 9200 ../../ai-team-strands-balanced     # or -extremely-aggressive / -extremely-defensive
cd ../engine
python simulator/run_match.py --home kak1=http://127.0.0.1:9100-9104 \
  --away balanced=http://127.0.0.1:9200-9204 --deadline 5 --replay llm.html
../live/stop_all.sh
```

- A match takes 2–5 minutes and makes ~750 model calls per team.
- The summary shows calls, late answers and the slowest answer per team.
- Agent logs are in `live/logs/<name>-<agent>.log`. kak1 writes one JSON line per decision: `src` (`rule`, `llm`, `fallback_*`, `override_*`), the command, and positions.
- `start_team.sh` sets `PITCH_HALF_X=55` and `MATCH_SECONDS=300` for kak1, to match the simulator (see below).
- The memory and gateway sample teams need AWS resources (`MEMORY_ID`, `GATEWAY_URL`) and don't run here.

## 5. Engine tests

```bash
python simulator/test_engine.py && python simulator/test_external.py
```

## Limits — read before trusting a result

- **Pitch size.** The engine uses the documented pitch (x −55..55, y −35..35). The live pitch is ~8× smaller in both directions: 132 live turns stayed within x −6.9..6.6 and y −3.5..3.6. kak1 handles this with `PITCH_HALF_X` in `kaklib/tactics.py` (7 when deployed; `PITCH_HALF_Y` follows the same ratio, ~4.45). The simulator runs kak1 with 55 (set by `sim_team.py` and `start_team.sh`). Anything tied to the live pitch size can't be tested here.
- **Match length.** The engine plays 5 minutes; live matches are 120 s. kak1's `MATCH_SECONDS` is 120 when deployed and 300 in the simulator.
- **Physics are approximations.** No fouls, ball never out, rough shot and pass models. Own goals are far more common here than live.
- **Built-in opponents are not the live AWS teams.** `sample` plays the AWS fallback rules, not the model-driven team.
- **Possession bug in old sample code.** Teams using an old `lib/state.py` may freeze in the engine, which inflates our results against them.
