# kak1 — test team

Built from the balanced sample's folder layout and deploy scripts. The logic is ours.

## How a player decides each tick

1. **Rules in code** (`kaklib/tactics.py`) work out a safe command. Shooting in range and the keeper's distribution are final; no model call.
2. Otherwise **one model call** (Nova Micro) gets pre-computed facts plus the rule suggestion and may pick something better. Timeout 1.5 s (tick ~2 s, server limit 5 s).
3. The reply is **checked** (command type, every parameter, player/team id, has-the-ball). Late or invalid → the rule command is sent.

What the rules do:
- **Score/clock modes:** `chase` (losing late) pushes DEF up and shoots from further; `protect` (winning late) drops forwards back and stops long shots.
- **Opponent read** from each snapshot: `counter` (3+ of them in our half), `patient` (they sit deep), `direct` (2+ pressing our carrier), long shots when their keeper is off his line.
- **Long shots** on by default (FWD 40, MID 38, DEF 25). `LONG_SHOTS=0` turns them off.
- **Possession fix:** `agentId_N` exists on both teams; we pick the holder nearest the ball. Also reads `hasBall`.
- **Coach override:** a message containing `kak1 STYLE counter` (or `direct` / `patient` / `normal` / `auto`). The `kak1` tag is needed because all 10 players hear every message.

## Files

| Path | What |
|---|---|
| `ai-*/src/main.py` | One per player: id, model, timeout. |
| `kaklib/tactics.py` | Rules, modes, opponent read. Tunables at the top. |
| `kaklib/fast_agent.py` | Model call, timeout + cancel, validation, one JSON log line per decision. |
| `kaklib/test_agent_path.py` | 18 checks: possession formats, validation, timeouts, player selection. |
| `ai-*/test_local.py` | Per-player check. Offline by default; `--llm` calls Bedrock. |
| `sim_team.py` | Adapter for the local simulator (not deployed). |

`deploy-all.sh` / `deploy_all.py` copy `../lib` and then `kaklib/` on top into each agent. Agent names are `kak1_<pos>_agent`, so kak1 doesn't overwrite other teams in the same account.

## Run

```bash
cd agentic-football-sample-agents && source .venv/bin/activate && cd ai-team-kak1
python3 kaklib/test_agent_path.py
python3 ai-mid/test_local.py            # offline
python3 ai-mid/test_local.py --llm      # real Bedrock
```

Deploy (Python 3.10 environment, as the workshop requires):

```bash
cd agentic-football-sample-agents && source .venv310/bin/activate && cd ai-team-kak1
./deploy-all.sh            # or ./deploy-all.sh ai-gk
```

Then paste the 5 `kak1_*` Runtime ARNs into My Team.

## Reading the logs

Each decision logs `{"pid","src","ms","cmd","tick","raw"}`. `src`: `rule`, `llm`, `fallback_timeout`, `fallback_invalid`, `fallback_busy`, `fallback_error:*`. The first call per agent also logs `payload_check` (the real payload's keys).
