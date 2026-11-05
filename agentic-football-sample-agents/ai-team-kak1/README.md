# kak1 — test team

Built from the balanced sample's folder layout and deploy scripts. The logic is ours.

## How a player decides each tick

1. **Rules in code** (`kaklib/tactics.py`) work out a safe command. Clear shots and everything the keeper does are final; no model call.
2. Otherwise **one model call** (Nova Micro) gets pre-computed facts plus the rule suggestion and may pick something better. Timeout 1.5 s (tick ~2 s, server limit 5 s).
3. The reply is **checked** (command type, every parameter, player/team id, has-the-ball). Late or invalid → the rule command is sent.

Formation **1-2-1**: GK (0), DEF (1), MID1 (2, left), MID2 (3, right), FWD (4, single forward).
Roles come from the player id, not the folder: `ai-fwd1` plays MID2 and `ai-fwd2` plays FWD.
Set the same formation on the portal's My Team page.

The rules are our coach's instructions (the model reads the same text in `fast_agent.py`):
- **Team:** never leave our goal unprotected; under pressure a safe pass beats a dribble; take
  clear shots only (close, centred, no defender in the way). One player per loose ball.
- **GK:** never leaves the area; stands on the ball–goal line; passes at once to the nearest
  unmarked mate, preferring the DEF; steps out to narrow the angle on a runner clear through.
- **DEF:** never crosses halfway; goal-side of the most dangerous opponent; tackles only within
  reach and goal-side; passes to a free midfielder, else back to the keeper.
- **MID1/MID2:** one channel each; through balls into space for the FWD first, the other
  midfielder second; otherwise recycle to the DEF; only the nearer one presses in the middle
  third, the other covers; both drop to cover when our half is outnumbered.
- **FWD:** never drops behind halfway; sits between their two deepest defenders; shoots clear
  chances away from the keeper; when blocked, lays off to a midfielder.
- **Hard zones** (GK area, DEF own half, FWD their half) are enforced on the model's answers
  too: a breaking answer is logged as `fallback_zone` and the rule command is sent.
- **Score/clock modes:** `chase` (losing late) shoots from a little further and sprints more.
- **Possession fix:** `agentId_N` exists on both teams; we pick the holder nearest the ball.
- **Coach messages** reach the model only if they contain `kak1` (all 10 players hear them).

## Files

| Path | What |
|---|---|
| `ai-*/src/main.py` | One per player: id, model, timeout. |
| `kaklib/tactics.py` | Coach rules, hard zones, score/clock modes. Tunables at the top. |
| `kaklib/fast_agent.py` | Model call, timeout + cancel, validation, one JSON log line per decision. |
| `kaklib/test_agent_path.py` | 49 checks: possession formats, validation, coach rules, hard zones, timeouts, player selection. |
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

Each decision logs `{"pid","src","ms","cmd","tick","params","t","me","ball","owner","team","score","raw"}` (positions are rounded, `me`/`ball` as [x, y]). `src`: `rule`, `llm`, `fallback_timeout`, `fallback_invalid`, `fallback_zone`, `override_<command>` (press/shoot policy), `fallback_busy`, `fallback_error:*`. The first call per agent also logs `payload_check` (the real payload's keys).
