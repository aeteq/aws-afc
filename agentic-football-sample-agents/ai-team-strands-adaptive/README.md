# Adaptive team

Five agents based on the Balanced deployment layout, with a conservative goalkeeper,
covering defender, linking midfielder, striker and supporting forward. Existing teams
are unchanged. Project name: `TeamStrandsAdaptive` (a separate CloudFormation stack).

## Decisions

`../lib/adaptive_tactics.py` computes the same assignments independently from each
snapshot. Exactly one eligible outfield player presses; the defender stays back unless
the ball threatens the defensive third. The forwards occupy separate lanes. Low
stamina disables sprinting. Positions mirror for home and away.

On the ball, candidates are ranked: a short, reasonably central, unobstructed shot;
safe passes ranked by lane clearance, receiver pressure and progression; then movement.
The goalkeeper stays deep and distributes only to available passing targets.
These are tunable heuristics, not calibrated probabilities. Shot aim currently uses
TR; goalkeeper-aware aim selection needs verification of the engine's aim semantics.

`../lib/adaptive_runtime.py` asks the existing position-specific Nova model to select
an index only when multiple choices exist. Off-ball assignments execute directly.
Invalid model output or model exceptions use candidate zero. Each request uses a
fresh conversation. There is no memory service or gateway dependency. Logs record
command, player, source (rules/model/fallback/last_resort), and elapsed milliseconds.

Possession must identify one player unambiguously. Both legacy and agentId state
formats are supported; duplicated agent IDs need `possessionTeamId` to disambiguate.
An explicitly free ball never permits a possession action. Invalid team IDs emit no
commands. Malformed state falls back to a conservative stance.

## Configuration

Set these environment variables on each deployed runtime (or locally):

- `ADAPTIVE_MATCH_SECONDS`: actual full match duration in the same seconds used by
  `gameTime`. Default `0` disables late-game adaptation. In the last quarter, a lead
  shifts positions back; a deficit pushes outfield support forward.
- `ADAPTIVE_RULES_ONLY=1`: skip model selection for a deterministic comparison team.

The workshop reference is absent from this checkout. Confirm match duration, time
semantics, field coordinates and command behavior against the server before tuning.
No response-deadline guarantee is implemented: model/network timeout behavior must
be measured against the actual server deadline before competitive deployment.

## Verify and deploy

From this directory:

```sh
python3 test_tactics.py
AWS_DEFAULT_REGION=us-east-1 python3 deploy_all.py
```

Deployment prerequisites and model access are the same as the Balanced team. The
Python deployment script stages the shared library in all five runtime packages.
`aws-targets.json` starts empty and is populated from the active AWS identity at deploy.
The models remain Micro (GK/FWD1), Lite (DEF/FWD2), Pro (MID).

## Evaluate

No competitive strength is claimed until live matches have been played. Compare
against all five existing teams, swapping home/away and repeating each pairing.
Keep match settings fixed. Record wins, draws, goal difference, p50/p95 latency,
model fallback frequency, and server-rejected or late commands. Compare rules-only
with model selection first; then tune one heuristic at a time. Local tests verify
coordination and command safety, not engine acceptance or match performance.
