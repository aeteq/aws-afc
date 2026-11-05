# Agentic Football Cup — Complete Workshop Reference

Nov 5, 2026 · @Constantinos

## Overview

You build 5 AI agents (one per player), deploy them to Amazon Bedrock AgentCore in your own AWS account, and they play live 5v5 matches against AWS reference teams or other teams in the room. Every decision on the pitch comes from your agents' logic; the match server, agent loop and Player Portal are run by the workshop.

| Item | Detail |
| --- | --- |
| Level / duration | 200, about 4 hours, no AI experience required |
| Region | us-east-1; temporary AWS accounts provided free at the event |
| Phase 1 — Build | Get sample agents, test locally, deploy 5 agents to AgentCore |
| Phase 2 — Play | Log in to the Player Portal, register 5 ARNs, practice, then tournament |
| Phase 3 — Supercharge | Prompt engineering, Memory, Gateway, Observability, multi-agent coordination |
| AI opponents | Bedrock Ballers / The Benchmark FC (balanced), Total Attack United (extremely aggressive), Fort Knox Athletic (extremely defensive) |

### AWS stack

| Service | Role in the workshop |
| --- | --- |
| Amazon Bedrock | Foundation models (default Amazon Nova Micro) that reason about game state |
| Bedrock AgentCore Runtime | Serverless runtime hosting your 5 agents |
| Bedrock AgentCore Harness | Config-only agent loop (model + instructions + tools), no code needed; fully supported alternative to Runtime |
| Strands Agents SDK | Open-source Python framework the sample agents use |
| Amazon CloudWatch | Logs, traces and metrics for each agent |
| AWS IAM | Agent permissions; cross-account role lets the platform invoke your agents |
| Kiro | Optional agentic IDE; Claude Code, Cursor or similar also work |

### What happens each tick

1. About every 2 seconds the workshop's Agent Loop sends the current game state to each of your 5 agents separately.
2. Each agent (optionally calling Bedrock) returns one command for its own player.
3. The Match Server (physics and rules) applies the commands and the game continues.
4. The Player Portal shows the match in 3D and reads AgentCore building blocks and evaluations from your account through a cross-account IAM role.

## Football basics

Football is simple rules with very complex play: two teams, one ball, score in the other team's net. The workshop frames it as the ultimate multi-agent coordination problem. Real matches are 90 minutes and 11v11; this game is 5v5.

| Role | Job | Think of it as |
| --- | --- | --- |
| Goalkeeper | Only player allowed to use hands; last line of defence | The safety net |
| Defenders | Stop the opponent scoring, win the ball back | The shield |
| Midfielders | Control tempo, link defence and attack, create chances | The brain |
| Forwards | Score, make runs, finish chances | The weapon |

| Zone | Guidance |
| --- | --- |
| Defensive third | Protect your goal; avoid risks |
| Midfield | The battleground; whoever controls it usually controls the match |
| Attacking third | Where chances are created and goals scored |
| Penalty box | High-stakes area (no penalty kicks in this game) |

**Principles of good teams:** spacing (don't crowd each other), width (stretch the defence), depth (passing options at different distances), and positioning (be where the ball is going, not where it is).

**What each agent must do:** perceive the pitch, decide (pass, shoot, move or defend), coordinate as a team, and adapt to the opponent rather than follow a script.

## Game mechanics and rules

Matches are 5v5 and last 5 minutes, fully automated from kick-off to final whistle; each agent gets about one call every 2 seconds and has 5 seconds to answer.

| Parameter | Value |
| --- | --- |
| Players | 5 per side; playerId 0–4, 0 is always the GK |
| Match length | 5 minutes (about 64 ticks; memory docs cite 5 agents × \~64 ticks ≈ 320 events) |
| Invocation rate | About 1 call every 2 seconds per agent; each agent is called separately |
| Decision timeout | 5 seconds; no answer means the player keeps its last command |
| Pitch | x from about −55 to +55, y from about −35 to +35 |
| Direction | Team 0 (HOME) defends −x, attacks +x; Team 1 (AWAY) defends +x, attacks −x |
| Visibility | Agents see only what a player on the pitch could see; opponent code and strategy are hidden |

### Scoring and restarts

- A goal counts when the ball fully crosses the goal line between the posts.
- After a goal, play restarts from the centre with a kickoff, all players reset to their starting positions, and the team that conceded kicks off.
- Tip from the workshop: recognise kickoff (`playMode` = KICK\_OFF); the seconds after a goal are a window to set your shape before the opponent is organised.

### Continuous play

- The ball never goes out of bounds; there are no throw-ins, corners, goal kicks or penalty kicks.
- Possession changes only through tackles, interceptions and goals, so focus on ball control and positioning, not set pieces.
- There are no natural stoppages, so stamina and positioning must be managed without breaks.

### Trade-offs the rules create

- Pressing too hard risks fouls and cards; not pressing lets the opponent keep the ball. FREE\_KICK exists as a play mode, but the pages give no foul or card rules.
- Aggressive positioning creates chances but exposes you to counter-attacks.
- The best agents read score, time remaining, cards and player count, and adapt.

## Actions (commands)

Each agent returns a JSON array with exactly one command for its own player. One-shot commands use `duration` 0; maintained commands persist for `duration` ticks; tactical commands stay active until cleared.

| Group | Command | Parameters | Notes |
| --- | --- | --- | --- |
| Movement | MOVE\_TO | target\_x (float), target\_y (float), sprint (bool) | Sprint is faster but drains stamina |
| Movement | FOLLOW\_PLAYER | target\_player\_id (int), target\_team (HOME/AWAY), distance (float) | Maintained; loose marking or tracking a run |
| Attacking | SHOOT | aim\_location (TL, TR, BL, BR, CENTER), power (0.0–1.0) | Needs the ball |
| Attacking | PASS | target\_player\_id (int), type (GROUND, AERIAL, THROUGH) | Needs the ball; THROUGH plays into space ahead of the target |
| Attacking | GK\_DISTRIBUTE | target\_player\_id (int), method (THROW, KICK) | GK only |
| Defensive | PRESS\_BALL | intensity (0.0–1.0) | Maintained; above 0.3 attempts tackles, above 0.5 sprints |
| Defensive | MARK | target\_player\_id (int), tightness (LOOSE, TIGHT) | Maintained man-marking |
| Defensive | INTERCEPT | aggressive (bool) | Maintained; positions to cut out a pass |
| Defensive | SLIDE\_TACKLE | target\_player\_id (int), sprint (bool), distance (float) | One-shot, risky; listed only in the sample prompts |
| Tactical | SET\_STANCE | stance (0 Balanced, 1 Attacking, 2 Defensive) | Stays active until cleared |
| Tactical | CLEAR\_OVERRIDE | none | Player returns to default AI behaviour |
| Tactical | RESET | none | Clears all overrides for the whole team |

### Response format

```json
[{"commandType": "MOVE_TO", "playerId": 0, "parameters": {"target_x": -49.0, "target_y": 2.0, "sprint": false}, "duration": 0}]
```

More examples from the sample prompts:

- `[{"commandType":"GK_DISTRIBUTE","playerId":0,"parameters":{"target_player_id":1,"method":"THROW"},"duration":0}]`
- `[{"commandType":"INTERCEPT","playerId":0,"parameters":{"aggressive":true},"duration":3}]`

The shared library forces the correct playerId and extracts JSON even when the model adds text, but prompts still say "Return ONLY the JSON array, no text before or after."

## Stats and game state

Every tick each agent receives the full pitch snapshot plus which player it controls; per-player stats are stamina, speed, sprinting state and current or last action.

```json
{
  "gameState": {
    "tick": 42,
    "gameTime": 84.0,
    "playMode": "OPEN_PLAY",
    "score": { "home": 1, "away": 0 },
    "ball": {
      "position": { "x": 12.5, "y": -3.0, "z": 0.0 },
      "velocity": { "x": 1.2, "y": 0.0, "z": 0.0 },
      "isFree": false,
      "possessionAgentId": "agentId_2"
    },
    "players": [ ... ],
    "teamChat": []
  },
  "teamId": 0,
  "myPlayers": [2]
}
```

| Field | Meaning |
| --- | --- |
| tick, gameTime | Tick counter and elapsed seconds |
| playMode | KICK\_OFF, OPEN\_PLAY or FREE\_KICK |
| score | home and away goals |
| ball | position (x, y, z), velocity, isFree, possessionAgentId |
| teamChat | Latest coach instructions from the Player Portal |
| teamId | 0 = HOME, 1 = AWAY |
| myPlayers | Index (0–4) of the one player this call controls |

### Per-player fields (`players[]`)

| Field | Type | Meaning |
| --- | --- | --- |
| agentId | string | e.g. agentId\_0; index 0 is always the GK |
| teamCode | string | home or away |
| position | {x, y} | Location on the pitch |
| velocity | {x, y} | Movement vector |
| orientation | float | Facing direction in degrees |
| stamina | float | 0–100; sprinting drains it, walking conserves it |
| speed | float | Current movement speed |
| isSprinting | bool | Whether the player is sprinting |
| currentAction | string | Active action state, e.g. IDLE |
| lastAction | string | Last command type executed |

The sample code's `lib/state.py` turns this JSON into a short text summary for the LLM, which keeps prompts small and fast.

## Strategies and tactics

The workshop's advice is to blend styles: start with possession, counter when the opponent pushes up, press in short bursts, and go compact to protect a lead.

### Attacking strategies

| Strategy | Principles | Agent behaviour |
| --- | --- | --- |
| Possession ("make them chase") | Short passing, patient build-up, rotation, more players near the ball | PASS type GROUND to nearby teammates; hold formation and recycle; cluster support round the carrier |
| Counter-attack ("strike like lightning") | Defend deep, win the ball, go forward fast, skip midfield | Low compact line; timed tackles and interceptions; sprint forward on winning the ball; PASS type AERIAL or THROUGH to forwards |
| High press ("suffocate them") | Immediate pressure, press triggers, whole team presses, high line | Move at the carrier and close passing lanes; press harder when the opponent faces their own goal; defenders sit in the middle third. Burns stamina: press in bursts |
| Wing play ("stretch and cross") | Width, overlapping runs, crosses, target player | Wide agents near the sidelines; coordinate 2v1 overlaps; PASS type AERIAL from wide attacking spots; one forward central |

### Defensive strategies

| Strategy | Principles | Agent behaviour |
| --- | --- | --- |
| Zonal | Assigned zones, keep shape, hand attackers on, cover depth | Position by zone coordinates; only engage opponents in your zone; stagger depth |
| Man-to-man | Assign targets, stay tight, deny the ball, switch when needed | MARK the assigned opponent; mirror them; stand between mark and ball; reassign marks as play changes |
| Compact | Narrow shape, short lines, block the centre, delay | Tight horizontal band; small gap between lines; force play wide; jockey rather than dive in |

### Decision with the ball

1. Near goal with a clear shot: SHOOT.
2. Teammate in a better position: PASS.
3. Space ahead and no pressure: MOVE\_TO (advance with the ball).
4. Under pressure with no options: PASS back and reset.

Weigh risk against reward using the game state: a 40-yard shot is wasteful at 2-0 up but worth it when losing in the last minute.

### Off the ball

Four of five agents are without the ball at any moment. They should make runs into space, pull defenders away to open lanes, offer a safe nearby passing option, or recover into defensive shape after losing the ball.

### Adjusting to the score

| Situation | Adjustment |
| --- | --- |
| Winning comfortably | Slow down, keep possession, no unnecessary risks |
| Winning narrowly | Stay organised; protect the lead without sitting too deep |
| Drawing | Balanced; look for chances without overcommitting |
| Losing narrowly | Push players forward, shoot more, accept defensive risk |
| Losing badly | All-out attack, high press, shoot on sight |

### Stamina

- Sprint in bursts for pressing, breakaways and tracking back.
- Rotate who presses so one agent doesn't do all the work.
- Position well so less running is needed; be where the ball is going.
- When ahead, slow the tempo and let the clock run.
- An exhausted agent effectively makes it 4v5.

## Sample teams and AI opponents

All three sample teams use a 1-1-1-2 shape (GK, DEF, MID, FWD1, FWD2) and the same command list and response format; only the role section of each prompt differs. Start with Balanced, then mix prompts across teams to build a hybrid.

| Opponent | Style | What it tests |
| --- | --- | --- |
| The Benchmark FC | Balanced | Baseline for overall performance |
| Total Attack United | Extremely aggressive | Sweeper-keeper, defender joins every attack, forwards camp near goal: exposes weak counter-attacking |
| Fort Knox Athletic | Extremely defensive | Everyone deep, MID as extra defender, few shots: tests whether you can break a low block |

### Role instructions by position

| Position | Balanced | Aggressive | Defensive |
| --- | --- | --- | --- |
| GK (0) | Stay on the line between ball and goal centre; GK\_DISTRIBUTE THROW to nearest teammate; INTERCEPT only loose balls near the box; avoid sprinting | Sweeper-keeper at or past halfway when in possession; KICK long from the defensive third; shoot within \~35 units; press and intercept early; sprint freely | Never leave the line or pass x = −45 (HOME) / +45 (AWAY); THROW to nearest defender; INTERCEPT only within 5 units; never sprint |
| DEF (1) | Shield the GK; MARK the most dangerous opponent; INTERCEPT in the defensive third; press when the carrier enters your zone; SLIDE\_TACKLE as last resort; pass to MID/FWD, don't dribble; hold shape | Join every attack; carry the ball forward; shoot from \~30 units; only forward through balls; press at maximum intensity; only track back if last defender | Never cross halfway; MARK TIGHT; intercept everything in own half; press only in own half; pass immediately; never shoot; SET\_STANCE 2 always |
| MID (2) | Link play; pass forward to well-placed forwards or back under pressure; press in the middle third; move into space; shoot within \~25 units; track back | Second striker; shoot from up to \~35 units; through balls only, never backward; stay near the forwards; never track back; lead the press | Extra defender; never cross halfway; MARK TIGHT; pass only back to DEF/GK; never shoot; SET\_STANCE 2 always |
| FWD1 (3) | Score; shoot within \~25 units; runs into space for through balls; high press; stay left/centre; sprint on attacking runs only | Pure goal scorer; shoot from up to \~40 units at power 1.0; camp in the box; never track back past halfway; one-twos with FWD2 | Forward in name only; stay near halfway; MARK their DEF/MID; shoot only inside the box (\~15 units); pass back; SET\_STANCE 2 always |
| FWD2 (4) | Same as FWD1 on the right side | Mirrors FWD1 | Mirrors FWD1 |

### Balanced GK priority list (example)

1. If you have the ball: GK\_DISTRIBUTE (THROW to the nearest teammate).
2. If the ball is loose near your box: INTERCEPT.
3. Otherwise: MOVE\_TO a point between the ball and the goal centre.

Customisation tips from the library: rewrite the DEF prompt as a second midfielder for a 3-forward shape; pair aggressive forwards with a balanced defender; keep prompts short because of the 5-second window.

## Prompt engineering and coordination

The system prompt is the highest-leverage thing to tune: a mediocre model with a great prompt often beats a strong model with a vague one. Different positions need different prompts.

### Four layers of a football agent prompt

1. **Identity and role:** "You are a defensive midfielder on a 5v5 team. You are Player 2 on the home team."
2. **Decision hierarchy:** a numbered priority list that settles trade-offs each tick (e.g. MARK the nearest threat, then hold to receive, then PRESS\_BALL a loose ball, then PASS, then MOVE\_TO forward).
3. **Situational rules:** winning means play safe; losing means push and shoot; stamina below 30% means no sprinting.
4. **Constraints:** e.g. never leave the defensive third empty, never shoot from beyond halfway, never sprint below 20% stamina.

### Template

```
You are a [POSITION] on a 5v5 football team. You are Player [ID] on the [HOME/AWAY] team.
Your role: [ONE SENTENCE]
Decision priorities (in order):
1. [HIGHEST]
2. [SECOND]
3. [THIRD]
4. [DEFAULT]
Situational adjustments:
- When WINNING by 2+: [behaviour]
- When LOSING: [behaviour]
- When LOW STAMINA: [behaviour]
You must NEVER:
- [Constraint 1]
- [Constraint 2]
Available commands: MOVE_TO, FOLLOW_PLAYER, SHOOT, PASS, GK_DISTRIBUTE, PRESS_BALL, MARK, INTERCEPT, SLIDE_TACKLE, SET_STANCE, CLEAR_OVERRIDE, RESET
```

**Anti-patterns:** the vague prompt ("play well and win") gives no priorities; the novel (hundreds of words of drama) wastes tokens and time. Encode strategy as behaviours, not labels: instead of "play counter-attack", say what to do when the opponent has the ball, when you win it, and when you're in the attacking third.

### Coordination

| Failure | What it looks like | Fix |
| --- | --- | --- |
| Ball magnet | All 5 chase the ball | Give each agent a position and zone |
| Ghost town | Good spacing, no passes | Teammate-aware passing rules |
| Confusion | Two agents press or mark the same player | "If a teammate already marks him, mark the next nearest threat" |
| Statue garden | Perfect shape, no reaction | "Maintain shape" rules that still react to the ball |

For each role define a primary zone, a primary action, a trigger to leave the zone, and a handoff rule. Example zones: DEF x −55 to −15 (MARK nearest opponent in zone, hand off to MID when the ball reaches midfield); MID x −15 to 15 (support the carrier, push up when losing late, leave the defensive third to DEF). Add awareness rules such as "before shooting, pass if a teammate is better placed" and "if the carrier is under pressure, move toward them".

## Coaching during matches

Coach messages are suggestions, not commands: they arrive in `teamChat` on the next tick, and agents act on them only if your code reads that field. All 10 players hear every message, opponents included.

- **How to send:** Player Portal → Locker Room → Watch Live, or the Viewer Portal with your team code.
- **Name your players** on the My Team page first, then address them: "Arango, shoot the ball now!" beats "Team, you need to shoot". "Salomon, move left, tackle Rincon, and shoot as quickly as you can" beats "tackle the opponents".
- **Be specific and scoped:** "Win the ball back quickly in the opponent's half", "defenders hold position", "forwards push higher".
- **One change per message;** split complex shifts into a sequence.
- **History:** agents can treat "push forward, we need a goal" as an escalation from an earlier "hold position" only if your code keeps conversation history. The platform doesn't carry coaching across matches.
- When instructions conflict, agents tend to follow the latest one, so change tactics deliberately.

Example messages: "Press higher, win the ball back quickly", "Hold position, wait for counter-attack", "Focus on possession, slow the game down", "Push forward, we need a goal".

## Player Portal and match flow

Log in at [agentic-football.aws.dev](https://agentic-football.aws.dev) with the team code from your organiser, register 5 ARNs, test, play at least one practice match, then play tournament games.

| Section | Nav label | What you do there |
| --- | --- | --- |
| Training Camp | Command Center | Team status, standings, upcoming matches |
| My Team | My Team | Register 5 agent ARNs, pick a formation, name players, customise coach and jersey |
| Locker Room | Locker Room | View matches, run practice, Watch Live, send coach instructions |
| Trophy Room | Trophy Room | AgentCore building blocks discovered in your AWS account |
| Coach Room | Coach Room | AgentCore Evaluations (LLM-as-a-judge) of decision quality |

1. **Set up:** pick a logo and a formation (e.g. 2-1-1, 1-2-1, 3-0-1). The formation only sets starting positions; after kick-off your agents decide where to move.
2. **Register agents:** paste the AgentCore ARN into each slot (GK, DEF, MID, FWD), from Bedrock console → AgentCore → Runtime, or Harness if you used Harness. Set player names and avatars.
3. **Test your agents:** sends a sample game state. If it fails, check the agent shows Ready and the cross-account IAM trust policy. Harness agents can fail on output format; retry or use a larger model (e.g. Claude Sonnet 4.6, Nova Pro).
4. **Practice:** play the AWS reference teams. Stats (latency, success rate, command breakdown) are recorded but don't count. At least one practice match is required before the admin can start your official matches.
5. **Find opponent:** challenge other ready teams; the server invokes both teams' agents.
6. **Watch live:** Locker Room → Watch Live, or the public Viewer Portal at /viewer/\<event-code> (standings, bracket, matches) or /viewer/\<team-code> (adds coaching). Choppy playback is only your WebSocket stream; agents play at full speed on the server.
7. **Iterate:** observe, change prompts, redeploy, replay. LLMs are non-deterministic, so play several matches per change and try different models. The portal also tracks XP and daily challenges in a Team Progress panel.

## Build and deploy

There are two ways to deploy: code agents on AgentCore Runtime (macOS, Windows or CloudShell paths) or no-code Harness agents configured in the console. Both end with 5 ARNs to register in the portal.

### Code path (Runtime)

```bash
git clone --filter=blob:none --sparse https://github.com/aws-samples/sample-ai-possibilities
cd sample-ai-possibilities
git sparse-checkout set agentic-football-sample-agents
cd agentic-football-sample-agents
```

| Path | Contents |
| --- | --- |
| lib/agent\_base.py | Agent factory and invoke handler |
| lib/fallback.py | Rule-based fallback per position |
| lib/parsing.py | Pulls JSON commands out of the LLM text and forces the right playerId |
| lib/state.py | Summarises game state for the LLM |
| ai-team-strands-balanced / -extremely-aggressive / -extremely-defensive | Sample teams, each with ai-gk, ai-def, ai-mid, ai-fwd1, ai-fwd2, deploy-all.sh, deploy-all-windows.ps1 |
| ai-team-strands-memory / -gateway | Phase 3 variants |
| \<agent>/src/main.py | MY\_PLAYER\_ID, POSITION\_LABEL, SYSTEM\_PROMPT, fallback, `create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-micro-v1:0")` |
| \<agent>/test\_local.py | Offline tests; add `--llm` to call Bedrock |

Each agent has three layers of error handling (LLM, parsing, rule-based fallback), so it never freezes when the model times out or returns bad output.

1. **Install:** Python 3.10+, AWS CLI, uv, `pip install strands-agents bedrock-agentcore-starter-toolkit`, then one agent's requirements.txt. uv is needed to cross-compile for the AgentCore Linux ARM64 runtime; Phase 3 deploys need Python 3.10 exactly.
2. **Credentials:** paste the export lines from Workshop Studio's "Get AWS CLI credentials", then check with `aws sts get-caller-identity`.
3. **Test:** `python3 ai-gk/test_local.py` for each of the 5 agents; `--llm` sends a sample state to the model.
4. **Deploy:** `cd ai-team-strands-balanced && ./deploy-all.sh` (or `./deploy-all.sh ai-gk` for one agent). The script stages code, fills `.bedrock_agentcore.yaml`, runs `agentcore deploy`, auto-creates the IAM role and cleans up.
5. **Get ARNs:** Bedrock console → AgentCore → Runtime → each agent's Runtime ARN.

### Harness path (no code)

1. Bedrock console → AgentCore → Harness → dropdown on Quick create → **Advanced create harness** (Quick create can't set model, instructions or role).
2. Name (e.g. ai\_gk\_agent), model, Instructions = the position's prompt from the Agent Prompts Library.
3. Service role: **Use another role** → team-SolutionAccessRole-\*. "Create default role" fails because participants can't create IAM roles.
4. Repeat for all 5 positions and copy each Harness ARN.

Model choice: a strong but slow model can time out; a fast weak model reasons poorly and, on Harness, breaks the JSON format more often. Larger models (Claude Sonnet, Nova Pro) follow the format more reliably than Nova Micro. Harness JSON format is enforced only by the prompt, not by code.

## AgentCore upgrades (Phase 3)

Three optional upgrades: Memory, so agents remember earlier ticks; Gateway, so agents call calculation tools; Observability, so you can see why they decided. Each one adds latency per tick, so check it against the 5-second budget.

| Upgrade | Team folder | What it adds | Check it works |
| --- | --- | --- | --- |
| Memory | ai-team-strands-memory | Short-term memory via AgentCoreMemorySessionManager; each position has its own isolated session in one shared resource (AITeamMatchMemory). Each tick: recall, reason, record | AgentCore → Memory → Observability: Create events about 320 per match, 0 errors; Retrieve extracted memory = 0 is expected |
| Gateway | ai-team-strands-gateway | 4 Lambda tools behind the MCP gateway afwc-tactical-tools | CloudWatch runtime logs show lines like "Tool #N: evaluate-shot" |
| Observability | Already on for all samples | Session metrics, traces, spans, logs; GenAI Observability dashboard shows invocations, latency, errors and tokens per agent | Enable CloudWatch Transaction Search once (1% indexing is free), then view traces under /aws/spans/default |

### Gateway tools

| Tool | Returns | Main users |
| --- | --- | --- |
| calculate\_pass\_options | Pass success probability per teammate, based on interception risk | All positions |
| evaluate\_shot | Shot success probability plus a recommended aim point | MID, FWD1, FWD2 |
| find\_open\_space | Open space on a grid, by zone (attack, midfield, defence) | MID, FWD1, FWD2 |
| get\_defensive\_assignment | Opponent threat ranking for marking priority | GK, DEF |

### Example trace timings

| Agent type | Total per call | Breakdown |
| --- | --- | --- |
| Basic | \~800 ms | State summary \~5 ms, Nova Micro call \~700 ms, parsing \~2 ms |
| Memory | \~1,200 ms | Plus ListEvents \~150 ms and CreateEvent \~100 ms |
| Gateway | \~2,000 ms | Two LLM calls (\~500 + \~400 ms) plus a tool call \~200 ms |

Log patterns to watch for in /aws/bedrock-agentcore/runtimes/\<name>-DEFAULT: "LLM parse failed, using fallback" (bad JSON), "agent error", AccessDeniedException (missing Memory or Gateway permission), and Timeout (the server moved on without your answer). After deploying Memory or Gateway agents, replace the ARNs on the My Team page.

## Agent concepts

Each player is an agent running a loop every tick: it perceives the game state, reasons about it with an LLM, and acts by returning one command. The workshop's goal is to move your team up the scale below.

| Level | Behaviour |
| --- | --- |
| Reactive | Hard-coded if-then rules; fast but predictable |
| Reasoning | LLM weighs options each tick; the workshop's starting point |
| Adaptive | Remembers what worked and spots opponent patterns (memory) |
| Collaborative | Five agents coordinate as one unit; the main challenge |

| System component | Football analogy | What it means here |
| --- | --- | --- |
| Foundation model | The player's football brain | Trade off capability, context, latency and cost |
| System prompt | Manager's team talk | Role, tactics, priorities, constraints, coordination rules |
| State management | Shared team memory | Conversation history plus agent state (formation, stamina, opponent habits, coaching) |
| Orchestration | The captain | Turn timing, sharing info, assigning roles, settling conflicts |
| Tool integration | Coaching earpiece | Game commands, plus calculators and lookups |
| Guardrails | The referee | Validate output, cap resources, hard rules, fallback actions |
| Observability | Video analysis room | Decision traces, latency, error rates, tactical drift |

AgentCore features you'll touch: Runtime (serverless hosting), Harness (config-only agents), Identity (scoped permissions), Gateway, Memory and Observability. Browser and Code Interpreter exist but aren't used in this workshop.

## Setup, access and cleanup

Workshop-provided accounts are cleaned up automatically after the event; cleanup is only needed if you used your own AWS account.

- **Access:** the event dashboard provides the AWS account and the "Get AWS CLI credentials" export lines. The Event Access page didn't load during this capture.
- **Tools:** Kiro is optional but recommended (the setup guide is in the appendix); CloudShell is the no-install alternative. Any coding tool that can edit code and run a terminal works.
- **Lost in the repo?** `cd ~/sample-ai-possibilities/agentic-football-sample-agents`

### Cleanup (own account only)

1. Run `agentcore destroy` in each team folder, or delete the agents under AgentCore → Runtime.
2. Gateway team: delete the 4 targets on afwc-tactical-tools, then the gateway itself.
3. Gateway team: delete the 4 Lambda functions afwc-gateway-tool-\*.
4. Memory team: delete AITeamMatchMemory.
5. Delete the IAM roles afwc-gateway-tool-lambda-role, AfwcGatewayExecutionRole and AmazonBedrockAgentCoreSDKRuntime-\*.
6. Check that Runtime, Gateways and Memory are all empty.

## Gaps and inconsistencies

A few things in the instructions don't line up. Check them in practice matches instead of assuming.

| Topic | What the pages say | Implication |
| --- | --- | --- |
| SLIDE\_TACKLE | Missing from the Actions table but listed in every sample prompt and the prompt template | Probably valid; test it in practice |
| Fouls and cards | Rules mention fouls, cards and player count; FREE\_KICK is a play mode; no penalty or card details given | Treat aggressive tackling as risky |
| Response window | Actions page says 5 s; prompt-engineering page says \~500 ms; tick interval is \~2 s | Aim for under 2 s per call |
| Formations | Portal offers 2-1-1, 1-2-1, 3-0-1; samples use 1-1-1-2 | The formation only sets starting spots; prompts decide roles |
| Benchmark team name | "Bedrock Ballers FC" on the home page, "The Benchmark FC" in Phase 2 | Same balanced reference team |
| Coaching privacy | All 10 players, opponents included, see every message | Name your own players; opponents' agents may react too |

### Sources

All from the Agentic Football Cup Cyprus workshop on AWS Workshop Studio: Home, Introduction, What is Football?, Football Strategies and Tactics, Before Kickoff (4 pages), How to Play, Actions and Stats, Game Rules, Coaching During Matches, Phase 1 (3 pages including Harness tabs), Phase 2, Phase 3 (prompt engineering, Memory, Gateway, Observability, coordination), What's Next, Agent Prompts Library (Balanced, Aggressive, Defensive), Cleanup. [Workshop home](https://catalog.us-east-1.prod.workshops.aws/event/dashboard/en-US/workshop)
