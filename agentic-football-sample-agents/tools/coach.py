#!/usr/bin/env python3
"""Between-match coach: Claude reads what happened and proposes a few tactic changes.

    python3 coach.py /tmp/kak1logs --profile afc                  # last match -> proposal
    python3 coach.py /tmp/kak1logs --profile afc --write          # also write coach_config.json
    python3 coach.py /tmp/kak1logs --dry-run                      # print the prompt, no call

Input: the same log exports as live_report.py. It builds the live report of the chosen
match(es), the mined on-ball outcomes (mine_examples.py), and the current values of the
tunable constants in kaklib/tactics.py, and asks Claude (Bedrock, no latency limit) for:

  changes        at most MAX_CHANGES parameter changes, each with a reason and the evidence
  prompt_rules   at most 5 short rules for the in-match model (candidate choice)
  watch          what to check in the next match to confirm or undo the changes

Every change is clamped to the parameter's bounds below and to +-MAX_STEP of its current
value, so one bad read can't wreck the team. --write saves the result to
ai-team-kak1/kaklib/coach_config.json (versioned, with its sources) for tactics.py to load.
Nothing deploys from here: run bench.py and review the diff, then deploy as usual.

Needs: pip install "anthropic[bedrock]", and Bedrock access to the model in the account
(--profile / AWS_PROFILE). Costs one Opus call per run (a few cents).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_report  # noqa: E402
import mine_examples  # noqa: E402

TACTICS = os.path.join(HERE, "..", "ai-team-kak1", "kaklib", "tactics.py")
CONFIG = os.path.join(HERE, "..", "ai-team-kak1", "kaklib", "coach_config.json")
MODEL = "anthropic.claude-opus-5-5"
FALLBACK_MODEL = "anthropic.claude-opus-4-8"
MAX_CHANGES, MAX_STEP = 3, 0.25

# name -> (low, high, meaning). Distances are documented-pitch units (live x ~7.9).
BOUNDS = {
    "SHOOT_RANGE.FWD": (14, 40, "FWD shoots when this close to their goal"),
    "SHOOT_RANGE.MID": (12, 38, "MID shoots when this close to their goal"),
    "SHOOT_RANGE.DEF": (10, 26, "DEF shoots when this close to their goal"),
    "CHASE_SHOOT_BONUS": (0, 8, "extra shoot range when losing late"),
    "KEEPER_OUT_SHOOT_BONUS": (0, 8, "extra shoot range when their keeper is off his line"),
    "OWN_BOX": (10, 25, "inside this distance of our goal the carrier clears the ball"),
    "DEF_THIRD": (25, 50, "depth that counts as our defensive third (mids stay ahead of the ball)"),
    "PRESSURE_DIST": (2, 8, "an opponent this close means the carrier is under pressure"),
    "PRESSED_DIST": (4, 14, "radius for counting pressers; 2+ inside = release long"),
    "GROUND_LANE_MIN": (2, 10, "ground pass only with at least this much free lane"),
    "GROUND_DIST_MAX": (8, 30, "ground pass only up to this distance"),
    "SPRINT_MIN_STAMINA": (0.15, 0.6, "no sprinting below this stamina (0-1)"),
}


def current_values(path=TACTICS):
    """Module-level constants of tactics.py that are in BOUNDS (dict entries as NAME.KEY)."""
    tree = ast.parse(open(path).read())
    vals = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        try:
            v = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(tgt, ast.Tuple) and isinstance(v, tuple):     # A, B = 8.0, 2
            for name, x in zip(tgt.elts, v):
                if isinstance(name, ast.Name) and name.id in BOUNDS and isinstance(x, (int, float)):
                    vals[name.id] = x
            continue
        if not isinstance(tgt, ast.Name):
            continue
        if isinstance(v, dict):
            for k, x in v.items():
                if f"{tgt.id}.{k}" in BOUNDS:
                    vals[f"{tgt.id}.{k}"] = x
        elif tgt.id in BOUNDS and isinstance(v, (int, float)):
            vals[tgt.id] = v
    return vals


def build_inputs(paths, match):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    recs = [r for f in files for r in live_report._records(f)]
    matches = [m for m in live_report._split(recs)
               if len({r.get("t") for r in m if "pid" in r}) >= 40]
    if not matches:
        sys.exit("no full match in the logs")
    chosen = matches if match == "all" else [matches[int(match)]]
    buf = io.StringIO()
    examples = []
    with contextlib.redirect_stdout(buf):
        for i, m in enumerate(chosen):
            live_report.report(m, i)
            examples += mine_examples.mine(m, "live", i)
    return buf.getvalue(), examples


def outcome_table(examples):
    from collections import Counter, defaultdict
    by = defaultdict(list)
    for ex in examples:
        kind = ex["params"].get("type") or ex["params"].get("method") or ""
        by[(ex["role"], ex["cmd"], kind)].append(ex)
    lines = ["role command n mean_value outcomes"]
    for k, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        c = Counter(l for ex in xs for l in ex["labels"])
        v = sum(ex["value"] for ex in xs) / len(xs)
        lines.append(f"{k[0]} {k[1]} {k[2]} n={len(xs)} value={v:+.2f} " +
                     ", ".join(f"{l} {n}" for l, n in c.most_common()))
    return "\n".join(lines)


SYSTEM = """You are the analyst for a 5-a-side football team run by code (an AWS Agentic Football \
Cup team). Between matches you read what happened and propose small, evidence-based changes to the \
team's tactic parameters. The live pitch is about x -7..7 and y -4.5..4.5 ("live units"); the \
parameters are in documented-pitch units, 7.9 times larger in x. Our goal is at x=-7 in the \
report's attack direction.

Rules:
- Propose a change only when the report or the outcome table shows evidence for it, and quote it.
  Few matches means high noise: prefer no change over a guess.
- At most 3 changes. Stay inside each parameter's bounds. Small steps.
- prompt_rules are short instructions for an in-match model that picks one of 2-4 code-generated \
candidate actions; write them as situations -> preference, at most 5, each under 25 words.
- watch: what to look at in the next match to confirm or undo each change."""


SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "changes": {"type": "array", "items": {
            "type": "object",
            "properties": {"param": {"type": "string", "enum": sorted(BOUNDS)},
                           "value": {"type": "number"},
                           "reason": {"type": "string"},
                           "evidence": {"type": "string"}},
            "required": ["param", "value", "reason", "evidence"],
            "additionalProperties": False}},
        "prompt_rules": {"type": "array", "items": {"type": "string"}},
        "watch": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "changes", "prompt_rules", "watch"],
    "additionalProperties": False,
}


def user_message(report, table, values):
    params = "\n".join(f"{k} = {values.get(k)}   (bounds {lo}..{hi}; {meaning})"
                       for k, (lo, hi, meaning) in sorted(BOUNDS.items()))
    return (f"CURRENT PARAMETERS\n{params}\n\nOUTCOMES OF OUR ON-BALL DECISIONS "
            f"(value: goal_for +5, shot +2, kept +1, lost -2, goal_against -5)\n{table}\n\n"
            f"MATCH REPORT\n{report}")


def clamp(changes, values):
    out, notes = [], []
    for c in changes[:MAX_CHANGES]:
        p = c["param"]
        lo, hi, _ = BOUNDS[p]
        cur = values.get(p)
        v = max(lo, min(hi, float(c["value"])))
        if cur:
            v = max(cur * (1 - MAX_STEP), min(cur * (1 + MAX_STEP), v))
        v = round(v, 2)
        if v != c["value"]:
            notes.append(f"{p}: proposed {c['value']}, clamped to {v}")
        if cur is not None and abs(v - cur) < 1e-9:
            continue
        out.append(dict(c, value=v, was=cur))
    return out, notes


def ask_claude(system, user, profile, region):
    from anthropic import AnthropicBedrockMantle, BetaFallbackState, BetaRefusalFallbackMiddleware
    client = AnthropicBedrockMantle(
        aws_region=region, aws_profile=profile,
        middleware=[BetaRefusalFallbackMiddleware([{"model": FALLBACK_MODEL}])])
    with BetaFallbackState():
        msg = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": SCHEMA}})
    if msg.stop_reason == "refusal":
        sys.exit(f"model refused: {getattr(msg, 'stop_details', None)}")
    if msg.stop_reason == "max_tokens":
        sys.exit("answer cut off at max_tokens")
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text), msg.model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="log folders or files")
    ap.add_argument("--match", default="-1", help="index among full matches, or 'all' (default: last)")
    ap.add_argument("--profile", default=os.getenv("AWS_PROFILE"))
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    ap.add_argument("--dry-run", action="store_true", help="print the prompt and stop")
    ap.add_argument("--write", action="store_true", help=f"write {os.path.relpath(CONFIG)}")
    args = ap.parse_args()

    report, examples = build_inputs(args.paths, args.match)
    values = current_values()
    user = user_message(report, outcome_table(examples), values)
    if args.dry_run:
        print(SYSTEM, "\n\n", user)
        return
    ans, model = ask_claude(SYSTEM, user, args.profile, args.region)
    changes, notes = clamp(ans["changes"], values)
    result = {"version": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
              "model": model, "matches": args.match, "summary": ans["summary"],
              "params": {c["param"]: c["value"] for c in changes},
              "changes": changes, "clamped": notes,
              "prompt_rules": ans["prompt_rules"][:5], "watch": ans["watch"]}
    print(json.dumps(result, indent=2))
    if args.write:
        with open(CONFIG, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"\nwrote {os.path.abspath(CONFIG)}", file=sys.stderr)


if __name__ == "__main__":
    main()
