#!/usr/bin/env bash
# Start a team's 5 agents as local servers on <base_port>..<base_port+4>, calling real Bedrock.
# Usage: ./start_team.sh <name> <base_port> <team_dir>
#   e.g. ./start_team.sh kak1 9100 ../../ai-team-kak1
#        ./start_team.sh balanced 9200 ../../ai-team-strands-balanced
# Needs: the venv active and AWS credentials (AWS_PROFILE or exported keys). Logs: ./logs/<name>-<agent>.log
set -euo pipefail
name=$1; base=$2; team=$(cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/logs"
export AWS_REGION=${AWS_REGION:-us-east-1} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
# The simulator uses the documented pitch (x +-55). kak1 scales from the live pitch (+-7) unless told
# otherwise, so run it unscaled here. Other teams ignore this variable.
export PITCH_HALF_X=${PITCH_HALF_X:-55}
export MATCH_SECONDS=${MATCH_SECONDS:-300}   # simulator matches are 5 min (live ones are 120 s)
i=0
for agent in ai-gk ai-def ai-mid ai-fwd1 ai-fwd2; do
  python "$here/serve.py" "$team/$agent" $((base + i)) > "$here/logs/$name-$agent.log" 2>&1 &
  i=$((i + 1))
done
for _ in $(seq 1 30); do
  up=0
  for p in $(seq "$base" $((base + 4))); do
    curl -s -o /dev/null "localhost:$p/ping" && up=$((up + 1))
  done
  [ "$up" -eq 5 ] && { echo "$name: 5/5 agents up on $base-$((base + 4))"; exit 0; }
  sleep 2
done
echo "$name: only $up/5 agents came up; see $here/logs/" >&2; exit 1
