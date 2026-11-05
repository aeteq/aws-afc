#!/usr/bin/env bash
# Stop every local agent server started by start_team.sh.
pkill -f "simulation/live/serve.py" && echo "stopped" || echo "nothing running"
