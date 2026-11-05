"""Run one agent folder's src/main.py as a local AgentCore server.
Usage: python serve.py <agent_dir> <port>   (e.g. ../../ai-team-strands-balanced/ai-gk 9200)"""
import os
import runpy
import sys

agent_dir, port = os.path.abspath(sys.argv[1]), int(sys.argv[2])
os.chdir(agent_dir)
module = runpy.run_path(os.path.join(agent_dir, "src", "main.py"), run_name="served")
module["app"].run(port=port)
