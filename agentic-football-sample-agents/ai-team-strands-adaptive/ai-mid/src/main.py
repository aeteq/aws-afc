"""Adaptive team: MID, player 2."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path
setup_lib_path(__file__)
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from adaptive_runtime import register
app = BedrockAgentCoreApp()
register(app, 2, "MID", "us.amazon.nova-pro-v1:0")
if __name__ == "__main__":
    app.run()
