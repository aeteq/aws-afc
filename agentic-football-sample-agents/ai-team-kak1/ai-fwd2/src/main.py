"""kak1 FWD agent (player 4: the single forward in 1-2-1). Logic: lib/fast_agent.py (model layer) + lib/tactics.py (rules).
Tune per player here: MODEL_ID, LLM_TIMEOUT_MS. Deployed by ../deploy-all.sh."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for p in ("../../../lib",      # local: shared workshop lib
          "../../kaklib",      # local: our modules
          "../lib"):           # deployed: shared lib + kaklib staged together
    sys.path.insert(0, os.path.normpath(os.path.join(_here, p)))

os.environ.setdefault("MODEL_ID", "us.amazon.nova-micro-v1:0")
os.environ.setdefault("LLM_TIMEOUT_MS", "1500")   # tick is ~2 s, server limit 5 s (docs disagree: ~500 ms)

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402
from fast_agent import create_handler  # noqa: E402

MY_PLAYER_ID = 4
POSITION_LABEL = "FWD"

app = BedrockAgentCoreApp()
create_handler(app, MY_PLAYER_ID)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", "8080")))
