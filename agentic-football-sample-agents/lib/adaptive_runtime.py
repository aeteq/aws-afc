"""Stateless model selection from locally validated tactical choices."""
import json
import os
import time
from adaptive_tactics import candidates, mode
from json_tolerant import parse_json_tolerant


def select(text, options):
    parsed = parse_json_tolerant(text)
    value = parsed[0] if parsed else None
    index = value.get('choice') if isinstance(value, dict) else None
    if type(index) is not int or not 0 <= index < len(options):
        raise ValueError('Expected an available integer choice')
    return options[index]


def register(app, default_pid, label, model_id):
    # Imported here so tactical tests need no AWS or Strands packages.
    from strands import Agent
    from strands.models import BedrockModel
    from agent_base import parse_prompt
    from state import summarize_state
    match_seconds = float(os.environ.get('ADAPTIVE_MATCH_SECONDS', '0'))
    rules_only = os.environ.get('ADAPTIVE_RULES_ONLY', '0') == '1'

    @app.entrypoint
    async def invoke(payload, context):
        started = time.monotonic()
        data = parse_prompt(payload)
        team = data.get('teamId')
        if type(team) is not int or team not in (0, 1):
            app.logger.error('Missing or invalid teamId; no commands emitted')
            yield '[]'
            return
        owned = data.get('myPlayers', [default_pid])
        pid = owned[0] if isinstance(owned, list) and owned else default_pid
        if type(pid) is not int or pid not in range(5):
            yield '[]'
            return
        state = data.get('gameState', {})
        source = 'rules'
        try:
            options = candidates(state, team, pid, match_seconds)
            command = options[0]
            if len(options) > 1 and not rules_only:
                try:
                    # A fresh conversation prevents history leaking between matches.
                    agent = Agent(model=BedrockModel(model_id=model_id, temperature=0, max_tokens=64),
                        system_prompt='Choose one supplied football action. Options are ranked best first. Prefer a clear shot, otherwise a safe progressive pass. Return ONLY {"choice":0} with the selected zero-based index. Do not invent commands.')
                    prompt = summarize_state(state, team, pid, label)
                    prompt += '\nMode: ' + mode(state, team, match_seconds)
                    prompt += '\nChoices: ' + json.dumps(options)
                    command = select(str(agent(prompt)), options)
                    source = 'model'
                except Exception as exc:
                    source = 'fallback'
                    app.logger.warning(f'Adaptive model fallback: {exc}')
        except Exception as exc:
            source = 'last_resort'
            app.logger.error(f'Adaptive state error: {exc}')
            command = dict(commandType='SET_STANCE',teamId=team,playerId=pid,parameters={'stance':2 if pid<2 else 0},duration=0)
        app.logger.info(json.dumps(dict(player=pid,source=source,latency_ms=round((time.monotonic()-started)*1000),command=command['commandType'])))
        yield json.dumps([command])
    return invoke
