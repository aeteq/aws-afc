"""Deterministic shared assignments and ranked actions; no network dependencies."""
import math
from state import _player_idx, _is_my_team, dist


def clamp(x, low, high):
    return max(low, min(high, x))


def lane_clearance(a, b, opponents):
    """Minimum opponent distance to the finite passing/shooting segment."""
    dx, dy = b['x'] - a['x'], b['y'] - a['y']
    length2 = dx * dx + dy * dy
    def clearance(p):
        p = p['position']
        t = clamp(((p['x']-a['x'])*dx + (p['y']-a['y'])*dy) / max(length2, .001), 0, 1)
        return math.hypot(p['x']-a['x']-t*dx, p['y']-a['y']-t*dy)
    return min(map(clearance, opponents), default=100)


def possession(state):
    """Resolve possession conservatively: ambiguous cross-team IDs are unknown."""
    ball = state.get('ball', {})
    if ball.get('isFree') is True:
        return None
    players = state.get('players', [])
    if isinstance(ball.get('possessionAgentId'), str):
        matches = [p for p in players if p.get('agentId') == ball['possessionAgentId']]
    elif ball.get('possessionPlayerId') is not None:
        matches = [p for p in players if _player_idx(p) == ball['possessionPlayerId']]
    else:
        return None
    if ball.get('possessionTeamId') in (0, 1):
        matches = [p for p in matches if _is_my_team(p, ball['possessionTeamId'])]
    return matches[0] if len(matches) == 1 else None


def mode(state, team, match_seconds=0):
    score = state.get('score', {})
    diff = score.get('home', 0) - score.get('away', 0)
    diff *= 1 if team == 0 else -1
    if match_seconds > 0 and state.get('gameTime', 0) >= match_seconds * .75:
        return 'protect' if diff > 0 else 'chase' if diff < 0 else 'balanced'
    return 'balanced'


def candidates(state, team, pid, match_seconds=0):
    """Return ranked, fully formed commands. Off-ball assignments are mandatory."""
    def cmd(kind, **params):
        return dict(commandType=kind, teamId=team, playerId=pid, parameters=params, duration=0)
    players = state.get('players', [])
    ours = sorted([p for p in players if _is_my_team(p, team)], key=_player_idx)
    opponents = [p for p in players if not _is_my_team(p, team)]
    me = next((p for p in ours if _player_idx(p) == pid), None)
    if me is None:
        return [cmd('SET_STANCE', stance=2 if pid < 2 else 0)]
    pos = me['position']; ball = state.get('ball', {}).get('position', {'x':0,'y':0})
    direction = 1 if team == 0 else -1
    goal = {'x':55*direction, 'y':0}
    tactical_mode = mode(state, team, match_seconds)
    shift = {'protect':-10, 'balanced':0, 'chase':8}[tactical_mode]
    stamina = me.get('stamina', 1)
    def move(x, y, urgent=False):
        return cmd('MOVE_TO', target_x=clamp(x*direction,-54,54), target_y=clamp(y,-32,32), sprint=bool(urgent and stamina > .35))
    holder = possession(state)
    if holder is me:
        options = []
        blockers = [p for p in opponents if _player_idx(p) != 0]
        distance = dist(pos, goal)
        angle = abs(math.atan2(pos['y'], max(.1, 55-pos['x']*direction)))
        if pid != 0 and distance < 26 and angle < .7 and lane_clearance(pos, goal, blockers) > 3:
            options.append(cmd('SHOOT', aim_location='TR', power=clamp(.6+distance/80,.6,1)))
        passes = []
        for p in ours:
            if p is me: continue
            target = p['position']; clearance = lane_clearance(pos, target, opponents)
            pressure = min((dist(target,o['position']) for o in opponents),default=100)
            if clearance < 3 or pressure < 4: continue
            quality = min(clearance,12)+min(pressure,12)+.25*direction*(target['x']-pos['x'])-.12*dist(pos,target)
            passes.append((quality,_player_idx(p)))
        for _, target in sorted(passes, key=lambda item:(-item[0],item[1]))[:2]:
            options.append(cmd('GK_DISTRIBUTE', target_player_id=target, method='KICK') if pid == 0 else cmd('PASS',target_player_id=target,type='GROUND'))
        # Keep the goalkeeper deep even when no safe distribution is available.
        options.append(move(-50,clamp(ball['y']*.3,-4,4)) if pid == 0 else move(min(48,pos['x']*direction+6),pos['y']))
        return options
    if pid == 0:
        return [move(-50, clamp(ball['y']*.3,-4,4))]
    we_have_ball = holder is not None and _is_my_team(holder,team)
    # Preserve the covering defender unless the ball threatens our defensive third.
    press_pool = [p for p in ours if _player_idx(p)>1 or (_player_idx(p)==1 and ball['x']*direction < -25)]
    presser = min(press_pool,key=lambda p:(dist(p['position'],ball),_player_idx(p)),default=None)
    if not we_have_ball and presser is me:
        return [cmd('PRESS_BALL',intensity=.8 if stamina>.35 else .4)]
    if pid == 1:
        return [move(min(-12,ball['x']*direction-18)+min(shift,0),clamp(ball['y']*.4,-12,12))]
    if pid == 2:
        return [move(clamp(ball['x']*direction-10,-30,20)+shift,clamp(-ball['y']*.4,-12,12))]
    # Separate striker/support lanes; drop back together when defending.
    x = (32 if pid == 3 else 18) if we_have_ball else (5 if pid == 3 else -8)
    return [move(x+shift,-12 if pid==3 else 12,we_have_ball)]
