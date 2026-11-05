import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'lib'))
from adaptive_tactics import candidates, mode, possession
from adaptive_runtime import select


def fixture():
    return {'gameTime':90,'score':{'home':1,'away':0},'ball':{'position':{'x':0,'y':0},'isFree':True},
        'players':[{'playerId':i,'teamId':t,'position':{'x':x*(1 if t==0 else -1),'y':y},'stamina':1}
        for t in (0,1) for i,(x,y) in enumerate([(-50,0),(-30,0),(-5,0),(20,-12),(15,12)])]}


class TacticsTests(unittest.TestCase):
    def test_one_presser_and_cover(self):
        s=fixture(); commands=[candidates(s,0,i)[0] for i in range(5)]
        self.assertEqual(sum(c['commandType']=='PRESS_BALL' for c in commands),1)
        self.assertLess(commands[1]['parameters']['target_x'],0)
        self.assertEqual(commands[0]['parameters']['target_x'],-50)

    def test_mirror(self):
        s=fixture()
        for i in range(5):
            a,b=candidates(s,0,i)[0],candidates(s,1,i)[0]
            self.assertEqual(a['commandType'],b['commandType'])
            if a['commandType']=='MOVE_TO':
                self.assertEqual(a['parameters']['target_x'],-b['parameters']['target_x'])

    def test_modes(self):
        s=fixture()
        self.assertEqual(mode(s,0),'balanced')
        self.assertEqual(mode(s,0,100),'protect')
        self.assertEqual(mode(s,1,100),'chase')

    def test_wide_shot_and_free_ball(self):
        s=fixture(); s['players'][3]['position']={'x':45,'y':32}
        s['ball'].update(isFree=False,possessionPlayerId=3,possessionTeamId=0)
        self.assertNotIn('SHOOT',[c['commandType'] for c in candidates(s,0,3)])
        s['ball']['isFree']=True
        self.assertIsNone(possession(s))

    def test_ambiguous_possession(self):
        s=fixture(); s['ball'].update(isFree=False,possessionPlayerId=3)
        self.assertIsNone(possession(s))
        self.assertNotIn('PASS',[c['commandType'] for c in candidates(s,0,3)])

    def test_selection_rejects_invented_actions(self):
        options=candidates(fixture(),0,0)
        self.assertEqual(select('{"choice":0}',options),options[0])
        for text in ['{"choice":true}','{"choice":-1}','{"choice":999}','{"commandType":"RESET"}','oops']:
            with self.assertRaises(ValueError): select(text,options)

    def test_safe_shot_and_blocker(self):
        s=fixture(); s['players'][3]['position']={'x':40,'y':0}
        s['ball'].update(isFree=False,possessionPlayerId=3,possessionTeamId=0)
        self.assertEqual(candidates(s,0,3)[0]['commandType'],'SHOOT')
        s['players'][6]['position']={'x':45,'y':0}
        self.assertNotIn('SHOOT',[c['commandType'] for c in candidates(s,0,3)])

    def test_low_stamina_and_separate_lanes(self):
        s=fixture(); s['ball'].update(isFree=False,possessionPlayerId=2,possessionTeamId=0)
        s['players'][3]['stamina']=.1
        a,b=candidates(s,0,3)[0],candidates(s,0,4)[0]
        self.assertFalse(a['parameters']['sprint'])
        self.assertLess(a['parameters']['target_y'],b['parameters']['target_y'])

if __name__=='__main__': unittest.main()
