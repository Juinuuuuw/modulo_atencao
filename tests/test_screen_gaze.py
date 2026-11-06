import copy
import tempfile
import unittest

import numpy as np

from modules.screen_gaze import ScreenMap, ScreenGaze, FIT_POINTS, same_layout
from server import create_app


def vector(point):
    x, y = point
    return [.2+.5*x, .2+.5*y, .4+.04*x, .4+.04*y]


class ScreenGazeTests(unittest.TestCase):
    def test_window_border_shift_is_not_a_new_screen(self):
        self.assertTrue(same_layout([1920,1080,8,8],[1920,1080,0,0]))
        self.assertFalse(same_layout([1920,1080,8,8],[1280,720,8,8]))
        self.assertFalse(same_layout([1920,1080,8,8],[1920,1080,1928,8]))
    def test_first_point_restarts_when_fullscreen_finishes(self):
        gaze=ScreenGaze();gaze.start('notebook','notebook','P001','camera:0')
        obs={'notebook':{'capture':0,'sequence':1,'features':vector((.15,.15)),'uncertainty':.01}}
        displays={'window':{'display':'notebook','visible':True,'ready':True,'cue_id':gaze.cue(0)['cue_id'],
                            'seen':0,'layout':[1000,700,0,0]}}
        gaze.tick(0,obs,{'notebook':'camera:0'},displays)
        displays['window']['layout']=[1920,1080,0,0]
        obs['notebook']['capture']=.1
        gaze.tick(.1,obs,{'notebook':'camera:0'},displays)
        self.assertTrue(gaze.active)
        self.assertIsNone(gaze.calibration['started'])
        self.assertEqual(gaze.calibration['layout'],[1920,1080,0,0])

    def test_waits_for_valid_eyes_and_allows_late_samples(self):
        gaze = ScreenGaze()
        gaze.start('notebook','notebook','P001','camera:0')
        obs, displays = {}, {}
        def tick(now, valid):
            cue = gaze.cue(now)
            obs['notebook'] = {'capture':now,'sequence':now,'features':vector((.15,.15)) if valid else None,
                               'uncertainty':.01,'quality_reason':None if valid else 'Mais de um rosto'}
            displays['window'] = {'display':'notebook','visible':True,'ready':True,
                                  'cue_id':cue['cue_id'],'seen':now,'layout':[1280,720,0,0]}
            gaze.tick(now,obs,{'notebook':'camera:0'},displays)
        tick(0,False)
        tick(20,False)
        self.assertEqual(gaze.cue(20)['phase'],'waiting')
        self.assertIn('Mais de um rosto',gaze.cue(20)['reason'])
        tick(21,True)
        tick(24.1,False)
        self.assertEqual(gaze.cue(24.1)['phase'],'collecting')
        for i in range(6):
            tick(24.2+i*.1,True)
        self.assertEqual(gaze.cue(25)['index'],1)

    def test_blinks_are_not_calibration_samples(self):
        gaze = ScreenGaze()
        gaze.start('notebook','notebook','P001','camera:0')
        obs, displays = {}, {}
        def tick(now, ear):
            cue = gaze.cue(now)
            obs['notebook'] = {'capture':now,'sequence':now,'features':vector((.05,.08)),'uncertainty':.01,'ear':ear}
            displays['window'] = {'display':'notebook','visible':True,'ready':True,
                                  'cue_id':cue['cue_id'],'seen':now,'layout':[1280,720,0,0]}
            gaze.tick(now,obs,{'notebook':'camera:0'},displays)
        tick(0,.3)
        for i in range(10):
            tick(.6+i*.1,.08)
        self.assertEqual(gaze.cue(1.6)['valid_samples'],0)
        self.assertIn('piscando',gaze.cue(1.6)['reason'])
        tick(1.7,.3)
        self.assertEqual(gaze.cue(1.7)['valid_samples'],1)

    def test_affine_position_and_domain(self):
        model = ScreenMap()
        model.fit([{'step':i,'features':vector(p)} for i,p in enumerate(FIT_POINTS) for _ in range(8)])
        np.testing.assert_allclose(model.predict(vector((.35,.65))), [.35,.65], atol=.01)
        self.assertIsNone(model.predict([50]*4))
        self.assertIsNone(model.predict([float('nan')]*4))
        with self.assertRaises(ValueError):
            ScreenMap().fit([{'step':i,'features':[.5]*4} for i in range(5) for _ in range(8)])
        with self.assertRaises(ValueError):
            ScreenMap().fit([])

    def run_calibration(self, wrong_check=False, duplicate=False, center_checks=False, swapped_fit=False, jitter=False, include_check=True):
        gaze = ScreenGaze()
        gaze.start('notebook','notebook','P001','camera:0',include_check=include_check)
        assignments = {'notebook':'camera:0'}
        observations, displays = {}, {}
        for sequence in range(900):
            now = sequence/10
            cue = gaze.cue(now)
            if not cue['active']:
                break
            point = (.05,.95) if wrong_check and cue['mode']=='check' else cue['marker']
            if center_checks and cue['mode']=='check':point=(.5,.5)
            if swapped_fit and cue['mode']=='fit':point=(point[1],point[0])
            if jitter and cue['mode']=='check' and sequence%3==0:point=(point[0]+.4,point[1])
            observations['notebook'] = {'capture':now,'sequence':0 if duplicate else sequence,
                                        'features':vector(point),'uncertainty':.01,'pose':None}
            displays['window'] = {'display':'notebook','visible':True,'ready':True,
                                  'cue_id':cue['cue_id'],'seen':now,'layout':[1280,720,0,0]}
            gaze.tick(now, observations, assignments, displays)
        return gaze, now, observations, assignments, displays

    def test_eight_steps_then_position_without_other_display(self):
        gaze, now, obs, assignments, displays = self.run_calibration()
        self.assertEqual(gaze.cue(now)['phase'], 'complete')
        self.assertTrue(gaze.models['notebook:notebook']['validation']['accepted'])
        obs['notebook']['features'] = vector((.7,.3))
        position = gaze.position('notebook:notebook',now,obs,assignments,displays)
        self.assertTrue(position['visible'])
        np.testing.assert_allclose([position['x'],position['y']],[.7,.3],atol=.01)
        self.assertFalse(gaze.position('notebook:notebook',now+1,obs,assignments,displays)['visible'])
        # Uma classificação geral discordante não interfere no cursor desta webcam.
        self.assertTrue(gaze.position('notebook:notebook',now,obs,assignments,displays)['visible'])
        obs['notebook']['features'] = None
        self.assertFalse(gaze.position('notebook:notebook',now,obs,assignments,displays)['visible'])

    def test_bad_holdout_is_not_approved_but_keeps_estimate_for_inspection(self):
        gaze,now,obs,assignments,displays=self.run_calibration(wrong_check=True)
        self.assertEqual(gaze.cue(now)['phase'],'complete')
        self.assertFalse(gaze.models['notebook:notebook']['validation']['accepted'])
        self.assertFalse(gaze.position('notebook:notebook',now,obs,assignments,displays).get('verified',False))

    def test_duplicate_frames_do_not_create_map(self):
        gaze,now,*_=self.run_calibration(duplicate=True)
        self.assertEqual(gaze.cue(now)['phase'],'failed')
        self.assertFalse(gaze.models)

    def test_constant_center_and_swapped_axes_are_rejected(self):
        for options in ({'center_checks':True},{'swapped_fit':True}):
            gaze,now,*_=self.run_calibration(**options)
            self.assertEqual(gaze.cue(now)['phase'],'complete')
            self.assertFalse(gaze.models['notebook:notebook']['validation']['accepted'])

    def test_jitter_is_rejected_even_with_good_medians(self):
        gaze,now,*_=self.run_calibration(jitter=True)
        self.assertEqual(gaze.cue(now)['phase'],'complete')
        self.assertLess(gaze.cue(now)['validation']['mean_error'],.01)
        self.assertGreater(gaze.cue(now)['validation']['p90_error'],.16)

    def test_default_finishes_after_nine_points_without_claiming_accuracy(self):
        gaze,now,obs,assignments,displays=self.run_calibration(include_check=False)
        self.assertEqual(gaze.cue(now)['index'],9)
        self.assertEqual(len(FIT_POINTS),9)
        self.assertIsNone(gaze.models['notebook:notebook']['validation']['accepted'])
        self.assertEqual(gaze.models['notebook:notebook']['map'].data['kind'],'ScreenRegression')
        position=gaze.position('notebook:notebook',now,obs,assignments,displays)
        self.assertTrue(position['visible'])
        self.assertFalse(position['verified'])
        gaze.begin_check('notebook','notebook')
        self.assertTrue(gaze.active)
        self.assertEqual(gaze.cue(now)['mode'],'check')

    def test_eyelids_recover_vertical_gaze_when_iris_height_is_flat(self):
        # Webcam acima da tela: a íris quase não sobe/desce, mas as pálpebras acompanham.
        rng=np.random.default_rng(3)
        def sample(point):
            x,y=point
            return [.5+.3*(x-.5)+rng.normal(0,.004),.45+rng.normal(0,.004),.5,.5], .32-.12*y+rng.normal(0,.003)
        rows=[]
        for i,p in enumerate(FIT_POINTS):
            for _ in range(20):
                features,ear=sample(p);rows.append({'step':i,'features':features,'ear':ear})
        model=ScreenMap();model.fit(rows,[1920,1080])
        self.assertIn('ear',model.data['features'])
        self.assertLess(model.data['loo_error_px'],model.data['candidates']['iris']['loo_error_px'])
        for point in ((.3,.7),(.8,.2)):
            predictions=[model.predict(f,geometry={'ear':e}) for f,e in (sample(point) for _ in range(15))]
            np.testing.assert_allclose(np.median(predictions,axis=0),point,atol=.03)
        self.assertNotIn('hx',model.data['features'])

    def test_simplest_model_wins_when_iris_explains_everything(self):
        model=ScreenMap()
        model.fit([{'step':i,'features':vector(p)} for i,p in enumerate(FIT_POINTS) for _ in range(8)])
        self.assertEqual(model.data['model'],'iris')
        v=vector((.7,.3));expected=model.predict(v)
        v[2:]=[1.2,-.1]
        np.testing.assert_allclose(model.predict(v),expected)

    def test_blinks_are_removed_from_fit_and_hide_live_position(self):
        rows=[{'step':i,'features':vector(p),'ear':.3} for i,p in enumerate(FIT_POINTS) for _ in range(10)]
        rows+=[{'step':i,'features':vector((p[0],.99)),'ear':.1} for i,p in enumerate(FIT_POINTS) for _ in range(3)]
        model=ScreenMap();model.fit(rows)
        np.testing.assert_allclose(model.predict(vector((.5,.2)),geometry={'ear':.3}),[.5,.2],atol=.01)
        self.assertIsNone(model.predict(vector((.5,.2)),geometry={'ear':.1}))
        gaze,now,obs,assignments,displays=self.run_calibration()
        obs['notebook']['ear']=.05
        position=gaze.position('notebook:notebook',now,obs,assignments,displays)
        self.assertFalse(position['visible'])
        self.assertIn('Piscada',position['reason'])

    def test_operator_view_does_not_need_target_page(self):
        gaze,now,obs,assignments,displays=self.run_calibration()
        obs['notebook']['features']=vector((.6,.4))
        self.assertFalse(gaze.position('notebook:notebook',now,obs,assignments,{})['visible'])
        position=gaze.position('notebook:notebook',now,obs,assignments,{},require_view=False)
        self.assertTrue(position['visible'])
        np.testing.assert_allclose([position['x'],position['y']],[.6,.4],atol=.01)
        self.assertTrue(gaze.summary('notebook:notebook')['calibrated'])
        self.assertFalse(gaze.summary('tv:notebook')['calibrated'])

    def test_screen_can_use_eye_reading_rejected_by_general_classifier(self):
        gaze,now,obs,assignments,displays=self.run_calibration()
        obs['notebook']['screen_features']=obs['notebook']['features']
        obs['notebook']['features']=None
        self.assertTrue(gaze.position('notebook:notebook',now,obs,assignments,displays)['visible'])

    def test_camera_window_and_robot_pose_changes_hide_position(self):
        gaze, now, obs, assignments, displays = self.run_calibration()
        moved = copy.deepcopy(displays)
        moved['window']['layout'][0] = 900
        self.assertFalse(gaze.position('notebook:notebook',now,obs,assignments,moved)['visible'])
        self.assertFalse(gaze.position('notebook:notebook',now,obs,{'notebook':'other'},displays)['visible'])
        gaze.models['notebook:notebook']['pose'] = [0,0]
        obs['notebook']['pose'] = [.2,0]
        self.assertFalse(gaze.position('notebook:notebook',now,obs,assignments,displays)['visible'])

    def test_api_requires_login_and_assigned_camera(self):
        with tempfile.TemporaryDirectory() as output:
            app = create_app(output, 'test-only', start_worker=False)
            client = app.test_client()
            self.assertEqual(client.get('/api/screen-gaze').status_code,401)
            self.assertEqual(client.get('/api/screen-gaze/diagnostic').status_code,401)
            client.post('/api/login',json={'key':'test-only'})
            self.assertFalse(client.get('/api/screen-gaze').json['gaze']['visible'])
            data = {'action':'start','display':'notebook','participant':'P001','role':'notebook'}
            self.assertEqual(client.post('/api/screen-gaze',json=data).status_code,400)
            experiment = app.extensions['experiment']
            experiment.assignments['notebook'] = 'camera:0'
            self.assertEqual(client.post('/api/screen-gaze',json=data).status_code,200)
            self.assertTrue(client.get('/api/screen-gaze').json['calibration']['active'])
            self.assertEqual(client.post('/api/screen-gaze',json={'action':'cancel','display':'notebook'}).status_code,200)
            self.assertFalse(experiment.screen_gaze.active)

    def test_completed_calibration_saves_actual_samples_for_diagnosis(self):
        import json
        from pathlib import Path
        from modules.multicamera import Experiment
        gaze,now,*_=self.run_calibration(include_check=False)
        with tempfile.TemporaryDirectory() as output:
            experiment=Experiment(output)
            experiment.screen_gaze=gaze
            experiment.tick(now)
            path=Path(output)/'calibrations'/f"{gaze.calibration['id']}.json"
            diagnostic=json.loads(path.read_text())
            self.assertEqual(diagnostic['map']['kind'],'ScreenRegression')
            self.assertGreaterEqual(len(diagnostic['samples']),30)
            self.assertIsNone(diagnostic['validation']['accepted'])


class LiveGazeTests(unittest.TestCase):
    def test_classifier_reports_robot_part(self):
        from modules.target_classifier import TargetClassifier, fuse
        centers = {('NOTEBOOK',0,None):[.5,.6,.5,.5],('TV',1,None):[.5,.2,.5,.3],('NAO',2,'head'):[.2,.4,.3,.5],
                   ('NAO',3,'left-leg'):[.2,.7,.3,.7],('NAO',4,'right-leg'):[.3,.7,.35,.7],('OUTRO_LOCAL',5,None):[.9,.5,.8,.5]}
        rng=np.random.default_rng(1)
        samples=[{'role':'notebook','target':t,'step':step,'part':part,'features':list(np.asarray(c)+rng.normal(0,.005,4))}
                 for (t,step,part),c in centers.items() for _ in range(10)]
        classifier=TargetClassifier();classifier.fit(samples)
        prediction=classifier.predict('notebook',[.2,.4,.3,.5])
        self.assertEqual((prediction['target'],prediction['part']),('NAO','head'))
        self.assertIsNone(classifier.predict('notebook',[.5,.6,.5,.5])['part'])
        fused=fuse([prediction,{'target':'INDETERMINADO','score':0,'reason':'x'}])
        self.assertEqual(fused['part'],'head')

    def test_live_endpoint_reports_screens_and_robot(self):
        with tempfile.TemporaryDirectory() as output:
            app = create_app(output, 'test-only', start_worker=False)
            client = app.test_client()
            self.assertEqual(client.get('/api/gaze/live').status_code,401)
            client.post('/api/login',json={'key':'test-only'})
            experiment = app.extensions['experiment']
            experiment.live = {'target':'NAO','score':.7,'reason':'concordância','part':'left-leg'}
            data = client.get('/api/gaze/live').json
            # Sem calibrações de posição, a visão ao vivo usa o classificador de alvos.
            self.assertEqual((data['robot']['looking'],data['robot']['part']),(True,'left-leg'))
            self.assertEqual(data['source'],'nenhuma')
            self.assertFalse(data['screens']['tv']['calibrated'])
            self.assertEqual(data['screens']['notebook']['estimates'],[])
            self.assertGreater(experiment.live_viewer_seen,0)
