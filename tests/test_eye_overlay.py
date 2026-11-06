import json
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from modules.multicamera import Experiment, Frame
from modules.visual_attention_monitor import FaceLandmarkExtractor
from modules.target_classifier import feature_status, screen_feature_status


class EyeOverlayTests(unittest.TestCase):
    def test_screen_reading_does_not_inherit_target_classifier_ear_threshold(self):
        from modules.visual_attention_monitor import VisualObservation
        obs=VisualObservation(True,0,ear=.14,gaze_feature_x=.5,gaze_feature_y=.5,
                              head_x=.5,head_y=.5,iris_diameter=8)
        self.assertIsNone(feature_status(obs)[0])
        self.assertIsNotNone(screen_feature_status(obs)[0])

    def test_two_faces_never_silently_select_another_person(self):
        extractor=FaceLandmarkExtractor.__new__(FaceLandmarkExtractor)
        extractor.mesh=SimpleNamespace(process=lambda _:SimpleNamespace(multi_face_landmarks=[object(),object()]))
        observation=extractor.extract(np.zeros((480,640,3),np.uint8),0)
        self.assertEqual(observation.face_count,2)
        self.assertFalse(observation.detected)
        vector,reason=feature_status(observation)
        self.assertIsNone(vector)
        self.assertIn('Mais de um rosto',reason)

    def test_extractor_returns_normalized_iris_centers_and_neutral_vector(self):
        points=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
        for i in FaceLandmarkExtractor.LEFT_IRIS:points[i]=SimpleNamespace(x=.4,y=.425)
        for i in FaceLandmarkExtractor.RIGHT_IRIS:points[i]=SimpleNamespace(x=.6,y=.425)
        for index,x,y in [(234,.2,.5),(454,.8,.5),(10,.5,.2),(152,.5,.8),
                          (33,.35,.425),(133,.45,.425),(362,.55,.425),(263,.65,.425),
                          (159,.4,.4),(160,.4,.4),(145,.4,.45),(144,.4,.45),
                          (386,.6,.4),(385,.6,.4),(374,.6,.45),(380,.6,.45)]:
            points[index]=SimpleNamespace(x=x,y=y)
        extractor=FaceLandmarkExtractor.__new__(FaceLandmarkExtractor)
        extractor.mesh=SimpleNamespace(process=lambda _:SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=points)]))
        observation=extractor.extract(np.zeros((480,640,3),np.uint8),0)
        np.testing.assert_allclose(observation.eye_centers,[[.4,.425],[.6,.425]])
        np.testing.assert_allclose(observation.gaze_direction,[0,0],atol=1e-12)

    def test_unknown_before_calibration_and_stale_geometry_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Experiment(directory)
            now=time.perf_counter()
            frame=Frame(None,b'',now,now,.01,3)
            exp.observations['nao']={'capture':now,'features':[.5]*4,'eyes':[[.4,.4],[.6,.4]],'direction':[.2,0],'sequence':3}
            exp.predictions['nao']={'target':'TV','score':1}
            exp.live={'target':'TV','score':1,'reason':'concordância'}
            self.assertEqual(exp.eye_overlay('nao',frame)['target'],'INDETERMINADO')
            exp.classifier.models['nao']={}
            overlay=exp.eye_overlay('nao',frame)
            self.assertEqual(overlay['target'],'TV')
            self.assertEqual(overlay['kind'],'illustrative_2d')
            json.dumps(overlay,allow_nan=False)
            exp.observations['nao']['capture']=now-.6
            overlay=exp.eye_overlay('nao',frame)
            self.assertIsNone(overlay['eyes'])
            self.assertEqual(overlay['target'],'INDETERMINADO')

    def test_disagreement_never_becomes_a_red_or_green_target(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Experiment(directory)
            now=time.perf_counter()
            exp.observations['notebook']={'capture':now,'features':[.5]*4,'eyes':[[.4,.4],[.6,.4]],'direction':[.2,0]}
            exp.classifier.models['notebook']={}
            exp.predictions['notebook']={'target':'TV','score':1}
            exp.live={'target':'INDETERMINADO','score':0,'reason':'câmeras discordam'}
            overlay=exp.eye_overlay('notebook',Frame(None,b'',now,now,.01,1))
            self.assertEqual(overlay['target'],'INDETERMINADO')

    def test_task_region_can_be_green_while_exact_target_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Experiment(directory);now=time.perf_counter()
            exp.observations['notebook']={'capture':now,'features':[.5]*4,'eyes':[[.4,.4],[.6,.4]],'direction':[.2,0]}
            exp.classifier.models['notebook']={}
            regions={key:{'target':'DENTRO','score':.9,'reason':'região calibrada'} for key in ['screens','robot','task']}
            exp.predictions['notebook']={'target':'INDETERMINADO','score':0,'regions':regions}
            exp.live={'target':'INDETERMINADO','score':0,'reason':'alvo ambíguo','regions':regions}
            frame=Frame(None,b'',now,now,.01,1)
            result=exp.eye_overlay('notebook',frame)
            self.assertEqual(result['target'],'INDETERMINADO')
            self.assertEqual(result['regions']['task']['target'],'DENTRO')
            exp.classifier.models={}
            self.assertEqual(exp.eye_overlay('notebook',frame)['regions']['task']['target'],'INDETERMINADO')

    def test_auxiliary_view_uses_fusion_color_but_not_when_face_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            exp=Experiment(directory);now=time.perf_counter()
            exp.classifier.models['notebook']={}
            exp.observations['nao']={'capture':now,'features':None,'eyes':[[.4,.4],[.6,.4]],'direction':None}
            exp.live={'target':'TV','score':.9,'reason':'uma câmera disponível'}
            frame=Frame(None,b'',now,now,.01,1)
            result=exp.eye_overlay('nao',frame)
            self.assertEqual(result['regions']['task']['target'],'DENTRO')
            self.assertFalse(result['calibrated'])
            self.assertEqual(result['color_basis'],'fused')
            exp.observations['nao']['eyes']=None
            self.assertEqual(exp.eye_overlay('nao',frame)['regions']['task']['target'],'INDETERMINADO')
