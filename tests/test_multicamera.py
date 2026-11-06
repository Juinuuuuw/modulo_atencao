import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import zipfile

import cv2
import numpy as np

from modules.multicamera import Experiment, Frame
from modules.calibration_protocol import calibration_plan
from modules.target_classifier import TARGETS, UNKNOWN, TargetClassifier, fuse
from modules.validation import evaluate_checks, evaluate_intervals, validate_annotations
from server import create_app


VECTORS = {label: [0.15 + i * 0.25, 0.2 + i * 0.23, 0.2 + i * 0.2, 0.25 + i * 0.17]
           for i, label in enumerate(TARGETS)}


def samples():
    return [{"role": role, "target": label, "features": vector, "step": i}
            for role in ("notebook", "nao") for i, (label, vector) in enumerate(VECTORS.items()) for _ in range(10)]


class ClassificationTests(unittest.TestCase):
    def test_ambiguous_screens_still_classify_task_without_inventing_exact_target(self):
        rows=samples()
        for row in rows:
            if row['target']=='TV':row['features']=VECTORS['NOTEBOOK']
        classifier=TargetClassifier();classifier.fit(rows)
        result=classifier.predict('notebook',VECTORS['NOTEBOOK'])
        self.assertEqual(result['target'],UNKNOWN)
        self.assertEqual(result['regions']['screens']['target'],'DENTRO')
        self.assertEqual(result['regions']['task']['target'],'DENTRO')

    def test_task_fusion_tolerates_target_disagreement_but_not_task_outside_disagreement(self):
        result=fuse([{'target':'TV','score':.95},{'target':'NAO','score':.95}])
        self.assertEqual(result['target'],UNKNOWN)
        self.assertEqual(result['regions']['task']['target'],'DENTRO')
        self.assertEqual(result['regions']['screens']['target'],UNKNOWN)
        result=fuse([{'target':'TV','score':.95},{'target':'OUTRO_LOCAL','score':.95}])
        self.assertEqual(result['regions']['task']['target'],UNKNOWN)

    def test_ambiguous_or_far_all_regions_do_not_turn_green(self):
        rows=samples()
        for row in rows:row['features']=[.5]*4
        classifier=TargetClassifier();classifier.fit(rows)
        for vector in ([.5]*4,[100]*4):
            result=classifier.predict('notebook',vector)
            self.assertTrue(all(r['target']==UNKNOWN for r in result['regions'].values()))

    def test_prototypes_reject_ambiguous_and_far_inputs(self):
        classifier = TargetClassifier()
        classifier.fit(samples())
        for label, vector in VECTORS.items():
            self.assertEqual(classifier.predict("notebook", vector)["target"], label)
        self.assertEqual(classifier.predict("notebook", [100] * 4)["target"], UNKNOWN)
        self.assertEqual(classifier.predict("notebook", None)["target"], UNKNOWN)
        identical = samples()
        for row in identical:
            row["features"] = [0.5] * 4
        classifier.fit(identical)
        self.assertEqual(classifier.predict("notebook", [0.5] * 4)["target"], UNKNOWN)

    def test_partial_camera_is_excluded_instead_of_guessing_missing_targets(self):
        classifier = TargetClassifier()
        rows = [s for s in samples() if s["role"] == "notebook" or s["target"] != "TV"]
        report = classifier.fit(rows)
        self.assertEqual(report["usable_cameras"], ["notebook"])
        self.assertTrue(report["warnings"])
        self.assertEqual(classifier.predict("nao", VECTORS["TV"])["target"], UNKNOWN)

    def test_opposite_outside_fixations_do_not_create_a_center_target(self):
        rows = samples()
        for row in rows:
            if row["target"] == "OUTRO_LOCAL": row["features"] = [-0.4]*4
        rows += [{"role":role,"target":"OUTRO_LOCAL","features":[1.4]*4,"step":12}
                 for role in ("notebook","nao") for _ in range(10)]
        classifier = TargetClassifier()
        classifier.fit(rows)
        self.assertEqual(classifier.predict("notebook",[-0.4]*4)["target"],"OUTRO_LOCAL")
        self.assertEqual(classifier.predict("notebook",[1.4]*4)["target"],"OUTRO_LOCAL")
        self.assertNotEqual(classifier.predict("notebook",[0.5]*4)["target"],"OUTRO_LOCAL")

    def test_fusion_disagreement_and_single_camera_threshold(self):
        a = {"target": "NAO", "score": 0.9}
        b = {"target": "TV", "score": 0.9}
        self.assertEqual(fuse([a,b])["target"], UNKNOWN)
        self.assertEqual(fuse([a])["target"], "NAO")
        self.assertEqual(fuse([{**a,"score":0.5}])["target"], UNKNOWN)

    def test_stale_predictions_and_moved_nao_do_not_remain_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Experiment(directory)
            exp.classifier.fit(samples())
            now = time.perf_counter()
            for role in ("notebook", "nao"):
                exp.observations[role] = {"features": VECTORS["NAO"], "capture": now,
                                          "uncertainty": 0.01, "pose": [0.0, 0.0]}
            exp.pose_reference["nao"] = [0.0, 0.0]
            exp.tick(now)
            self.assertEqual(exp.live["target"], "NAO")
            exp.observations["nao"]["pose"] = [0.2, 0]
            exp.tick(now)
            self.assertEqual(exp.predictions["nao"]["target"], UNKNOWN)
            exp.tick(now + 1)
            self.assertEqual(exp.live["target"], UNKNOWN)

    def test_holdout_does_not_train_model(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Experiment(directory)
            exp.assignments.update(notebook="n:0", nao="n:1")
            exp.classifier.fit(samples())
            original = copy.deepcopy(exp.classifier.models)
            exp.begin_calibration(check=True)
            exp.capture_step()
            for index in range(18):
                exp.tick(exp.calibration["started"] + 0.5 + index * 0.19)
            self.assertEqual(exp.classifier.models, original)
            self.assertEqual(exp.samples, [])


class GuidedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.exp=Experiment(self.directory.name)
        self.exp.assignments.update(notebook="test:0",nao="test:1")

    def tearDown(self):
        self.exp.close()
        self.directory.cleanup()

    def acknowledge(self, now, visible=True):
        cue_id=self.exp.cue().get("cue_id")
        for display in ("notebook","tv"):
            self.exp.displays[display]={"display":display,"seen":now,"ready":True,"visible":visible,"cue_id":cue_id}
        for role in ("notebook","nao"):
            self.exp.frames[self.exp.assignments[role]]=Frame(None,b"",now,now,0.01,1)

    def test_quick_protocol_covers_head_and_both_legs_in_fit_and_check(self):
        fit=calibration_plan(False,"quick")
        check=calibration_plan(True,"quick")
        self.assertEqual(len(fit),9)
        self.assertEqual(len(check),6)
        for plan in (fit,check):
            self.assertEqual({s.get("part") for s in plan if s["target"]=="NAO"},{"head","left-leg","right-leg"})
            self.assertTrue(all("marker" in s for s in plan if s["target"] in ("TV","NOTEBOOK")))

    def test_auto_waits_for_matching_visible_display_ack_and_drops_interrupted_step(self):
        self.exp.begin_calibration(participant="P001",profile="quick",automatic=True)
        now=time.perf_counter()
        self.exp._advance_calibration(now)
        self.assertIsNone(self.exp.calibration["started"])
        self.acknowledge(now)
        self.exp.displays["tv"]["cue_id"]="old-cue"
        self.exp._advance_calibration(now)
        self.assertIsNone(self.exp.calibration["started"])
        self.acknowledge(now)
        self.exp._advance_calibration(now)
        self.assertEqual(self.exp.calibration["started"],now)
        self.exp.samples=[{"step":0,"role":"notebook","features":[0]*4,"target":"NOTEBOOK"}]
        self.acknowledge(now+.4,visible=False)
        self.exp._advance_calibration(now+.4)
        self.assertEqual(self.exp.samples,[])
        self.assertIsNone(self.exp.calibration["started"])

    def test_full_auto_protocol_enters_independent_check_and_finishes(self):
        self.exp.begin_calibration(participant="P001",profile="quick",automatic=True)
        now=time.perf_counter()
        for step_index in range(15):
            cal=self.exp.calibration
            target=cal["plan"][cal["index"]]["target"]
            self.acknowledge(now)
            self.exp._advance_calibration(now)
            if cal["mode"]=="fit":
                self.exp.samples.extend({"step":cal["index"],"role":role,"target":target,
                                         "features":VECTORS[target]} for role in ("notebook","nao") for _ in range(8))
            else:
                fitted=copy.deepcopy(self.exp.classifier.models)
                for j in range(10):
                    t=now+1.05+j*.19
                    self.acknowledge(t)
                    for role in ("notebook","nao"):
                        self.exp.observations[role]={"features":VECTORS[target],"capture":t,"uncertainty":.01,"pose":None}
                    self.exp.tick(t)
                self.assertEqual(self.exp.classifier.models,fitted)
            self.acknowledge(now+3.01)
            self.exp._advance_calibration(now+3.01)
            now+=3.3
        self.assertTrue(self.exp.calibration["complete"])
        self.assertEqual(self.exp.calibration["mode"],"check")
        self.assertTrue(self.exp.check_report["sufficient"])
        self.assertEqual(self.exp.check_report["accuracy"],1)

    def test_missing_observations_pause_instead_of_silently_accepting_step(self):
        self.exp.begin_calibration(participant="P001",profile="quick",automatic=True)
        now=time.perf_counter()
        for _ in range(2):
            self.acknowledge(now);self.exp._advance_calibration(now)
            self.acknowledge(now+3.01);self.exp._advance_calibration(now+3.01)
            now+=4
        self.assertTrue(self.exp.calibration["paused"])
        self.assertIsNotNone(self.exp.calibration["error"])
        self.assertEqual(self.exp.calibration["index"],0)

    def test_next_participant_drops_previous_models(self):
        self.exp.classifier.fit(samples())
        self.exp.calibration_participant="P001"
        self.exp.cancel_calibration()
        self.assertFalse(self.exp.classifier.models)
        self.assertIsNone(self.exp.calibration_participant)

    def test_assisted_profile_collects_without_display_acknowledgements(self):
        self.exp.begin_calibration(participant='P001',profile='assisted',chain_check=True)
        self.assertFalse(self.exp.calibration['automatic'])
        self.assertNotIn('ponto',self.exp.calibration['plan'][0]['point'])
        self.exp.capture_step()
        self.assertIsNotNone(self.exp.calibration['started'])
        self.assertEqual(self.exp.displays,{})

    def test_assisted_profile_chains_to_independent_verification(self):
        self.exp.begin_calibration(participant='P001',profile='assisted',chain_check=True)
        cal=self.exp.calibration
        self.exp.samples=[{'role':role,'target':step['target'],'step':i,'features':VECTORS[step['target']]}
                          for role in ('notebook','nao') for i,step in enumerate(cal['plan']) for _ in range(8)]
        cal['index']=len(cal['plan'])-1
        cal['started']=time.perf_counter()-3.1
        self.exp._advance_calibration(time.perf_counter())
        self.assertEqual(self.exp.calibration['mode'],'check')
        self.assertEqual(self.exp.calibration['profile'],'assisted')
        self.assertFalse(self.exp.calibration['automatic'])
        self.assertTrue(self.exp.classifier.models)

    def test_assisted_step_with_no_face_stays_on_same_target_for_retry(self):
        self.exp.begin_calibration(participant='P001',profile='assisted',chain_check=True)
        self.exp.capture_step()
        self.exp._advance_calibration(self.exp.calibration['started']+3.1)
        self.assertEqual(self.exp.calibration['index'],0)
        self.assertIsNone(self.exp.calibration['started'])
        self.assertIsNotNone(self.exp.calibration['error'])


class ValidationTests(unittest.TestCase):
    def test_task_verification_is_reported_separately_from_exact_target(self):
        result=evaluate_checks([{'expected':'TV','target':UNKNOWN,'regions':{'task':{'target':'DENTRO'}}}])
        self.assertEqual(result['accuracy'],0)
        self.assertEqual(result['task_metrics']['accuracy'],1)
        self.assertEqual(result['task_metrics']['coverage'],1)
    def test_lost_frames_and_abstentions_count_in_accuracy(self):
        result = evaluate_intervals([{ "t": 0, "target": "TV"}],
                                    [{"start":0,"end":2,"target":"TV"}], 2)
        self.assertAlmostEqual(result["accuracy"], 0.25)
        self.assertAlmostEqual(result["coverage"], 0.25)
        self.assertAlmostEqual(result["matrix"]["TV"][UNKNOWN], 1.5)

    def test_duration_weighting_boundaries_and_human_unknown(self):
        rows = [{"t":0,"target":"TV"},{"t":0.2,"target":"NAO"},{"t":0.4,"target":"NAO"}]
        intervals = [{"start":0,"end":0.4,"target":"TV"},
                     {"start":0.4,"end":0.8,"target":UNKNOWN}]
        result = evaluate_intervals(rows, intervals, 0.8)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["total"], 0.4)
        self.assertAlmostEqual(result["ambiguous_seconds"], 0.4)
        self.assertAlmostEqual(result["annotation_coverage"], 1)

    def test_overlap_nonfinite_and_out_of_bounds_are_rejected(self):
        invalid = [
            [{"start":0,"end":2,"target":"TV"},{"start":1,"end":3,"target":"TV"}],
            [{"start":float("nan"),"end":2,"target":"TV"}],
            [{"start":0,"end":11,"target":"TV"}],
            [{"start":0,"end":2,"target":"INVALID"}],
        ]
        for intervals in invalid:
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                validate_annotations(intervals, 10)

    def test_empty_metrics_are_not_perfect_scores(self):
        result = evaluate_intervals([], [], 10)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["kappa"])
        self.assertIsNone(result["macro_f1"])


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.app = create_app(self.directory.name, "test-access", start_worker=False)
        self.client = self.app.test_client()
        self.headers = {"Authorization":"Bearer test-access"}
        self.exp = self.app.extensions["experiment"]

    def tearDown(self):
        self.exp.close()
        self.directory.cleanup()

    def register_and_configure(self):
        response = self.client.post("/api/nodes/register", json={"id":"test", "name":"Test computer",
            "cameras":[{"id":str(i),"name":f"Camera {i}"} for i in range(3)]}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post("/api/configure", json={"assignments":{
            "notebook":"test:0", "nao":"test:1", "validation":"test:2"}}, headers=self.headers)
        self.assertEqual(response.status_code, 200)

    def upload(self, index, sequence=1):
        _, jpeg = cv2.imencode(".jpg",np.zeros((60,80,3), np.uint8))
        return self.client.post(f"/api/nodes/test/frame/{index}", data=jpeg.tobytes(),
                                content_type="image/jpeg", headers={**self.headers,
                                "X-Capture-Time":str(time.perf_counter()), "X-Sync-Uncertainty":"0.01",
                                "X-Sequence":str(sequence)})

    def test_authentication_and_origin_checks_protect_video_and_control(self):
        self.assertEqual(self.client.get("/api/state").status_code,401)
        self.assertEqual(self.client.get("/api/preview/notebook").status_code,401)
        self.assertEqual(self.client.post("/api/login",json={"key":"wrong"}).status_code,401)
        self.assertEqual(self.client.post("/api/login",json={"key":"test-access"},
                                         headers={"Origin":"http://malicious.example"}).status_code,403)
        self.assertEqual(self.client.post("/api/login",json={"key":"test-access"}).status_code,200)
        self.assertEqual(self.client.get("/api/state").status_code,200)

    def test_duplicate_assignment_and_replayed_frames_rejected(self):
        self.register_and_configure()
        duplicate = self.client.post("/api/configure",json={"assignments":{
            "notebook":"test:0","nao":"test:0","validation":"test:2"}},headers=self.headers)
        self.assertEqual(duplicate.status_code,400)
        self.assertTrue(self.upload(0).json["accepted"])
        self.assertFalse(self.upload(0).json["accepted"])
        self.assertEqual(self.client.get("/api/preview/notebook",headers=self.headers).status_code,200)
        response = self.client.get("/api/preview/notebook",headers=self.headers)
        self.assertEqual(response.headers["X-Frame-Sequence"],"1")
        self.assertEqual(json.loads(response.headers["X-Eye-Overlay"])["target"],"INDETERMINADO")
        self.assertEqual(response.data,self.exp.frames["test:0"].jpeg)
        unchanged = self.client.get("/api/preview/notebook?after=1",headers=self.headers)
        self.assertEqual(unchanged.status_code,204)
        self.upload(0,2)
        changed = self.client.get("/api/preview/notebook?after=1",headers=self.headers)
        self.assertEqual(changed.status_code,200)
        self.assertEqual(changed.headers["X-Frame-Sequence"],"2")
        state = self.client.get("/api/state",headers=self.headers).json
        self.assertGreater(state["cameras"]["notebook"]["fps"],0)

    def test_rgb_nao_frames_and_bad_images(self):
        self.register_and_configure()
        headers={**self.headers,"X-Width":"80","X-Height":"60","X-Head-Pose":"[0,0]",
                 "X-Capture-Time":str(time.perf_counter()),"X-Sync-Uncertainty":"0.01","X-Sequence":"1"}
        response = self.client.post("/api/nodes/test/frame/1",data=bytes(80*60*3),
                                    content_type="application/x-rgb",headers=headers)
        self.assertEqual(response.status_code,200)
        self.assertEqual(self.exp.frames["test:1"].pose,[0,0])
        response = self.client.post("/api/nodes/test/frame/1",data=b"invalid",content_type="image/jpeg",headers=headers)
        self.assertEqual(response.status_code,400)

    def test_full_record_review_and_export_without_camera_hardware(self):
        self.register_and_configure()
        self.exp.samples = samples()
        self.exp.calibration_participant = "P001"
        self.exp.fit_report = self.exp.classifier.fit(self.exp.samples)
        self.exp.checks = [{"expected":label,"target":label,"score":1,"step":i}
                           for i,label in enumerate(TARGETS) for _ in range(6)]
        self.exp.check_report = {**evaluate_checks(self.exp.checks),"sufficient":True}
        for i in range(3): self.assertEqual(self.upload(i).status_code,200)
        response = self.client.post("/api/recording/start",json={"participant":"P001"},headers=self.headers)
        self.assertEqual(response.status_code,200, response.json)
        session_id = response.json["id"]
        for i in range(3): self.upload(i,2)
        self.exp.tick(time.perf_counter())
        self.client.post("/api/events",json={"kind":"DECISAO_NOTEBOOK","note":"cena 1"},headers=self.headers)
        self.client.post("/api/recording/stop",headers=self.headers)
        details = self.client.get(f"/api/sessions/{session_id}",headers=self.headers).json
        self.assertEqual(len(details["frames"]),3)
        self.assertEqual(details["events"][0]["kind"],"DECISAO_NOTEBOOK")
        duration = details["metadata"]["duration"]
        response = self.client.post(f"/api/sessions/{session_id}/annotations",headers=self.headers,
            json={"annotator":"A01","intervals":[{"start":0,"end":duration,"target":"TV"}]})
        self.assertEqual(response.status_code,200)
        result = self.client.get(f"/api/sessions/{session_id}/report?annotator=A01",headers=self.headers)
        self.assertEqual(result.status_code,200)
        self.assertAlmostEqual(result.json["total"],duration)
        self.assertEqual(result.json["coverage"],0)
        with self.client.get(f"/api/sessions/{session_id}/frame/validation/2",headers=self.headers) as response:
            self.assertEqual(response.status_code,200)
        export = self.client.get(f"/api/sessions/{session_id}/predictions.csv",headers=self.headers)
        self.assertIn("t_seconds",export.data.decode("utf-8-sig"))
        self.assertTrue((Path(self.directory.name)/session_id/"report.json").exists())
        wrong_participant = self.client.post("/api/recording/start",json={"participant":"P002"},headers=self.headers)
        self.assertEqual(wrong_participant.status_code,400)

    def test_download_contains_both_connectors_and_no_access_key(self):
        response = self.client.get("/api/client.zip",headers=self.headers)
        self.assertEqual(response.status_code,200)
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            self.assertIn("nao_client.py",archive.namelist())
            self.assertIn("camera_client.py",archive.namelist())
            self.assertFalse(any("key" in path for path in archive.namelist()))

    def test_guided_display_routes_and_pause_resume(self):
        self.register_and_configure()
        with self.client.get("/alvo?display=notebook") as response:
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get("/api/calibration/cue").status_code,401)
        response=self.client.post("/api/calibration",headers=self.headers,
            json={"action":"fit","profile":"quick","automatic":True,"participant":"P001"})
        self.assertEqual(response.status_code,200)
        cue=self.client.get("/api/calibration/cue",headers=self.headers).json
        self.assertEqual(cue["total"],9)
        self.assertEqual(cue["step"]["target"],"NOTEBOOK")
        response=self.client.post("/api/calibration/display",headers=self.headers,
            json={"client":"test-screen","display":"notebook","cue_id":cue["cue_id"],"visible":True,"ready":True})
        self.assertEqual(response.status_code,200)
        self.assertTrue(self.exp.display_status()["notebook"]["ready"])
        for action,paused in (("pause",True),("resume",False)):
            self.client.post("/api/calibration",headers=self.headers,json={"action":action})
            self.assertEqual(self.exp.calibration["paused"],paused)
        self.client.post("/api/calibration",headers=self.headers,json={"action":"cancel"})
        self.assertFalse(self.client.get("/api/calibration/cue",headers=self.headers).json["active"])


if __name__ == "__main__":
    unittest.main()
