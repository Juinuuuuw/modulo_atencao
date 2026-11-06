"""A câmera da tela de apresentação (papel interno "validation") é analisada como as demais."""
import json
import tempfile
import time
import unittest

import cv2
import numpy as np

from modules.screen_gaze import ScreenGaze
from modules.target_classifier import TARGETS, TargetClassifier
from server import create_app

VECTORS = {label: [0.15 + i * 0.25, 0.2 + i * 0.23, 0.2 + i * 0.2, 0.25 + i * 0.17]
           for i, label in enumerate(TARGETS)}


def samples(roles):
    return [{"role": role, "target": label, "features": vector, "step": i}
            for role in roles for i, (label, vector) in enumerate(VECTORS.items()) for _ in range(10)]


class PresentationCameraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.app = create_app(self.directory.name, "test-access", start_worker=False)
        self.client = self.app.test_client()
        self.headers = {"Authorization": "Bearer test-access"}
        self.exp = self.app.extensions["experiment"]

    def tearDown(self):
        self.exp.close()
        self.directory.cleanup()

    def configure(self, presentation=True):
        self.client.post("/api/nodes/register", headers=self.headers, json={
            "id": "test", "name": "Test", "cameras": [{"id": str(i), "name": f"Camera {i}"} for i in range(3)]})
        response = self.client.post("/api/configure", headers=self.headers, json={"assignments": {
            "notebook": "test:0", "nao": "test:1", "validation": "test:2" if presentation else None}})
        self.assertEqual(response.status_code, 200)

    def test_classifier_trains_and_predicts_with_presentation_camera(self):
        classifier = TargetClassifier()
        report = classifier.fit(samples(("notebook", "nao", "validation")))
        self.assertEqual(report["usable_cameras"], ["notebook", "nao", "validation"])
        self.assertEqual(classifier.predict("validation", VECTORS["TV"])["target"], "TV")

    def test_missing_presentation_camera_does_not_warn_or_block(self):
        report = TargetClassifier().fit(samples(("notebook", "nao")))
        self.assertEqual(report["usable_cameras"], ["notebook", "nao"])
        self.assertEqual(report["warnings"], [])

    def test_presentation_preview_has_eye_overlay(self):
        self.configure()
        _, jpeg = cv2.imencode(".jpg", np.zeros((60, 80, 3), np.uint8))
        self.client.post("/api/nodes/test/frame/2", data=jpeg.tobytes(), content_type="image/jpeg", headers={
            **self.headers, "X-Capture-Time": str(time.perf_counter()), "X-Sync-Uncertainty": "0.01", "X-Sequence": "1"})
        response = self.client.get("/api/preview/validation", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("eyes", json.loads(response.headers["X-Eye-Overlay"]))

    def test_unassigned_presentation_camera_is_left_out_of_fusion(self):
        self.configure(presentation=False)
        with self.exp.lock:
            self.exp.tick(time.perf_counter())
        self.assertEqual(set(self.exp.predictions), {"notebook", "nao"})
        self.configure()
        with self.exp.lock:
            self.exp.tick(time.perf_counter())
        self.assertEqual(set(self.exp.predictions), {"notebook", "nao", "validation"})

    def test_assignments_come_back_after_server_restart_without_closing_cameras(self):
        self.configure()
        self.exp.close()
        # Novo servidor no mesmo diretório; o conector se registra de novo com outro ID.
        app = create_app(self.directory.name, "test-access", start_worker=False)
        client, exp = app.test_client(), app.extensions["experiment"]
        client.post("/api/nodes/register", headers=self.headers, json={
            "id": "again", "name": "Test", "cameras": [{"id": str(i), "name": f"Camera {i}"} for i in range(3)]})
        self.assertEqual(exp.assignments, {"notebook": "again:0", "nao": "again:1", "validation": "again:2"})
        beat = client.post("/api/nodes/again/heartbeat", headers=self.headers, json={}).json
        self.assertEqual(sorted(beat["cameras"]), ["0", "1", "2"])
        # Outro computador com as mesmas câmeras não recebe papéis.
        other = create_app(self.directory.name, "test-access", start_worker=False)
        other.test_client().post("/api/nodes/register", headers=self.headers, json={
            "id": "x", "name": "Outro", "cameras": [{"id": "0", "name": "Camera 0"}]})
        self.assertEqual(set(other.extensions["experiment"].assignments.values()), {None})
        exp.close()
        other.extensions["experiment"].close()

    def test_position_calibration_accepts_presentation_camera(self):
        gaze = ScreenGaze()
        gaze.start("tv", "validation", "P001", "test:2")
        self.assertTrue(gaze.active)
        with self.assertRaises(ValueError):
            ScreenGaze().start("tv", "outra", "P001", "test:2")


if __name__ == "__main__":
    unittest.main()
