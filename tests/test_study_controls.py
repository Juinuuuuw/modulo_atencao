"""Estudo pela aba "Olhar ao vivo": escolher o participante calibrado, iniciar, pausar e finalizar.

O estudo usa a calibração de posição salva do participante, com notebook, TV e robô
juntos como alvo da tarefa, sem exigir o classificador de alvos. A pausa não grava nada
e não entra no tempo do estudo.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules.multicamera import Experiment, read_lines
from server import create_app
from test_cross_zones import SOURCES, calibrate_world, seen

FRAME_SOURCES = ("n1:0", "n1:1")


class StudyFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.exp = Experiment(self.directory.name)
        self.exp.nodes = {"n1": {"name": "pc", "cameras": [{"id": "0"}, {"id": "1"}], "seen": time.perf_counter()}}
        # A câmera do NAO fica sem papel: uma câmera sem papel não impede o estudo.
        self.exp.assignments = {"notebook": "n1:0", "nao": None, "validation": "n1:1"}
        self.rng = np.random.default_rng(3)
        calibrate_world(self.exp.screen_gaze, "notebook", "notebook", "P013", 0, self.rng)
        calibrate_world(self.exp.screen_gaze, "tv", "validation", "P013", 500, self.rng)
        self.sequence = 0

    def tearDown(self):
        self.exp.close()
        self.directory.cleanup()

    def add_robot_map(self):
        # O mundo sintético não tem o robô: um mapa do robô pela câmera do notebook (como no
        # P005), com a zona longe de qualquer leitura, só marca a superfície como calibrada.
        models = self.exp.screen_gaze.models_for("P013")
        base = models["notebook:notebook"]
        zone = {**base["zone"], "center": [value + 10 for value in base["zone"]["center"]]}
        models["robot:notebook"] = {**base, "surface": "robot", "role": "notebook", "zone": zone, "cross_zones": {}}

    def look(self, surface, at=None):
        now = time.perf_counter() if at is None else at
        for camera in SOURCES:
            features, ear = seen(camera, surface, .5, .5, self.rng)
            self.exp.observations[camera] = {"capture": now, "sequence": now, "features": features,
                                             "uncertainty": .01, "ear": ear}
        return now

    def give_frames(self, experiment=None):
        self.sequence += 1
        for source in FRAME_SOURCES:
            (experiment or self.exp).receive(source, None, b"jpeg", time.perf_counter(), .01, self.sequence)

    def session(self, session_id):
        path = Path(self.directory.name) / session_id
        return path, json.loads((path / "session.json").read_text(encoding="utf-8"))


class StudyControlTests(StudyFixture):
    def test_study_uses_the_saved_position_calibration_of_the_three_surfaces(self):
        self.give_frames()
        with self.assertRaisesRegex(ValueError, "robô"):
            self.exp.start_recording("P013")
        self.add_robot_map()
        self.assertFalse(self.exp.classifier.models)
        session_id = self.exp.start_recording("P013")
        path, meta = self.session(session_id)
        self.assertEqual(meta["decision_source"], "posição")
        self.assertEqual(meta["task_targets"], ["NOTEBOOK", "TV", "NAO"])
        self.assertEqual(sorted(meta["screen_gaze_models"]), ["notebook:notebook", "robot:notebook", "tv:validation"])
        self.assertIsNone(meta["check_report"])
        self.assertIsNotNone(meta["attention_method"]["ear_reference"])

        start = self.look("tv")
        self.exp.tick(start)
        row = read_lines(path / "predictions.jsonl")[-1]
        self.assertEqual((row["target"], row["source"]), ("TV", "posição"))
        self.assertEqual(row["regions"]["task"]["target"], "DENTRO")
        self.assertEqual(row["cameras"]["validation"]["target"], "TV")
        self.assertTrue(row["cameras"]["validation"]["tracking_valid"])
        self.assertNotIn("nao", row["cameras"])
        # O alvo gravado é o mesmo da aba ao vivo: um alvo novo precisa durar 0,35 s.
        for offset in (.2, .4, .6):
            self.exp.tick(self.look("notebook", start + offset))
        rows = read_lines(path / "predictions.jsonl")
        self.assertEqual([r["raw_target"] for r in rows], ["TV", "NOTEBOOK", "NOTEBOOK", "NOTEBOOK"])
        self.assertEqual([r["target"] for r in rows], ["TV", "TV", "TV", "NOTEBOOK"])
        live = self.exp.live_gaze(start + .6)
        self.assertEqual((live["participant"], live["recording_participant"]), ("P013", "P013"))
        self.exp.stop_recording()
        self.assertEqual(self.session(session_id)[1]["status"], "complete")

    def test_maps_made_with_other_cameras_do_not_unlock_the_study(self):
        """Caso real (10/11): P030 calibrado nas câmeras do DESKTOP-KLTVOU2, estudo no Pro-Human-Lapto."""
        self.add_robot_map()
        self.give_frames()
        self.exp.screen_gaze.models_for("P013")["tv:validation"]["source"] = "DESKTOP-KLTVOU2/2"
        with self.assertRaisesRegex(ValueError, "a TV .*outras câmeras"):
            self.exp.start_recording("P013")
        live = self.exp.live_gaze(participant="P013")
        self.assertTrue(live["screens"]["tv"]["calibrated"])
        self.assertFalse(live["screens"]["tv"]["usable"])
        self.assertTrue(live["screens"]["notebook"]["usable"])

    def test_pause_records_nothing_and_does_not_count_in_study_time(self):
        self.add_robot_map()
        self.give_frames()
        session_id = self.exp.start_recording("P013")
        path, _ = self.session(session_id)
        self.give_frames()
        self.exp.tick(self.look("tv"))
        frames, rows = len(read_lines(path / "frames.jsonl")), len(read_lines(path / "predictions.jsonl"))
        self.assertEqual((frames, rows), (2, 1))

        self.exp.pause_recording(True)
        self.exp.pause_recording(True)  # clicar de novo não cria outra pausa
        paused_at = self.exp.study_time()
        time.sleep(.3)
        self.give_frames()
        self.exp.tick(self.look("notebook"))
        self.assertEqual(self.exp.study_time(), paused_at)
        self.assertEqual(len(read_lines(path / "frames.jsonl")), frames)
        self.assertEqual(len(read_lines(path / "predictions.jsonl")), rows)
        self.assertTrue(self.exp.state()["recording"]["paused"])

        self.exp.pause_recording(False)
        self.assertLess(self.exp.study_time() - paused_at, .1)
        self.give_frames()
        self.exp.tick(self.look("notebook"))
        self.assertEqual(len(read_lines(path / "frames.jsonl")), frames + 2)
        predictions = read_lines(path / "predictions.jsonl")
        self.assertEqual(len(predictions), rows + 1)
        events = read_lines(path / "events.jsonl")
        self.assertEqual([e["kind"] for e in events], ["PAUSA", "RETOMADA"])
        self.assertLess(events[1]["t"] - events[0]["t"], .1)
        self.assertGreaterEqual(predictions[-1]["t"], events[1]["t"])

        self.exp.stop_recording()
        _, meta = self.session(session_id)
        self.assertEqual(len(meta["pauses"]), 1)
        self.assertGreaterEqual(meta["pauses"][0]["duration"], .3)
        self.assertGreaterEqual(meta["paused_duration"], .3)
        self.assertLess(meta["duration"], meta["paused_duration"] + .3)
        self.assertLess(meta["duration"], .3)

    def test_stop_while_paused_closes_the_pause(self):
        self.add_robot_map()
        self.give_frames()
        session_id = self.exp.start_recording("P013")
        self.exp.pause_recording(True)
        paused_at = self.exp.study_time()
        time.sleep(.05)
        self.exp.stop_recording()
        _, meta = self.session(session_id)
        self.assertAlmostEqual(meta["duration"], paused_at)
        self.assertIsNotNone(meta["pauses"][0]["duration"])
        with self.assertRaises(ValueError):
            self.exp.pause_recording(True)

    def test_a_bad_sample_is_skipped_without_stopping_the_study(self):
        self.add_robot_map()
        self.give_frames()
        session_id = self.exp.start_recording("P013")
        path, _ = self.session(session_id)
        start = self.look("tv")
        with patch.object(self.exp, "_position_row", side_effect=ValueError("leitura estranha")):
            self.exp.tick(start)  # não pode derrubar o processamento das câmeras
        self.assertIn("leitura estranha", self.exp.state()["error"])
        self.assertEqual(read_lines(path / "predictions.jsonl"), [])
        self.exp.tick(self.look("tv", start + .2))
        self.assertIsNone(self.exp.state()["error"])
        self.assertEqual(len(read_lines(path / "predictions.jsonl")), 1)
        self.exp.stop_recording()
        self.assertEqual(self.session(session_id)[1]["dropped_samples"], 1)

    def test_panel_routes_start_pause_resume_and_finish(self):
        self.add_robot_map()
        app = create_app(self.directory.name, "test-only", start_worker=False)
        exp = app.extensions["experiment"]
        exp.screen_gaze = self.exp.screen_gaze
        exp.nodes, exp.assignments = self.exp.nodes, self.exp.assignments
        self.give_frames(exp)
        client = app.test_client()
        client.post("/api/login", json={"key": "test-only"})
        try:
            response = client.post("/api/recording/start", json={"participant": "P013"})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(client.post("/api/recording/pause", json={"paused": True}).status_code, 200)
            recording = client.get("/api/state").json["recording"]
            self.assertEqual((recording["paused"], recording["participant"], recording["source"]),
                             (True, "P013", "posição"))
            live = client.get("/api/gaze/live").json
            self.assertEqual((live["participant"], live["paused"]), ("P013", True))
            self.assertEqual(client.post("/api/recording/pause", json={"paused": False}).status_code, 200)
            self.assertFalse(client.get("/api/state").json["recording"]["paused"])
            self.assertEqual(client.post("/api/recording/stop").status_code, 200)
            self.assertIsNone(client.get("/api/state").json["recording"])
            unknown = client.post("/api/recording/start", json={"participant": "P999"})
            self.assertEqual(unknown.status_code, 400)
            self.assertIn("calibração", unknown.json["error"])
        finally:
            exp.close()


class StoryNotificationTests(StudyFixture):
    """O Klara avisa que a história começou; o painel mostra o pop-up para iniciar o estudo."""

    def test_notification_shows_in_state_and_becomes_a_milestone_while_recording(self):
        item = self.exp.notify("historia_iniciada", "Usuário iniciou a história", {"jornada": "Jesuíno Brilhante", "cena": 1})
        shown = self.exp.state()["notifications"]
        self.assertEqual([(n["id"], n["message"], n["session"]) for n in shown], [(item["id"], "Usuário iniciou a história", None)])
        self.assertEqual(shown[0]["details"], {"jornada": "Jesuíno Brilhante", "cena": "1"})
        self.assertNotIn("at", shown[0])
        self.assertLess(shown[0]["age"], 1)
        self.assertIn("P013", self.exp.state()["screen_gaze"]["participants"])

        self.add_robot_map()
        self.give_frames()
        session_id = self.exp.start_recording("P013")
        later = self.exp.notify("historia_iniciada", "Usuário iniciou a história", {"jornada": "Missão ENEM"})
        self.assertEqual(later["session"], session_id)
        path, _ = self.session(session_id)
        events = read_lines(path / "events.jsonl")
        self.assertEqual([e["kind"] for e in events], ["HISTORIA_INICIADA"])
        self.assertIn("Missão ENEM", events[0]["note"])
        self.exp.stop_recording()

        # O painel só mostra avisos recentes.
        for entry in self.exp.notifications:
            entry["at"] -= 601
        self.assertEqual(self.exp.state()["notifications"], [])

    def test_invalid_notifications_are_refused(self):
        for kind, message, details in [("História!", "oi", None), ("historia_iniciada", "", None),
                                       ("historia_iniciada", "x" * 201, None), ("historia_iniciada", "oi", {"ok": True}),
                                       ("historia_iniciada", "oi", {"lista": [1]}), ("historia_iniciada", "oi", "texto")]:
            with self.assertRaises(ValueError, msg=(kind, message, details)):
                self.exp.notify(kind, message, details)
        self.assertEqual(self.exp.state()["notifications"], [])

    def test_route_accepts_the_server_key_without_a_browser_login(self):
        app = create_app(self.directory.name, "test-only", start_worker=False)
        exp = app.extensions["experiment"]
        client = app.test_client()
        payload = {"kind": "historia_iniciada", "message": "Usuário iniciou a história", "details": {"jornada": "Jesuíno Brilhante"}}
        try:
            self.assertEqual(client.post("/api/notifications", json=payload).status_code, 401)
            self.assertEqual(client.post("/api/notifications", json=payload, headers={"Authorization": "Bearer errada"}).status_code, 401)
            foreign = client.post("/api/notifications", json=payload,
                                  headers={"Authorization": "Bearer test-only", "Origin": "http://outro-site.example"})
            self.assertEqual(foreign.status_code, 403)
            response = client.post("/api/notifications", json=payload, headers={"Authorization": "Bearer test-only"})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertIsNone(response.json["session"])
            bad = client.post("/api/notifications", json={**payload, "kind": "x y"}, headers={"Authorization": "Bearer test-only"})
            self.assertEqual(bad.status_code, 400)
            client.post("/api/login", json={"key": "test-only"})
            shown = client.get("/api/state").json["notifications"]
            self.assertEqual([(n["kind"], n["details"]["jornada"]) for n in shown], [("historia_iniciada", "Jesuíno Brilhante")])
        finally:
            exp.close()


if __name__ == "__main__":
    unittest.main()
