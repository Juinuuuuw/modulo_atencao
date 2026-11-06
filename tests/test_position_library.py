"""Calibração de posição por câmera e participante, robô por partes e visão ao vivo."""
import tempfile
import time
import unittest

import numpy as np

from modules.multicamera import Experiment
from modules.screen_gaze import ROBOT_FIT, ROBOT_PARTS, ScreenGaze


def vector(point, gain=1.0):
    x, y = point
    return [.2 + .5 * gain * x, .2 + .5 * gain * y, .4 + .04 * x, .4 + .04 * y]


def calibrate(gaze, surface="notebook", role="notebook", participant="P001", source="pc/0",
              screen=None, include_check=True, layout=(1280, 720, 0, 0), gain=1.0, start_at=0.0,
              on_tick=None):
    """Simula o participante olhando cada indicação; devolve o instante final."""
    screen = screen or surface
    gaze.start(surface, role, participant, source, include_check=include_check, screen=screen)
    labels, observations, displays = {role: source}, {}, {}
    now = start_at
    for sequence in range(2000):
        now = start_at + sequence / 10
        cue = gaze.cue(now)
        if not cue["active"]:
            break
        observations[role] = {"capture": now, "sequence": sequence, "features": vector(cue["marker"], gain),
                              "uncertainty": .01, "pose": None, "ear": .3}
        displays["window"] = {"display": screen, "visible": True, "ready": True, "cue_id": cue["cue_id"],
                              "seen": now, "layout": list(layout)}
        if on_tick:
            on_tick(sequence, now, displays, cue)
        gaze.tick(now, observations, labels, displays)
    return now


class RobotCalibrationTests(unittest.TestCase):
    def test_robot_parts_are_calibrated_validated_and_recognised(self):
        gaze = ScreenGaze()
        now = calibrate(gaze, surface="robot", role="nao", screen="notebook")
        cue = gaze.cue(now)
        self.assertEqual(cue["phase"], "complete", cue["reason"])
        report = gaze.models["robot:nao"]["validation"]
        self.assertEqual((report["parts_correct"], report["parts_total"]), (6, 6))
        self.assertEqual(report["level"], "alta")
        observations = {"nao": {"capture": now, "sequence": 1, "features": vector((.5, .23)), "uncertainty": .01, "ear": .3}}
        position = gaze.position("robot:nao", now, observations, {"nao": "pc/0"}, {})
        self.assertTrue(position["visible"])
        self.assertEqual(position["part"], "head")
        observations["nao"]["features"] = vector((.61, .77))
        self.assertEqual(gaze.position("robot:nao", now, observations, {"nao": "pc/0"}, {})["part"], "left-leg")
        # Longe de qualquer parte: não inventa uma parte do robô.
        observations["nao"]["features"] = vector((.5, -.6))
        self.assertIsNone(gaze.position("robot:nao", now, observations, {"nao": "pc/0"}, {})["part"])

    def test_robot_cue_names_each_part_in_order(self):
        gaze = ScreenGaze()
        gaze.start("robot", "notebook", "P001", "pc/0", include_check=True, screen="notebook")
        self.assertEqual(gaze.cue(0)["part"], ROBOT_FIT[0])
        self.assertIn("cabeça", gaze.cue(0)["part_label"])
        self.assertEqual(gaze.cue(0)["total"], 12)


class LibraryTests(unittest.TestCase):
    def test_each_camera_keeps_its_own_map_on_the_same_screen(self):
        gaze = ScreenGaze()
        calibrate(gaze, role="notebook", source="pc/0")
        calibrate(gaze, role="validation", source="pc/1", gain=.7, start_at=500)
        self.assertEqual(sorted(gaze.models), ["notebook:notebook", "notebook:validation"])

    def test_maps_are_saved_per_participant_and_reloaded(self):
        with tempfile.TemporaryDirectory() as store:
            gaze = ScreenGaze(store)
            calibrate(gaze, participant="P003")
            calibrate(gaze, participant="P004", gain=.8, start_at=500)
            reloaded = ScreenGaze(store)
            self.assertEqual(reloaded.participants(), ["P003", "P004"])
            observations = {"notebook": {"capture": 1, "sequence": 1, "features": vector((.3, .6)), "uncertainty": .01, "ear": .3}}
            position = reloaded.position("notebook:notebook", 1, observations, {"notebook": "pc/0"}, {},
                                         require_view=False, participant="P003")
            np.testing.assert_allclose([position["x"], position["y"]], [.3, .6], atol=.01)
            # Outra câmera no mesmo papel não usa o mapa salvo.
            self.assertFalse(reloaded.position("notebook:notebook", 1, observations, {"notebook": "outro/0"}, {},
                                               require_view=False, participant="P003")["visible"])

    def test_window_change_mid_calibration_waits_instead_of_failing(self):
        gaze = ScreenGaze()
        state = {"moved": None}

        def move(sequence, now, displays, cue):
            # No 5º ponto a janela sai da tela cheia por 3 s e depois volta.
            if cue["index"] == 4 and state["moved"] is None:
                state["moved"] = now
            if state["moved"] is not None and now - state["moved"] < 3:
                displays["window"]["layout"] = [1000, 600, 40, 40]

        now = calibrate(gaze, on_tick=move)
        self.assertIsNotNone(state["moved"])
        self.assertEqual(gaze.cue(now)["phase"], "complete", gaze.cue(now)["reason"])

    def test_second_target_window_does_not_break_calibration(self):
        gaze = ScreenGaze()

        def second_tab(sequence, now, displays, cue):
            if sequence > 30:
                displays["other-tab"] = {**displays["window"], "layout": [800, 600, 0, 0], "seen": now}

        now = calibrate(gaze, on_tick=second_tab)
        self.assertEqual(gaze.cue(now)["phase"], "complete", gaze.cue(now)["reason"])

    def test_validation_runs_automatically_and_hides_nothing_after(self):
        gaze = ScreenGaze()
        now = calibrate(gaze)
        validation = gaze.models["notebook:notebook"]["validation"]
        self.assertIn(validation["level"], ("alta", "moderada", "baixa"))
        self.assertIn("Validação", gaze.cue(now)["reason"])


class ZoneTests(unittest.TestCase):
    """Caso do P005: pela câmera do notebook a íris é quase igual olhando a tela ou o robô;
    o que muda é a pose da cabeça."""

    @staticmethod
    def reading(point, head, rng):
        x, y = point
        return [.45 + .1 * x + rng.normal(0, .004), .42 + .05 * y + rng.normal(0, .004),
                head[0] + rng.normal(0, .01), head[1] + rng.normal(0, .004)]

    def calibrate_both(self, robot_noise=0.0):
        rng = np.random.default_rng(7)
        gaze = ScreenGaze()
        screen_head, robot_head = (.68, .55), (.57, .50)

        def run(surface, head, noise, start_at):
            def on_tick(sequence, now, displays, cue, observations=None):
                pass
            gaze.start(surface, "notebook", "P005", "pc/0", include_check=True, screen="notebook")
            obs, displays = {}, {}
            for sequence in range(2000):
                now = start_at + sequence / 10
                cue = gaze.cue(now)
                if not cue["active"]:
                    break
                point = cue["marker"] if rng.random() >= noise else (rng.random(), rng.random())
                obs["notebook"] = {"capture": now, "sequence": sequence, "features": self.reading(point, head, rng),
                                   "uncertainty": .01, "ear": .28}
                displays["w"] = {"display": "notebook", "visible": True, "ready": True, "cue_id": cue["cue_id"],
                                 "seen": now, "layout": [1536, 864, 7, 7]}
                gaze.tick(now, obs, {"notebook": "pc/0"}, displays)
            return now

        run("notebook", screen_head, 0, 0)
        now = run("robot", robot_head, robot_noise, 500)
        return gaze, now, rng, screen_head, robot_head

    def test_head_pose_separates_robot_from_screen_for_the_same_camera(self):
        gaze, now, rng, screen_head, robot_head = self.calibrate_both()
        self.assertIsNotNone(gaze.models["robot:notebook"]["zone"])
        labels = {"notebook": "pc/0"}
        hits = {"tela": 0, "robô": 0}
        for _ in range(40):
            point = (rng.random(), rng.random())
            obs = {"notebook": {"capture": now, "sequence": 1, "features": self.reading(point, screen_head, rng), "uncertainty": .01, "ear": .28}}
            on_robot = gaze.position("robot:notebook", now, obs, labels, {}, require_view=False)
            on_screen = gaze.position("notebook:notebook", now, obs, labels, {}, require_view=False)
            hits["tela"] += bool(on_screen.get("inside")) and not on_robot.get("inside")
            part = list(ROBOT_PARTS.values())[int(rng.integers(len(ROBOT_PARTS)))]
            obs["notebook"]["features"] = self.reading(part, robot_head, rng)
            on_robot = gaze.position("robot:notebook", now, obs, labels, {}, require_view=False)
            on_screen = gaze.position("notebook:notebook", now, obs, labels, {}, require_view=False)
            hits["robô"] += bool(on_robot.get("inside")) and not on_screen.get("inside")
        # Antes: olhando a tela, o robô aparecia como olhado quase sempre.
        self.assertGreaterEqual(hits["tela"], 36)
        self.assertGreaterEqual(hits["robô"], 36)

    def test_robot_part_is_hidden_when_validation_could_not_tell_parts_apart(self):
        gaze, now, rng, _, robot_head = self.calibrate_both()
        model = gaze.models["robot:notebook"]
        model["validation"] = {**model["validation"], "part_accuracy": 2 / 6, "level": "baixa"}
        obs = {"notebook": {"capture": now, "sequence": 1, "features": self.reading((.5, .23), robot_head, rng),
                            "uncertainty": .01, "ear": .28}}
        position = gaze.position("robot:notebook", now, obs, {"notebook": "pc/0"}, {}, require_view=False)
        self.assertTrue(position["inside"])
        self.assertIsNone(position["part"])
        self.assertFalse(position["part_reliable"])

    def test_zone_survives_saving_and_old_maps_without_zone_still_work(self):
        with tempfile.TemporaryDirectory() as store:
            gaze = ScreenGaze(store)
            calibrate(gaze, participant="P010")
            reloaded = ScreenGaze(store)
            model = reloaded.models_for("P010")["notebook:notebook"]
            self.assertIsNotNone(model["zone"])
            model["zone"] = None  # mapa salvo antes desta versão
            obs = {"notebook": {"capture": 1, "sequence": 1, "features": vector((.3, .6)), "uncertainty": .01, "ear": .3}}
            position = reloaded.position("notebook:notebook", 1, obs, {"notebook": "pc/0"}, {}, require_view=False,
                                         participant="P010")
            self.assertTrue(position["inside"])
            self.assertIsNone(position["zone_ratio"])


class LiveViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.exp = Experiment(self.directory.name)
        self.exp.nodes = {"n1": {"name": "pc", "cameras": [{"id": "0"}, {"id": "1"}], "seen": time.perf_counter()},
                          "n2": {"name": "NAO", "cameras": [{"id": "0"}], "seen": time.perf_counter()}}
        self.exp.assignments = {"notebook": "n1:0", "nao": "n2:0", "validation": "n1:1"}

    def tearDown(self):
        self.directory.cleanup()

    def observe(self, now, points):
        for role, point in points.items():
            self.exp.observations[role] = {"capture": now, "sequence": now, "features": vector(point),
                                           "uncertainty": .01, "ear": .3}

    def test_cameras_vote_for_the_screen_and_robot_part(self):
        gaze = self.exp.screen_gaze
        calibrate(gaze, surface="notebook", role="notebook", participant="P009", source="pc/0")
        calibrate(gaze, surface="notebook", role="validation", participant="P009", source="pc/1", start_at=500)
        calibrate(gaze, surface="robot", role="nao", participant="P009", source="NAO/0", screen="notebook", start_at=1000)
        now = time.perf_counter()
        self.observe(now, {"notebook": (.4, .5), "validation": (.45, .55), "nao": (.5, -.8)})
        data = self.exp.live_gaze(now)
        self.assertEqual(data["participant"], "P009")
        self.assertEqual(data["source"], "posição")
        self.assertEqual(data["target"], "NOTEBOOK")
        self.assertEqual(len(data["screens"]["notebook"]["estimates"]), 2)
        # Só o NAO tem mapa do robô e vê a cabeça; as outras veem fora do notebook.
        # Fora do notebook não contradiz o robô: elas não calibraram o robô.
        self.observe(now, {"notebook": (.5, -2), "validation": (.5, -2), "nao": (.5, .23)})
        data = self.exp.live_gaze(now)
        self.assertEqual(data["target"], "NAO")
        self.assertEqual(data["robot"]["part"], "head")
        # Todas as câmeras com leitura válida e fora de tudo: outro local.
        self.observe(now, {"notebook": (.5, -2), "validation": (.5, -2), "nao": (.5, -.8)})
        self.assertEqual(self.exp.live_gaze(now)["target"], "OUTRO_LOCAL")
        # Uma câmera no notebook e outra contra: empate, não decide.
        self.observe(now, {"notebook": (.5, .5), "validation": (.5, -2), "nao": (.5, -.8)})
        self.assertEqual(self.exp.live_gaze(now)["target"], "INDETERMINADO")

    def test_participant_is_chosen_in_the_live_view(self):
        gaze = self.exp.screen_gaze
        calibrate(gaze, participant="P001", source="pc/0")
        calibrate(gaze, participant="P002", source="pc/0", start_at=500)
        now = time.perf_counter()
        self.observe(now, {"notebook": (.5, .5)})
        self.assertEqual(self.exp.live_gaze(now)["participant"], "P002")
        data = self.exp.live_gaze(now, participant="P001")
        self.assertEqual(data["participant"], "P001")
        self.assertEqual(data["participants"], ["P001", "P002"])
        self.assertEqual(data["screens"]["notebook"]["estimates"][0]["participant"], "P001")

    def test_camera_label_survives_client_reconnection(self):
        self.assertEqual(self.exp.camera_label("validation"), "pc/1")
        self.exp.nodes["n3"] = self.exp.nodes.pop("n1")
        self.exp.assignments["validation"] = "n3:1"
        self.assertEqual(self.exp.camera_label("validation"), "pc/1")


if __name__ == "__main__":
    unittest.main()
