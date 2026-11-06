import json
from pathlib import Path
import tempfile
import unittest

from modules.visual_attention_monitor import (
    AttentionStateMachine,
    CognitiveDistractionDetector,
    EventLogger,
    FivePointCalibrator,
    MonitorConfig,
    MonitorState,
    VisualObservation,
)


def observation(timestamp, x, y, ear=0.3):
    return VisualObservation(True, timestamp, ear, x, y, (500.0, 300.0), 12.0, 0.5, 0.5)


def test_five_point_calibration_uses_independent_axis_models():
    calibrator = FivePointCalibrator(1000, 800)
    calibrator.start(0.0)
    for index, (target_x, target_y) in enumerate(calibrator.points):
        for offset in (0.6, 1.5, 2.4):
            timestamp = index * 3.0 + offset
            calibrator.add(observation(timestamp, target_x / 1000, target_y / 800))
    calibrator.finish()
    predicted = calibrator.predict(observation(16.0, 0.5, 0.5), (0, 0, 1000, 800), MonitorConfig())
    assert abs(predicted.x - 500) < 2
    assert abs(predicted.y - 400) < 2


def test_gaze_event_is_backdated_by_two_seconds(tmp_path):
    logger = EventLogger("P01", tmp_path, 0.0)
    machine = AttentionStateMachine(MonitorConfig(), logger)
    machine.update(1.0, True, True, False, False)
    machine.update(3.0, True, True, False, False)
    assert machine.state is MonitorState.DISTRACTION_GAZE
    machine.update(4.0, True, False, False, False)
    assert logger.records[0].duracao_segundos == 3.0


def test_sleepy_has_priority_over_gaze(tmp_path):
    logger = EventLogger("P01", tmp_path, 0.0)
    machine = AttentionStateMachine(MonitorConfig(), logger)
    machine.update(0.0, True, True, True, False)
    machine.update(2.1, True, True, True, False)
    assert machine.state is MonitorState.SLEEPY


def test_cognitive_detector_uses_five_second_three_fps_window():
    config = MonitorConfig(cognitive_variance_threshold=0.001)
    detector = CognitiveDistractionDetector(config)
    detected = False
    for index in range(16):
        detected, _ = detector.update(index / 3, 0.5, 0.5, True)
    assert detected


def test_logger_writes_csv_and_json_and_data_loss(tmp_path):
    logger = EventLogger("P 01", tmp_path, 10.0)
    record = logger.append(11.0, 27.0, MonitorState.SLEEPY, total=20, lost=2)
    assert record.classificacao_duracao == "LONGO"
    assert record.porcentagem_perda_quadros_no_intervalo == 10.0
    content = json.loads(logger.json_path.read_text(encoding="utf-8"))
    assert content[0]["tipo_evento"] == "SLEEPY"


class VisualAttentionMonitorTests(unittest.TestCase):
    def test_calibration(self):
        test_five_point_calibration_uses_independent_axis_models()

    def test_gaze_event(self):
        with tempfile.TemporaryDirectory() as directory:
            test_gaze_event_is_backdated_by_two_seconds(Path(directory))

    def test_sleepy_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            test_sleepy_has_priority_over_gaze(Path(directory))

    def test_cognitive_window(self):
        test_cognitive_detector_uses_five_second_three_fps_window()

    def test_event_files(self):
        with tempfile.TemporaryDirectory() as directory:
            test_logger_writes_csv_and_json_and_data_loss(Path(directory))

    def test_data_loss_breaks_a_pending_gaze_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = EventLogger("P01", directory, 0.0)
            machine = AttentionStateMachine(MonitorConfig(), logger)
            machine.update(0.0, True, True, False, False)
            machine.update(1.0, False, False, False, False)
            machine.update(2.1, True, True, False, False)
            self.assertEqual(machine.state, MonitorState.EM_FOCO)
