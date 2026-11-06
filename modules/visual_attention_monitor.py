"""Nucleo do monitor visual de atencao baseado em regras.

As classes deste modulo nao dependem da interface grafica. Isso permite testar a
calibracao e os limiares temporais sem uma webcam conectada.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import csv
import json
import math
from pathlib import Path
from typing import Deque, Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np


class MonitorState(str, Enum):
    EM_FOCO = "EM_FOCO"
    DISTRACTION_GAZE = "DISTRACTION_GAZE"
    DISTRACTION_COGNITIVE = "DISTRACTION_COGNITIVE"
    SLEEPY = "SLEEPY"


@dataclass(frozen=True)
class MonitorConfig:
    ear_threshold: float = 0.25
    event_delay_seconds: float = 2.0
    cognitive_window_seconds: float = 5.0
    cognitive_sample_fps: float = 3.0
    cognitive_variance_threshold: float = 0.0004
    head_yaw_tolerance: float = 0.18
    head_pitch_tolerance: float = 0.20
    depth_change_threshold: float = 0.05
    posture_compensation: float = 0.15


@dataclass(frozen=True)
class VisualObservation:
    detected: bool
    timestamp: float
    ear: float | None = None
    gaze_feature_x: float | None = None
    gaze_feature_y: float | None = None
    glabella: tuple[float, float] | None = None
    iris_diameter: float | None = None
    head_x: float | None = None
    head_y: float | None = None
    eye_centers: tuple[tuple[float, float], tuple[float, float]] | None = None
    gaze_direction: tuple[float, float] | None = None
    face_count: int = 1


@dataclass(frozen=True)
class GazePrediction:
    x: float
    y: float
    on_task: bool
    head_on_task: bool
    depth_scale: float


@dataclass(frozen=True)
class EventRecord:
    participant_id: str
    timestamp_inicio: str
    timestamp_fim: str
    duracao_segundos: float
    tipo_evento: str
    porcentagem_perda_quadros_no_intervalo: float
    classificacao_duracao: str


class LinearAxisModel:
    """Regressao linear univariada, sem dependencia de scikit-learn."""

    def __init__(self) -> None:
        self.slope = 0.0
        self.intercept = 0.0
        self.fitted = False

    def fit(self, features: Sequence[float], targets: Sequence[float]) -> None:
        if len(features) < 2 or len(features) != len(targets):
            raise ValueError("A regressao requer ao menos duas amostras pareadas")
        design = np.column_stack((np.asarray(features, dtype=float), np.ones(len(features))))
        self.slope, self.intercept = np.linalg.lstsq(
            design, np.asarray(targets, dtype=float), rcond=None
        )[0]
        self.fitted = True

    def predict(self, feature: float) -> float:
        if not self.fitted:
            raise RuntimeError("Modelo de olhar ainda nao calibrado")
        return float(self.slope * feature + self.intercept)


class FivePointCalibrator:
    """Coleta cinco fixacoes de 3 s, descartando os 0,5 s iniciais."""

    FIXATION_SECONDS = 3.0
    SETTLING_SECONDS = 0.5

    def __init__(self, screen_width: int, screen_height: int, margin: int = 45) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.points = (
            (margin, margin),
            (screen_width - margin, margin),
            (screen_width - margin, screen_height - margin),
            (margin, screen_height - margin),
            (screen_width // 2, screen_height // 2),
        )
        self.started_at: float | None = None
        self.samples: list[list[VisualObservation]] = [[] for _ in self.points]
        self.model_x = LinearAxisModel()
        self.model_y = LinearAxisModel()
        self.reference_glabella: tuple[float, float] | None = None
        self.reference_iris_diameter: float | None = None
        self.reference_head: tuple[float, float] | None = None

    def start(self, timestamp: float) -> None:
        self.started_at = timestamp

    def point_index(self, timestamp: float) -> int:
        if self.started_at is None:
            raise RuntimeError("Calibracao nao iniciada")
        return min(int((timestamp - self.started_at) // self.FIXATION_SECONDS), 4)

    def is_complete(self, timestamp: float) -> bool:
        return self.started_at is not None and timestamp - self.started_at >= 15.0

    def add(self, observation: VisualObservation) -> None:
        if self.started_at is None or self.is_complete(observation.timestamp):
            return
        elapsed = observation.timestamp - self.started_at
        within_fixation = elapsed % self.FIXATION_SECONDS
        if within_fixation < self.SETTLING_SECONDS or not observation.detected:
            return
        if observation.gaze_feature_x is None or observation.gaze_feature_y is None:
            return
        self.samples[self.point_index(observation.timestamp)].append(observation)

    @property
    def progress(self) -> float:
        if self.started_at is None:
            return 0.0
        latest = max((s.timestamp for group in self.samples for s in group), default=self.started_at)
        return min(1.0, (latest - self.started_at) / 15.0)

    def finish(self) -> None:
        if any(not group for group in self.samples):
            missing = [str(i + 1) for i, group in enumerate(self.samples) if not group]
            raise RuntimeError("Sem deteccao valida nos pontos: " + ", ".join(missing))

        x_features: list[float] = []
        y_features: list[float] = []
        x_targets: list[float] = []
        y_targets: list[float] = []
        for point, group in zip(self.points, self.samples):
            for sample in group:
                x_features.append(float(sample.gaze_feature_x))
                y_features.append(float(sample.gaze_feature_y))
                x_targets.append(float(point[0]))
                y_targets.append(float(point[1]))
        self.model_x.fit(x_features, x_targets)
        self.model_y.fit(y_features, y_targets)

        center = self.samples[4]
        self.reference_glabella = tuple(
            float(np.median([sample.glabella[axis] for sample in center if sample.glabella]))
            for axis in (0, 1)
        )
        self.reference_iris_diameter = float(
            np.median([sample.iris_diameter for sample in center if sample.iris_diameter])
        )
        self.reference_head = tuple(
            float(np.median([getattr(sample, f"head_{axis}") for sample in center]))
            for axis in ("x", "y")
        )

    @property
    def calibrated(self) -> bool:
        return self.model_x.fitted and self.model_y.fitted

    def predict(
        self,
        observation: VisualObservation,
        aoi: tuple[float, float, float, float],
        config: MonitorConfig,
    ) -> GazePrediction:
        if not self.calibrated or observation.gaze_feature_x is None or observation.gaze_feature_y is None:
            raise RuntimeError("Predicao solicitada antes da calibracao")
        x = self.model_x.predict(observation.gaze_feature_x)
        y = self.model_y.predict(observation.gaze_feature_y)

        # Compensa translacao da cabeca, limitada a 15% de cada dimensao da tela.
        if observation.glabella and self.reference_glabella:
            dx = (observation.glabella[0] - self.reference_glabella[0]) / self.screen_width
            dy = (observation.glabella[1] - self.reference_glabella[1]) / self.screen_height
            correction_x = np.clip(dx, -config.posture_compensation, config.posture_compensation)
            correction_y = np.clip(dy, -config.posture_compensation, config.posture_compensation)
            x -= float(correction_x) * self.screen_width
            y -= float(correction_y) * self.screen_height

        depth_scale = 1.0
        if observation.iris_diameter and self.reference_iris_diameter:
            current = observation.iris_diameter
            reference = self.reference_iris_diameter
            relative_change = abs(reference - current) / max(reference, 1e-6)
            if relative_change >= config.depth_change_threshold:
                depth_scale = (current + abs(reference - current)) / current
                cx, cy = self.screen_width / 2, self.screen_height / 2
                x = cx + (x - cx) / depth_scale
                y = cy + (y - cy) / depth_scale

        left, top, right, bottom = aoi
        on_screen = 0 <= x < self.screen_width and 0 <= y < self.screen_height
        inside_aoi = left <= x <= right and top <= y <= bottom
        reference_head = self.reference_head or (0.5, 0.5)
        head_on_task = (
            observation.head_x is not None
            and observation.head_y is not None
            and abs(observation.head_x - reference_head[0]) <= config.head_yaw_tolerance
            and abs(observation.head_y - reference_head[1]) <= config.head_pitch_tolerance
        )
        return GazePrediction(x, y, on_screen and inside_aoi and head_on_task, head_on_task, depth_scale)


class CognitiveDistractionDetector:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.samples: Deque[tuple[float, float, float]] = deque()
        self.last_sample_at: float | None = None
        self.detected = False

    def update(self, timestamp: float, x_norm: float, y_norm: float, eligible: bool) -> tuple[bool, float | None]:
        if not eligible:
            self.samples.clear()
            self.last_sample_at = None
            self.detected = False
            return False, None
        period = 1.0 / self.config.cognitive_sample_fps
        if self.last_sample_at is not None and timestamp - self.last_sample_at + 1e-9 < period:
            return self.detected, None
        self.last_sample_at = timestamp
        self.samples.append((timestamp, x_norm, y_norm))
        cutoff = timestamp - self.config.cognitive_window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        required = int(self.config.cognitive_window_seconds * self.config.cognitive_sample_fps)
        if len(self.samples) < required:
            self.detected = False
            return False, None
        values = np.asarray([(x, y) for _, x, y in self.samples], dtype=float)
        variance = float(np.mean(np.sum((values - values.mean(axis=0)) ** 2, axis=1)))
        self.detected = variance < self.config.cognitive_variance_threshold
        return self.detected, variance


class EventLogger:
    FIELDS = tuple(EventRecord.__dataclass_fields__)

    def __init__(self, participant_id: str, output_dir: str | Path, session_started: float) -> None:
        self.participant_id = participant_id
        self.session_started = session_started
        self.wall_started = datetime.now(timezone.utc)
        stamp = self.wall_started.strftime("%Y%m%d_%H%M%S")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in participant_id)
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = directory / f"{safe_id}_{stamp}.csv"
        self.json_path = directory / f"{safe_id}_{stamp}.json"
        self.records: list[EventRecord] = []
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            csv.DictWriter(stream, fieldnames=self.FIELDS).writeheader()
        self.json_path.write_text("[]\n", encoding="utf-8")

    def _iso(self, monotonic_timestamp: float) -> str:
        moment = self.wall_started + timedelta(seconds=monotonic_timestamp - self.session_started)
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def append(self, start: float, end: float, event_type: MonitorState, total: int, lost: int) -> EventRecord:
        duration = max(0.0, end - start)
        record = EventRecord(
            participant_id=self.participant_id,
            timestamp_inicio=self._iso(start),
            timestamp_fim=self._iso(end),
            duracao_segundos=round(duration, 3),
            tipo_evento=event_type.value,
            porcentagem_perda_quadros_no_intervalo=round(100.0 * lost / max(total, 1), 3),
            classificacao_duracao="CURTO" if duration <= 15.0 else "LONGO",
        )
        self.records.append(record)
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            writer.writerow(asdict(record))
        self.json_path.write_text(
            json.dumps([asdict(item) for item in self.records], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return record


class AttentionStateMachine:
    """FSM deterministica com inicio retroativo para a regra dos 2 segundos."""

    def __init__(self, config: MonitorConfig, logger: EventLogger) -> None:
        self.config = config
        self.logger = logger
        self.state = MonitorState.EM_FOCO
        self.condition_started: dict[MonitorState, float | None] = {
            MonitorState.DISTRACTION_GAZE: None,
            MonitorState.SLEEPY: None,
        }
        self.active_started: float | None = None
        self.total_frames = 0
        self.lost_frames = 0
        self.active_count_snapshot = (0, 0)
        self.condition_count_snapshot: dict[MonitorState, tuple[int, int] | None] = {
            MonitorState.DISTRACTION_GAZE: None,
            MonitorState.SLEEPY: None,
        }

    def _qualified_state(self, timestamp: float, gaze_off: bool, sleepy: bool, cognitive: bool) -> tuple[MonitorState, float]:
        conditions = {
            MonitorState.DISTRACTION_GAZE: gaze_off,
            MonitorState.SLEEPY: sleepy,
        }
        for state, condition in conditions.items():
            if condition and self.condition_started[state] is None:
                self.condition_started[state] = timestamp
                self.condition_count_snapshot[state] = (self.total_frames - 1, self.lost_frames)
            elif not condition:
                self.condition_started[state] = None
                self.condition_count_snapshot[state] = None

        for state in (MonitorState.SLEEPY, MonitorState.DISTRACTION_GAZE):
            started = self.condition_started[state]
            if started is not None and timestamp - started >= self.config.event_delay_seconds:
                return state, started
        if cognitive:
            return MonitorState.DISTRACTION_COGNITIVE, timestamp
        return MonitorState.EM_FOCO, timestamp

    def update(self, timestamp: float, detected: bool, gaze_off: bool, eyes_closed: bool, cognitive: bool) -> EventRecord | None:
        self.total_frames += 1
        if not detected:
            self.lost_frames += 1
        # Quadros sem deteccao nao inventam uma mudanca comportamental.
        if not detected:
            if self.state is MonitorState.EM_FOCO:
                for pending in self.condition_started:
                    self.condition_started[pending] = None
                    self.condition_count_snapshot[pending] = None
            return None

        target, target_started = self._qualified_state(timestamp, gaze_off, eyes_closed, cognitive)
        if target == self.state:
            return None

        closed = None
        if self.state is not MonitorState.EM_FOCO and self.active_started is not None:
            total = self.total_frames - self.active_count_snapshot[0]
            lost = self.lost_frames - self.active_count_snapshot[1]
            closed = self.logger.append(
                self.active_started, timestamp, self.state, total, lost
            )
        self.state = target
        if target is MonitorState.EM_FOCO:
            self.active_started = None
        else:
            self.active_started = target_started if closed is None else timestamp
            snapshot = self.condition_count_snapshot.get(target)
            self.active_count_snapshot = snapshot or (self.total_frames - 1, self.lost_frames)
            if closed is not None:
                self.active_count_snapshot = (self.total_frames - 1, self.lost_frames)
        return closed

    def close(self, timestamp: float) -> EventRecord | None:
        if self.state is MonitorState.EM_FOCO or self.active_started is None:
            return None
        total = self.total_frames - self.active_count_snapshot[0]
        lost = self.lost_frames - self.active_count_snapshot[1]
        record = self.logger.append(
            self.active_started, timestamp, self.state, total, lost
        )
        self.state = MonitorState.EM_FOCO
        self.active_started = None
        return record


class FaceLandmarkExtractor:
    """Extrai EAR, iris, glabela e orientacao aproximada da cabeca."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    LEFT_IRIS = (468, 469, 470, 471, 472)
    RIGHT_IRIS = (473, 474, 475, 476, 477)

    def __init__(self) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @classmethod
    def _ear(cls, points: np.ndarray, indices: Sequence[int]) -> float:
        p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
        return (cls._distance(p2, p6) + cls._distance(p3, p5)) / max(
            2.0 * cls._distance(p1, p4), 1e-6
        )

    @staticmethod
    def _iris_feature(points: np.ndarray, iris: Sequence[int], corners: tuple[int, int], lids: tuple[tuple[int, int], tuple[int, int]]) -> tuple[float, float]:
        center = points[list(iris)].mean(axis=0)
        x_min, x_max = sorted((points[corners[0]][0], points[corners[1]][0]))
        top = points[list(lids[0]), 1].mean()
        bottom = points[list(lids[1]), 1].mean()
        y_min, y_max = sorted((top, bottom))
        return (
            float((center[0] - x_min) / max(x_max - x_min, 1e-6)),
            float((center[1] - y_min) / max(y_max - y_min, 1e-6)),
        )

    def extract(self, frame: np.ndarray, timestamp: float) -> VisualObservation:
        height, width = frame.shape[:2]
        result = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return VisualObservation(False, timestamp, face_count=0)
        if len(result.multi_face_landmarks) > 1:
            return VisualObservation(False, timestamp, face_count=len(result.multi_face_landmarks))
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.asarray([(p.x * width, p.y * height) for p in landmarks], dtype=float)
        left = self._iris_feature(points, self.LEFT_IRIS, (33, 133), ((159, 160), (145, 144)))
        right = self._iris_feature(points, self.RIGHT_IRIS, (362, 263), ((386, 385), (374, 380)))
        iris_diameter = (
            self._distance(points[469], points[471]) + self._distance(points[474], points[476])
        ) / 2.0
        face_left, face_right = sorted((points[234, 0], points[454, 0]))
        face_top, face_bottom = sorted((points[10, 1], points[152, 1]))
        head_x = float((points[1, 0] - face_left) / max(face_right - face_left, 1e-6))
        head_y = float((points[1, 1] - face_top) / max(face_bottom - face_top, 1e-6))
        # Vetor 2D ilustrativo no plano da imagem; não é um raio ocular 3D.
        direction = (
            float(np.clip(((left[0] + right[0]) / 2 - 0.5) * 2 + (head_x - 0.5) * 1.4, -1, 1)),
            float(np.clip(((left[1] + right[1]) / 2 - 0.5) * 1.2 + (head_y - 0.5), -1, 1)),
        )
        eye_centers = tuple(tuple(float(v) for v in points[list(indices)].mean(axis=0) / (width, height))
                            for indices in (self.LEFT_IRIS, self.RIGHT_IRIS))
        return VisualObservation(
            detected=True,
            timestamp=timestamp,
            ear=(self._ear(points, self.LEFT_EYE) + self._ear(points, self.RIGHT_EYE)) / 2.0,
            gaze_feature_x=(left[0] + right[0]) / 2.0,
            gaze_feature_y=(left[1] + right[1]) / 2.0,
            glabella=(float(points[168, 0]), float(points[168, 1])),
            iris_diameter=iris_diameter,
            head_x=head_x,
            head_y=head_y,
            eye_centers=eye_centers,
            gaze_direction=direction,
        )

    def close(self) -> None:
        self.mesh.close()
