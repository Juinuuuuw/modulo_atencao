"""Estados temporais derivados das observacoes multicamera.

Esta camada descreve comportamento observavel. Baixa atividade do olhar nao e
rotulada como distracao cognitiva sem validacao especifica no piloto.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


FOCUS_BY_TARGET = {
    "NOTEBOOK": "EM_FOCO_NOTEBOOK",
    "TV": "EM_FOCO_TV",
    "NAO": "EM_FOCO_ROBO",
}
EXTERNAL = "DISPERSAO_EXTERNA"
SLEEPY = "SLEEPY"
LOW_ACTIVITY = "BAIXA_ATIVIDADE_DO_OLHAR"
INDETERMINATE = "INDETERMINADO"
ATTENTION_STATES = (*FOCUS_BY_TARGET.values(), EXTERNAL, SLEEPY, LOW_ACTIVITY, INDETERMINATE)


@dataclass(frozen=True)
class AttentionConfig:
    external_delay_seconds: float = 2.0
    sleepy_delay_seconds: float = 2.0
    ear_absolute_threshold: float = 0.25
    ear_reference_ratio: float = 0.60
    gaze_activity_window_seconds: float = 5.0
    gaze_activity_fps: float = 3.0
    # Deve ser estimado no piloto. None calcula e registra V, mas nao infere estado.
    gaze_activity_threshold: float | None = None

    def export(self):
        return asdict(self)


class AttentionTimeline:
    """FSM com confirmacao temporal e inicio retroativo de eventos."""

    def __init__(self, config=None, ear_reference=None, roles=()):
        self.config = config or AttentionConfig()
        self.ear_reference = ear_reference
        personalized = (self.config.ear_reference_ratio * ear_reference
                        if ear_reference is not None and math.isfinite(ear_reference) else None)
        self.ear_threshold = min(self.config.ear_absolute_threshold, personalized) if personalized else self.config.ear_absolute_threshold
        self.roles = tuple(roles)
        self.state = None
        self.started_at = None
        self.pending_external = None
        self.pending_sleepy = None
        self.activity_samples = deque()
        self.last_activity_sample = None
        self.activity_target = None
        self.gaze_activity = None
        self.history = []

    def _activity(self, timestamp, gaze_point, target):
        if target != self.activity_target:
            self.activity_samples.clear()
            self.last_activity_sample = None
            self.gaze_activity = None
            self.activity_target = target
        if gaze_point is None or not all(math.isfinite(float(v)) for v in gaze_point):
            self.activity_samples.clear()
            self.last_activity_sample = None
            self.gaze_activity = None
            return None
        period = 1.0 / self.config.gaze_activity_fps
        if self.last_activity_sample is not None and timestamp - self.last_activity_sample + 1e-9 < period:
            return self.gaze_activity
        self.last_activity_sample = timestamp
        self.activity_samples.append((timestamp, float(gaze_point[0]), float(gaze_point[1])))
        cutoff = timestamp - self.config.gaze_activity_window_seconds
        while self.activity_samples and self.activity_samples[0][0] < cutoff:
            self.activity_samples.popleft()
        required = int(self.config.gaze_activity_window_seconds * self.config.gaze_activity_fps)
        if len(self.activity_samples) < required:
            self.gaze_activity = None
            return None
        values = np.asarray([(x, y) for _, x, y in self.activity_samples], dtype=float)
        center = values.mean(axis=0)
        self.gaze_activity = float(np.mean(np.sum((values - center) ** 2, axis=1)))
        return self.gaze_activity

    def _data_loss(self, start, end):
        relevant = [row for row in self.history if start <= row["t"] <= end]
        result = {}
        for role in self.roles:
            valid = sum(bool(row["tracking"].get(role)) for row in relevant)
            result[role] = round(100.0 * (1 - valid / len(relevant)), 3) if relevant else 100.0
        return result

    def _cameras(self, start, end):
        cameras = {camera for row in self.history if start <= row["t"] <= end
                   for camera in row["supporting_cameras"]}
        return sorted(cameras)

    def _close(self, end):
        if self.state is None or self.started_at is None or end <= self.started_at:
            return None
        duration = end - self.started_at
        return {
            "start": self.started_at,
            "end": end,
            "duration": duration,
            "state": self.state,
            "duration_class": "CURTO" if duration <= 15.0 else "LONGO",
            "supporting_cameras": self._cameras(self.started_at, end),
            "data_loss_percent": self._data_loss(self.started_at, end),
        }

    def _transition(self, state, boundary):
        if state == self.state:
            return []
        if self.started_at is not None:
            boundary = max(self.started_at, boundary)
        closed = self._close(boundary)
        self.state = state
        self.started_at = boundary
        return [closed] if closed else []

    def update(self, timestamp, target, ear=None, tracking=None, supporting_cameras=(), gaze_point=None):
        tracking = dict(tracking or {})
        self.history.append({"t": timestamp, "tracking": tracking,
                             "supporting_cameras": tuple(supporting_cameras)})
        activity = self._activity(timestamp, gaze_point if target in FOCUS_BY_TARGET else None, target)
        eyes_closed = ear is not None and math.isfinite(ear) and ear < self.ear_threshold

        self.pending_sleepy = (timestamp if self.pending_sleepy is None else self.pending_sleepy) if eyes_closed else None
        external = target == "OUTRO_LOCAL"
        self.pending_external = (timestamp if self.pending_external is None else self.pending_external) if external else None

        low_activity = (activity is not None and self.config.gaze_activity_threshold is not None
                        and activity < self.config.gaze_activity_threshold)
        if self.pending_sleepy is not None and timestamp - self.pending_sleepy >= self.config.sleepy_delay_seconds:
            desired, boundary = SLEEPY, self.pending_sleepy
        elif self.pending_external is not None and timestamp - self.pending_external >= self.config.external_delay_seconds:
            desired, boundary = EXTERNAL, self.pending_external
        elif low_activity:
            desired, boundary = LOW_ACTIVITY, timestamp
        elif target in FOCUS_BY_TARGET:
            desired, boundary = FOCUS_BY_TARGET[target], timestamp
        elif self.state in (EXTERNAL, SLEEPY) and (external or eyes_closed):
            desired, boundary = self.state, timestamp
        elif self.state is not None and (external or eyes_closed):
            # A condicao ainda esta dentro da janela de confirmacao.
            desired, boundary = self.state, timestamp
        else:
            desired, boundary = INDETERMINATE, timestamp

        events = self._transition(desired, boundary)
        return events, {
            "state": self.state,
            "since": self.started_at,
            "ear": ear,
            "ear_threshold": self.ear_threshold,
            "eyes_closed": eyes_closed,
            "gaze_activity": activity,
            "gaze_activity_threshold": self.config.gaze_activity_threshold,
        }

    def close(self, timestamp):
        event = self._close(timestamp)
        self.state = None
        self.started_at = None
        return [event] if event else []

