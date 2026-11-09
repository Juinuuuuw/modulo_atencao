"""Estabilidade temporal das leituras ao vivo.

Nas calibrações salvas (P001, P002, P005, P011), olhando um ponto fixo da validação, o
ponto estimado quadro a quadro saltava em média 0,077 da tela entre quadros seguidos.
Mediana de 0,25 s seguida de média exponencial de 0,1 s reduz o salto para ~0,014, sem
mudar o erro médio (que é sobretudo viés do mapa, não ruído). O atraso típico é ~0,2 s.

As calibrações continuam usando as leituras brutas: só a estimativa ao vivo é filtrada.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from .target_classifier import UNKNOWN


class FeatureSmoother:
    """Filtra íris, cabeça e pálpebras (gx, gy, hx, hy, EAR) de uma câmera.

    A mediana descarta quadros isolados ruins sem borrar o salto do olhar; a média
    exponencial tira o tremor que sobra. Piscadas não entram no filtro, e uma lacuna
    maior que `gap` (rosto perdido, olhos fechados) recomeça do zero.
    """

    def __init__(self, window=.25, tau=.10, gap=.5, blink_ratio=.6):
        self.window, self.tau, self.gap, self.blink_ratio = window, tau, gap, blink_ratio
        self.reset()

    def reset(self):
        self.buffer = deque()
        self.state = None
        self.last = None

    def update(self, t, values):
        """`values` = características + EAR do quadro, ou None sem leitura válida."""
        x = None if values is None else np.asarray(values, dtype=float)
        if x is None or not np.isfinite(x).all():
            if self.last is not None and t - self.last > self.gap:
                self.reset()
            return None
        if self.last is not None and (t - self.last > self.gap or t < self.last):
            self.reset()
        if len(self.buffer) >= 3 and x[-1] < self.blink_ratio * np.median([v[-1] for _, v in self.buffer]):
            # Piscada: íris e pálpebras saem do lugar por 1–3 quadros; mantém o valor anterior.
            return None if self.state is None else self.state.tolist()
        self.buffer.append((t, x))
        while self.buffer and t - self.buffer[0][0] > self.window:
            self.buffer.popleft()
        median = np.median([v for _, v in self.buffer], axis=0)
        if self.state is None:
            self.state = median
        else:
            alpha = 1 - math.exp(-max(t - self.last, 0.) / self.tau)
            self.state = self.state + alpha * (median - self.state)
        self.last = t
        return self.state.tolist()


class Hysteresis:
    """Só troca o valor mostrado quando o novo se mantém por `dwell` s.

    Perder a leitura (valor `unknown`: piscada, câmeras discordando por um instante) só
    apaga o valor anterior depois de `hold` s. Sem atualização por `stale` s, recomeça.
    """

    def __init__(self, dwell=.35, hold=.8, stale=2., unknown=UNKNOWN):
        self.dwell, self.hold, self.stale, self.unknown = dwell, hold, stale, unknown
        self.shown = self.candidate = None
        self.since = self.shown_since = self.updated = None

    def update(self, now, value):
        if self.updated is None or now - self.updated > self.stale or now < self.updated:
            self.shown = self.candidate = value
            self.since = self.shown_since = now
        else:
            if value != self.candidate:
                self.candidate, self.since = value, now
            wait = self.hold if value == self.unknown else self.dwell
            if value != self.shown and now - self.since >= wait:
                self.shown, self.shown_since = value, now
        self.updated = now
        return self.shown


class StableTarget:
    """Alvo e parte do robô mostrados ao operador, com histerese."""

    def __init__(self, dwell=.35, hold=.8, stale=2.):
        self.target = Hysteresis(dwell, hold, stale, UNKNOWN)
        self.part = Hysteresis(dwell, hold, stale, None)

    def update(self, now, target, part=None):
        shown = self.target.update(now, target)
        shown_part = self.part.update(now, part if target == "NAO" else None)
        pending = self.target.candidate if self.target.candidate != shown else None
        return {"target": shown, "part": shown_part if shown == "NAO" else None,
                "pending": pending, "held_s": round(now - self.target.shown_since, 2)}
