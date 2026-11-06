"""Mapa empírico de características faciais para uma tela específica.

Grade de nove pontos e regressão ponderada com a posição da íris e a abertura
das pálpebras (EAR), que acompanha o olhar vertical melhor que a íris numa webcam
acima da tela. O conjunto de características e a regularização são escolhidos
por validação deixando um ponto de fora (cada ponto previsto sem ele).
A pose da cabeça não entra: em dados reais (P005) ela piorou a previsão em
pontos de teste, porque a pessoa se reacomoda depois da calibração.
Os pontos de verificação são coletados depois do ajuste e não treinam o mapa.
Não estima um raio ocular físico nem substitui eye tracking dedicado.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid
import numpy as np

# Grade 3×3 percorrida em "serpentina" para reduzir saltos do olhar. A linha do
# meio cobre a metade inferior, onde a webcam acima da tela vê pior a íris.
# Margens verticais evitam sobrepor o cabeçalho e o rodapé da página de alvo.
FIT_POINTS = ((.05,.08),(.5,.08),(.95,.08),
              (.95,.48),(.5,.48),(.05,.48),
              (.05,.88),(.5,.88),(.95,.88))
CHECK_POINTS = ((.25,.25),(.75,.30),(.35,.80),(.72,.70))
N_FIT = len(FIT_POINTS)
N_TOTAL = N_FIT + len(CHECK_POINTS)
MIN_SAMPLES = 6
# Só descarta piscadas evidentes na coleta. Olhar para baixo fecha parcialmente
# as pálpebras; esse efeito vira característica do modelo (EAR), não descarte.
CALIBRATION_MIN_EAR = .12

# Candidatos em ordem de complexidade. Um modelo mais complexo só é escolhido
# quando reduz o erro deixando cada ponto de fora em pelo menos 10% e 5 px.
CANDIDATES = (
    ("iris", ("gx","gy")),
    ("iris_palpebras", ("gx","gy","ear")),
    ("iris_palpebras_curva", ("gx","gy","ear","gx2","gy2")),
)
CANDIDATE_LABELS = {"iris":"íris","iris_palpebras":"íris + pálpebras","iris_palpebras_curva":"íris (curva) + pálpebras",
                    "cabeca":"cabeça","cabeca_palpebras":"cabeça + pálpebras","iris_cabeca":"íris + cabeça"}
RIDGES = (1e-3, 1e-2, 1e-1)
DEFAULT_EAR = .3


def eyes_open(observation):
    ear = observation.get("ear")
    return ear is None or ear >= CALIBRATION_MIN_EAR


def screen_vector(observation):
    return observation.get("screen_features", observation.get("features"))


def screen_reason(observation):
    return observation.get("screen_quality_reason", observation.get("quality_reason"))


def same_layout(left,right):
    # Bordas do Windows podem deslocar a origem 8 px na transição de tela cheia.
    # Isso não é uma troca de monitor nem uma nova dimensão da apresentação.
    return bool(left and right and len(left)==len(right)==4 and
                all(abs(left[i]-right[i]) <= (2 if i<2 else 16) for i in range(4)))


def _columns(features, ear):
    f = np.atleast_2d(np.asarray(features, dtype=float))
    gx, gy, hx, hy = f[:,0], f[:,1], f[:,2], f[:,3]
    return {"gx":gx,"gy":gy,"hx":hx,"hy":hy,"ear":np.asarray(ear,dtype=float).reshape(-1),
            "gx2":gx*gx,"gy2":gy*gy,"gxgy":gx*gy}


def _clean(rows):
    """Remove piscadas relativas ao próprio ponto e leituras fora da curva."""
    features = np.asarray([r["features"] for r in rows], dtype=float)
    ear = np.asarray([r.get("ear") if r.get("ear") is not None else DEFAULT_EAR for r in rows], dtype=float)
    keep = np.isfinite(features).all(axis=1) & np.isfinite(ear)
    keep &= ear >= .75 * np.median(ear[keep]) if keep.any() else keep
    for axis in (0, 1):
        values = features[keep, axis]
        if not len(values):
            break
        median = np.median(values)
        spread = max(1.4826 * np.median(np.abs(values - median)), 1e-3)
        keep &= np.abs(features[:, axis] - median) <= 3.5 * spread
    return features[keep], ear[keep]


class ScreenMap:

    def __init__(self):
        self.data = None
        self.model = None

    @staticmethod
    def _matrix(names, features, ear, mean, scale):
        columns = _columns(features, ear)
        return (np.column_stack([columns[n] for n in names]) - mean) / scale

    @staticmethod
    def _solve(x, y, weights, ridge):
        design = np.column_stack([np.ones(len(x)), x])
        weighted = design * weights[:, None]
        regularizer = ridge * weights.sum() * np.eye(design.shape[1])
        regularizer[0, 0] = 0
        return np.linalg.solve(design.T @ weighted + regularizer, weighted.T @ y)

    @staticmethod
    def _apply(x, beta):
        return np.column_stack([np.ones(len(x)), x]) @ beta

    @classmethod
    def from_data(cls, data):
        """Recria um mapa salvo (biblioteca por participante)."""
        item = cls()
        item.data = data
        item.model = {"name": data["model"], "features": list(data["features"]),
                      "mean": np.asarray(data["mean"], dtype=float), "scale": np.asarray(data["scale"], dtype=float),
                      "beta": np.asarray(data["beta"], dtype=float)}
        return item

    def fit(self, samples, layout=None, points=FIT_POINTS, candidates=CANDIDATES):
        width, height = (layout or [1920,1080])[:2]
        points = tuple(tuple(p) for p in points)
        features, ears, targets, steps = [], [], [], []
        counts = []
        for i, point in enumerate(points):
            rows = [r for r in samples if r["step"] == i]
            if len(rows) < MIN_SAMPLES:
                raise ValueError(f"Ponto {i+1}: são necessárias {MIN_SAMPLES} leituras válidas")
            f, e = _clean(rows)
            if len(f) < MIN_SAMPLES:
                raise ValueError(f"Ponto {i+1}: leituras instáveis (piscadas ou movimento). Repita mantendo o olhar fixo.")
            counts.append(len(f))
            features.append(f); ears.append(e)
            targets.append(np.tile(point, (len(f), 1))); steps.append(np.full(len(f), i))
        centers = np.asarray([np.median(f, axis=0) for f in features])
        if not np.isfinite(centers).all() or np.any(np.ptp(centers[:, :2], axis=0) < 1e-6):
            raise ValueError("A câmera não mediu variação dos olhos nas duas direções. Confira os olhos na prévia.")
        F = np.vstack(features); E = np.concatenate(ears); T = np.vstack(targets); S = np.concatenate(steps)
        weights = 1.0 / np.asarray(counts, dtype=float)[S]
        pixels = np.asarray([width, height], dtype=float)
        results = []
        for name, names in candidates:
            raw = np.column_stack([_columns(F, E)[n] for n in names])
            mean, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
            x = (raw - mean) / scale
            best = None
            for ridge in RIDGES:
                errors = []
                for k in range(len(points)):
                    train = S != k
                    beta = self._solve(x[train], T[train], weights[train], ridge)
                    predicted = np.median(self._apply(x[~train], beta), axis=0)
                    errors.append(float(np.linalg.norm((predicted - points[k]) * pixels)))
                if best is None or np.mean(errors) < best[0]:
                    best = (float(np.mean(errors)), errors, ridge)
            beta = self._solve(x, T, weights, best[2])
            fitted = [float(np.linalg.norm((np.median(self._apply(x[S == k], beta), axis=0) - points[k]) * pixels))
                      for k in range(len(points))]
            results.append({"name": name, "features": list(names), "mean": mean, "scale": scale, "beta": beta,
                            "ridge": best[2], "loo_error_px": best[0], "loo_point_errors_px": best[1],
                            "fit_error_px": float(np.mean(fitted))})
        chosen = results[0]
        for candidate in results[1:]:
            # Mais complexidade só por ganho real: ≥10% e ≥5 px no erro deixando cada ponto de fora.
            if candidate["loo_error_px"] < min(.9 * chosen["loo_error_px"], chosen["loo_error_px"] - 5):
                chosen = candidate
        self.model = chosen
        point_ears = [float(np.median(e)) for e in ears]
        self.data = {"kind": "ScreenRegression", "screen_size": [width, height], "fit_points": points,
                     "model": chosen["name"], "model_label": CANDIDATE_LABELS[chosen["name"]],
                     "features": chosen["features"], "mean": chosen["mean"].tolist(), "scale": chosen["scale"].tolist(),
                     "beta": chosen["beta"].tolist(), "ridge": chosen["ridge"], "loo_error_px": chosen["loo_error_px"],
                     "loo_point_errors_px": chosen["loo_point_errors_px"], "fit_error_px": chosen["fit_error_px"],
                     "candidates": {r["name"]: {"loo_error_px": r["loo_error_px"], "fit_error_px": r["fit_error_px"], "ridge": r["ridge"]}
                                    for r in results},
                     "samples_per_point": counts, "point_ear": point_ears, "ear_reference": min(point_ears)}

    def is_blink(self, ear):
        return bool(self.data and ear is not None and ear < .6 * self.data["ear_reference"])

    def predict(self, vector, enforce_domain=True, geometry=None):
        if self.model is None or vector is None:
            return None
        x = np.asarray(vector, dtype=float)
        if x.shape != (4,) or not np.isfinite(x).all():
            return None
        ear = (geometry or {}).get("ear")
        if self.is_blink(ear):
            return None
        ear = self.data["ear_reference"] if ear is None else ear
        model = self.model
        point = self._apply(self._matrix(model["features"], x, [ear], model["mean"], model["scale"]), model["beta"])[0]
        if not np.isfinite(point).all():
            return None
        if enforce_domain and np.any(np.abs(point - .5) > 1.5):
            return None
        return [float(v) for v in point]



# Superfícies que recebem um mapa: as duas telas e o robô NAO físico.
SURFACES = ("notebook", "tv", "robot")
CAMERAS = ("notebook", "nao", "validation")
SURFACE_NAMES = {"notebook": "notebook", "tv": "apresentação / TV", "robot": "robô NAO"}

# Robô visto de frente pelo participante, em coordenadas do desenho do painel
# (viewBox 320×380, normalizado 0–1). "right-*" é o lado direito DO ROBÔ, que
# fica à esquerda de quem o olha de frente.
ROBOT_SIZE = (320, 380)
ROBOT_PARTS = {"head": (.5, .23), "right-arm": (.25, .53), "torso": (.5, .5),
               "left-arm": (.75, .53), "right-leg": (.39, .77), "left-leg": (.61, .77)}
ROBOT_PART_NAMES = {"head": "cabeça", "right-arm": "braço direito do robô (à sua esquerda)",
                    "torso": "peito / tronco", "left-arm": "braço esquerdo do robô (à sua direita)",
                    "right-leg": "perna direita do robô (à sua esquerda)",
                    "left-leg": "perna esquerda do robô (à sua direita)"}
ROBOT_FIT = ("head", "right-arm", "torso", "left-arm", "left-leg", "right-leg")
# A conferência repete as partes em outra ordem; não treina o mapa.
ROBOT_CHECK = ("torso", "left-leg", "head", "right-arm", "right-leg", "left-arm")
# Uma estimativa só é atribuída a uma parte quando cai perto dela no desenho.
ROBOT_PART_RADIUS = .22
# Com 6 pontos, o modelo com curva teria tantos parâmetros quanto pontos.
# No robô a pose da cabeça entra como candidata: olhar o robô costuma virar a
# cabeça, e a íris sozinha mal muda entre as partes (P005: íris quase constante).
# Com 6 pontos, o modelo com curva teria tantos parâmetros quanto pontos.
ROBOT_CANDIDATES = CANDIDATES[:2] + (("cabeca", ("hx", "hy")), ("cabeca_palpebras", ("hx", "hy", "ear")),
                                     ("iris_cabeca", ("gx", "gy", "hx", "hy")))
# A parte só é mostrada quando a validação daquela câmera acertou ao menos metade.
ROBOT_PART_MIN_ACCURACY = .5
# Zona: como ficam íris, cabeça e pálpebras olhando cada superfície. Mínimos de
# dispersão evitam rejeitar pequenos ajustes de postura.
ZONE_FLOORS = np.array([.012, .012, .03, .012, .02])
ZONE_QUANTILE, ZONE_MARGIN = 97.5, 1.3


def zone_vector(observation, center=None):
    features = screen_vector(observation)
    if features is None:
        return None
    ear = observation.get("ear")
    if ear is None:
        ear = center[4] if center is not None else DEFAULT_EAR
    x = np.asarray(list(features)[:4] + [ear], dtype=float)
    return x if x.shape == (5,) and np.isfinite(x).all() else None


def fit_zone(rows):
    """Região de leituras de uma superfície, por distância de Mahalanobis com encolhimento.

    Em dados reais (P005, câmera do notebook) a íris quase não difere entre olhar o
    robô e olhar a tela; a pose da cabeça separa ~99%. A zona usa as duas.
    """
    groups = []
    for step in sorted({r["step"] for r in rows}):
        x = np.asarray([list(r["features"])[:4] + [r.get("ear") if r.get("ear") is not None else DEFAULT_EAR]
                        for r in rows if r["step"] == step], dtype=float)
        x = x[np.isfinite(x).all(axis=1)]
        if not len(x):
            continue
        median = np.median(x, axis=0)
        spread = np.maximum(1.4826 * np.median(np.abs(x - median), axis=0), ZONE_FLOORS / 3)
        groups.append(x[(np.abs(x - median) <= 3.5 * spread).all(axis=1)])
    x = np.vstack(groups) if groups else np.empty((0, 5))
    if len(x) < 20:
        return None
    center = np.median(x, axis=0)
    scale = np.maximum(x.std(axis=0), ZONE_FLOORS)
    z = (x - center) / scale
    inverse = np.linalg.inv(.7 * np.cov(z.T) + .3 * np.eye(5))
    distances = np.sqrt(np.einsum("ij,jk,ik->i", z, inverse, z))
    return {"center": center.tolist(), "scale": scale.tolist(), "inverse": inverse.tolist(),
            "threshold": float(np.percentile(distances, ZONE_QUANTILE) * ZONE_MARGIN)}


def zone_ratio(zone, observation):
    """Distância à zona dividida pelo limiar: ≤ 1 está dentro. None sem zona/leitura."""
    if not zone:
        return None
    center = np.asarray(zone["center"])
    x = zone_vector(observation, center)
    if x is None:
        return None
    z = (x - center) / np.asarray(zone["scale"])
    return float(np.sqrt(z @ np.asarray(zone["inverse"]) @ z) / zone["threshold"])
# Uma tela "contém" o olhar com uma pequena folga para o erro nas bordas.
SCREEN_MARGIN = .06


def model_key(surface, role):
    return f"{surface}:{role}"


def robot_part(point):
    if point is None:
        return None
    name, distance = min(((n, float(np.hypot(point[0] - c[0], point[1] - c[1]))) for n, c in ROBOT_PARTS.items()),
                         key=lambda item: item[1])
    return name if distance <= ROBOT_PART_RADIUS else None


def inside_screen(point, margin=SCREEN_MARGIN):
    return point is not None and all(-margin <= v <= 1 + margin for v in point)


def precision_level(report):
    """Nível honesto: "alta" segue os limites originais; "moderada" serve para regiões grandes."""
    if report.get("surface") == "robot":
        accuracy, mean = report["part_accuracy"], report["mean_error"]
        return "alta" if accuracy >= 5 / 6 and mean <= .12 else "moderada" if accuracy >= .5 else "baixa"
    if (report["mean_error"] <= .09 and report["max_error"] <= .14 and report["p90_error"] <= .16
            and report["improvement_over_center"] >= .5):
        return "alta"
    if report["mean_error"] <= .16 and report["p90_error"] <= .30 and report["improvement_over_center"] >= .35:
        return "moderada"
    return "baixa"


def participant_file(participant):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", participant).strip("_")[:40] or "participante"
    return f"{slug}-{hashlib.sha1(participant.encode('utf-8')).hexdigest()[:8]}.json"


class ScreenGaze:
    """Calibrações de posição por participante, superfície (tela/robô) e câmera.

    Cada câmera tem o próprio mapa. Os mapas ficam salvos por participante e
    voltam após reiniciar o servidor ou reaplicar as mesmas câmeras.
    """

    def __init__(self, store=None):
        self.store = Path(store) if store else None
        self.library = {}
        self.participant = None
        self.calibration = None
        self.last_sequence = None
        self.load()

    @property
    def models(self):
        return self.library.get(self.participant, {})

    def models_for(self, participant=None):
        return self.library.get(participant or self.participant, {})

    def participants(self):
        return sorted(self.library)

    def select(self, participant):
        if participant:
            self.participant = participant

    @property
    def active(self):
        return bool(self.calibration and self.calibration["phase"] not in ("complete", "failed", "cancelled"))

    def reset(self):
        """Interrompe a calibração em curso; a biblioteca salva continua disponível."""
        self.calibration = None

    # ---------- persistência ----------
    def load(self):
        if not self.store or not self.store.exists():
            return
        for path in sorted(self.store.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                models = {}
                for key, item in data["models"].items():
                    models[key] = {**item, "map": ScreenMap.from_data(item["map"])}
                self.library[data["participant"]] = models
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def save(self, participant):
        if not self.store:
            return
        self.store.mkdir(parents=True, exist_ok=True)
        path = self.store / participant_file(participant)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"participant": participant, "saved_utc": datetime.now(timezone.utc).isoformat(),
                                         "models": self.export(participant)}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        temporary.replace(path)

    # ---------- calibração ----------
    @staticmethod
    def _plan(surface):
        if surface == "robot":
            return ([ROBOT_PARTS[p] for p in ROBOT_FIT], [ROBOT_PARTS[p] for p in ROBOT_CHECK],
                    list(ROBOT_FIT) + list(ROBOT_CHECK))
        return list(FIT_POINTS), list(CHECK_POINTS), None

    def start(self, surface, role, participant, source, include_check=False, screen=None):
        screen = screen or surface
        if surface not in SURFACES or role not in CAMERAS or screen not in ("notebook", "tv"):
            raise ValueError("Tela/câmera inválida")
        if surface != "robot" and screen != surface:
            raise ValueError("Calibre cada tela na própria janela")
        if not participant or len(participant) > 80:
            raise ValueError("Informe o participante")
        if not source:
            raise ValueError("Selecione a câmera no painel antes de calibrar posição")
        if self.active:
            raise ValueError("Já há uma calibração de posição em andamento")
        fit_points, check_points, parts = self._plan(surface)
        self.participant = participant
        self.calibration = {"id": uuid.uuid4().hex, "surface": surface, "screen": screen, "role": role,
                            "participant": participant, "source": source, "index": 0, "phase": "waiting",
                            "started": None, "samples": [], "checks": [], "map": ScreenMap(), "attempt": 0,
                            "fit_points": fit_points, "check_points": check_points, "parts": parts,
                            "layout": None, "client": None, "pose": None,
                            "reason": "Abra esta tela, ative a visualização e acompanhe as indicações.",
                            "validation": None, "include_check": include_check}
        self.last_sequence = None

    def begin_check(self, surface, role, participant=None, screen=None):
        if self.active:
            raise ValueError("Conclua a calibração atual antes de medir precisão")
        participant = participant or self.participant
        model = self.models_for(participant).get(model_key(surface, role))
        if not model:
            raise ValueError("Calibre esta câmera primeiro")
        fit_points, check_points, parts = self._plan(surface)
        self.participant = participant
        # A janela pode ser outra (nova aba), desde que no mesmo tamanho/monitor.
        self.calibration = {**model, "id": uuid.uuid4().hex, "surface": surface, "screen": screen or model.get("screen") or surface,
                            "participant": participant, "index": len(fit_points), "phase": "waiting", "started": None,
                            "samples": [], "checks": [], "attempt": 0, "include_check": True, "client": None,
                            "fit_points": fit_points, "check_points": check_points, "parts": parts,
                            "reason": "Olhe para cada indicação até ela mudar", "validation": None}
        self.last_sequence = None

    def cue(self, now=None):
        now = time.perf_counter() if now is None else now
        cal = self.calibration
        if not cal:
            return None
        active = cal["phase"] not in ("complete", "failed", "cancelled")
        index, n_fit = cal["index"], len(cal["fit_points"])
        points = cal["fit_points"] + cal["check_points"]
        rows = cal["samples"] if index < n_fit else cal["checks"]
        part = cal["parts"][index] if active and cal["parts"] and index < len(cal["parts"]) else None
        return {"active": active, "id": cal["id"], "display": cal["screen"], "screen": cal["screen"],
                "surface": cal["surface"], "role": cal["role"], "participant": cal["participant"],
                "cue_id": f"screen:{cal['id']}:{index}:{cal['attempt']}",
                "marker": points[index] if active and index < len(points) else None,
                "part": part, "part_label": ROBOT_PART_NAMES.get(part),
                "index": index, "total": n_fit + len(cal["check_points"]) if cal["include_check"] else n_fit,
                "fit_total": n_fit, "check_total": len(cal["check_points"]),
                "mode": "check" if index >= n_fit else "fit",
                "phase": cal["phase"], "reason": cal["reason"], "validation": cal["validation"],
                "valid_samples": sum(r["step"] == index for r in rows), "minimum_samples": MIN_SAMPLES,
                "remaining": max(0, 3 - (now - cal["started"])) if cal["started"] is not None else 3}

    def cancel(self, display):
        if self.calibration and self.calibration["screen"] == display and self.active:
            self.calibration.update(phase="cancelled", started=None, reason="Calibração de posição cancelada")

    def _interrupt(self, cal, reason):
        if cal["started"] is not None:
            cal["samples"][:] = [r for r in cal["samples"] if r["step"] != cal["index"]]
            cal["checks"][:] = [r for r in cal["checks"] if r["step"] != cal["index"]]
            cal["attempt"] += 1
        cal.update(started=None, phase="waiting", reason=reason)

    def tick(self, now, observations, labels, displays):
        cal = self.calibration
        if not cal or cal["phase"] in ("complete", "failed", "cancelled"):
            return
        if labels.get(cal["role"]) != cal["source"]:
            cal.update(phase="failed", reason="A câmera mudou. Refaça a calibração de posição.", started=None)
            return
        cue_id = self.cue(now)["cue_id"]
        views = [(client, d) for client, d in displays.items() if d["display"] == cal["screen"]
                 and d["visible"] and d["ready"] and d["cue_id"] == cue_id and now - d["seen"] < 1]
        # Só a janela que começou conta; outra aba da mesma tela não derruba a coleta.
        own = [v for v in views if cal["client"] in (None, v[0])]
        if views and not own:
            self._interrupt(cal, "Outra janela de alvo está aberta para esta tela. Use só a janela que iniciou a calibração.")
            return
        ready = bool(own)
        if ready:
            client, view = max(own, key=lambda item: item[1]["seen"])
            if cal["surface"] == "robot":
                cal["client"] = client
            else:
                if not view.get("layout"):
                    cal.update(phase="waiting", reason="Atualize esta página para habilitar o mapa de posição.")
                    return
                if cal["layout"] is None:
                    cal.update(layout=view["layout"], client=client)
                elif not same_layout(cal["layout"], view["layout"]):
                    if cal["index"] == 0:
                        # A entrada em tela cheia pode terminar após o primeiro heartbeat.
                        # Reiniciar só o primeiro ponto, antes de existir um ajuste concluído.
                        cal.update(layout=view["layout"], client=client, started=None, phase="waiting",
                                   attempt=cal["attempt"] + 1, reason="Tela estabilizada; preparando o primeiro ponto")
                        cal["samples"].clear()
                        return
                    # Antes, isso encerrava a calibração no ponto 8 ou 9. Agora só espera.
                    self._interrupt(cal, "A janela mudou de tamanho ou de monitor. Volte à tela cheia no mesmo monitor para continuar.")
                    return
                cal["client"] = client
        obs = observations.get(cal["role"])
        fresh = bool(obs and now - obs["capture"] <= .5)
        if not ready or not fresh:
            self._interrupt(cal, "Mantenha a tela visível e a câmera com leitura recente.")
            return
        if cal["started"] is None:
            if screen_vector(obs) is None or obs.get("uncertainty", 1) > .15:
                cal.update(phase="waiting", reason=screen_reason(obs) or "Aguardando leitura válida dos olhos antes de iniciar o ponto")
                return
            cal.update(started=now, phase="collecting",
                       reason="Olhe para a parte indicada do robô" if cal["surface"] == "robot" else "Fixe o olhar no ponto amarelo")
            return
        elapsed = now - cal["started"]
        sequence = obs.get("sequence", obs["capture"])
        # No robô o olhar sai da tela e procura a parte: mais tempo de acomodação.
        settling = .9 if cal["surface"] == "robot" else .6
        n_fit = len(cal["fit_points"])
        if (settling <= elapsed <= 10 and cal["started"] + settling <= obs["capture"] <= cal["started"] + 10
                and screen_vector(obs) is not None and obs.get("uncertainty", 1) <= .15
                and eyes_open(obs) and sequence != self.last_sequence):
            self.last_sequence = sequence
            row = {"step": cal["index"], "features": list(screen_vector(obs)), "pose": obs.get("pose"),
                   **{k: obs.get(k) for k in ("capture", "sequence", "glabella", "iris_diameter", "ear")}}
            (cal["samples"] if cal["index"] < n_fit else cal["checks"]).append(row)
        cal["reason"] = (screen_reason(obs) or "Sem leitura válida dos olhos" if screen_vector(obs) is None
                         else "Incerteza de sincronização alta" if obs.get("uncertainty", 1) > .15
                         else "Olhos fechados ou piscando; mantenha os olhos abertos" if not eyes_open(obs)
                         else "Leitura válida; mantenha o olhar na indicação")
        if elapsed < 3 + (settling - .6):
            return
        rows = cal["samples"] if cal["index"] < n_fit else cal["checks"]
        if sum(r["step"] == cal["index"] for r in rows) < MIN_SAMPLES:
            if elapsed < 10:
                return
            cal.update(phase="failed", started=None, reason="Poucas leituras neste ponto. " + cal["reason"] + ". Ajuste o enquadramento e tente novamente.")
            return
        if cal["index"] == n_fit - 1:
            try:
                robot = cal["surface"] == "robot"
                cal["map"].fit(cal["samples"], list(ROBOT_SIZE) if robot else cal["layout"],
                               points=cal["fit_points"], candidates=ROBOT_CANDIDATES if robot else CANDIDATES)
                poses = [r["pose"] for r in cal["samples"] if r.get("pose") is not None]
                if poses and float(np.max(np.ptp(poses, axis=0))) > .10:
                    raise ValueError("A cabeça do NAO moveu durante os pontos. Mantenha a câmera fixa e repita.")
                cal["pose"] = np.median(poses, axis=0).tolist() if poses else None
                cal["zone"] = fit_zone(cal["samples"])
            except ValueError as error:
                cal.update(phase="failed", reason=str(error), started=None)
                return
        cal["index"] += 1
        cal.update(started=None, phase="waiting", reason="Preparando a próxima indicação")
        if cal["index"] == n_fit and not cal["include_check"]:
            cal["validation"] = {"accepted": None, "level": None, "status": "not_measured"}
            self.store_model(cal)
            cal.update(phase="complete", reason=self._fit_message(cal) + " Use Validar para conferir.")
            return
        if cal["index"] < n_fit + len(cal["check_points"]):
            return
        report = self._evaluate(cal)
        if report is None:
            cal.update(phase="failed", started=None, reason="Validação sem leituras válidas numa indicação. Repita.")
            return
        cal["validation"] = report
        self.store_model(cal)
        cal.update(phase="complete", reason=f"{self._fit_message(cal)} Validação: {self.validation_text(report)}.")

    @staticmethod
    def _fit_message(cal):
        data = cal["map"].data
        if cal["surface"] == "robot":
            return (f"Robô calibrado · modelo {data['model_label']} · erro estimado "
                    f"{100 * data['loo_error_px'] / ROBOT_SIZE[1]:.0f}% da altura do robô.")
        return f"Calibração concluída · modelo {data['model_label']} · erro estimado {data['loo_error_px']:.0f} px."

    @staticmethod
    def validation_text(report):
        if not report or report.get("level") is None:
            return "não medida"
        label = {"alta": "precisão alta", "moderada": "precisão moderada (regiões grandes)", "baixa": "precisão baixa"}[report["level"]]
        if report.get("surface") == "robot":
            return f"{label} · {report['parts_correct']}/{report['parts_total']} partes certas"
        return f"{label} · erro médio {report['mean_error_px']:.0f} px"

    def _evaluate(self, cal):
        n_fit = len(cal["fit_points"])
        robot = cal["surface"] == "robot"
        size = np.asarray(ROBOT_SIZE if robot else cal["layout"][:2], dtype=float)
        errors, all_errors, baseline_errors, pixel_errors, parts = [], [], [], [], []
        for i, point in enumerate(cal["check_points"], n_fit):
            rows = [r for r in cal["checks"] if r["step"] == i]
            predictions = [p for p in (cal["map"].predict(r["features"], False, r) for r in rows) if p is not None]
            if not predictions:
                return None
            prediction = np.median(predictions, axis=0)
            errors.append(float(np.linalg.norm(prediction - point)))
            pixel_errors.append(float(np.linalg.norm((prediction - point) * size)))
            all_errors.extend(float(np.linalg.norm(np.asarray(p) - point)) for p in predictions)
            baseline_errors.append(float(np.linalg.norm(np.asarray([.5, .5]) - point)))
            if robot:
                expected = cal["parts"][i]
                parts.append({"expected": expected, "predicted": robot_part(prediction.tolist()),
                              "point": prediction.tolist()})
        baseline = float(np.mean(baseline_errors))
        report = {"surface": cal["surface"], "mean_error": float(np.mean(errors)), "max_error": max(errors),
                  "point_errors": errors, "p90_error": float(np.percentile(all_errors, 90)),
                  "center_baseline_error": baseline, "improvement_over_center": 1 - float(np.mean(errors)) / baseline,
                  "mean_error_px": float(np.mean(pixel_errors)), "point_errors_px": pixel_errors,
                  "limits": {"mean": .09, "maximum": .14, "p90": .16, "improvement_over_center": .5},
                  "unit": "normalized_screen_distance"}
        if robot:
            correct = sum(p["expected"] == p["predicted"] for p in parts)
            report.update(parts=parts, parts_correct=correct, parts_total=len(parts),
                          part_accuracy=correct / len(parts), unit="normalized_robot_drawing")
        report["level"] = precision_level(report)
        report["accepted"] = report["level"] == "alta"
        return report

    def store_model(self, cal):
        model = {k: cal.get(k) for k in ("map", "role", "surface", "screen", "source", "participant", "validation",
                                         "layout", "client", "pose", "zone")}
        model["calibration_id"] = cal["id"]
        self.library.setdefault(cal["participant"], {})[model_key(cal["surface"], cal["role"])] = model
        self.participant = cal["participant"]
        try:
            self.save(cal["participant"])
        except OSError:
            pass

    # ---------- estimativa ----------
    def position(self, key, now, observations, labels, displays, require_view=True, participant=None):
        """Posição estimada de uma câmera numa superfície. Com require_view=False (painel
        do operador durante o estudo) não exige a página de alvo aberta."""
        model = self.models_for(participant).get(key)
        hidden = lambda reason: {"visible": False, "reason": reason, "estimated": True}
        cal = self.calibration
        if self.active and model_key(cal["surface"], cal["role"]) == key:
            return hidden("Calibração desta câmera em andamento")
        if not model:
            return hidden("Calibre esta câmera para mostrar o alvo do olhar")
        if labels.get(model["role"]) != model["source"]:
            return hidden("A câmera deste mapa não está atribuída a esse papel; reaplique as mesmas câmeras ou recalibre")
        if require_view and model["surface"] != "robot":
            if not any(v["display"] == model["surface"] and v["visible"] and now - v["seen"] <= 1
                       and same_layout(v.get("layout"), model["layout"]) for v in displays.values()):
                return hidden("Ative a visualização numa janela do mesmo tamanho/monitor da calibração")
        obs = observations.get(model["role"])
        if not obs or now - obs["capture"] > .6 or screen_vector(obs) is None or obs.get("uncertainty", 1) > .15:
            return hidden(screen_reason(obs) if obs and screen_reason(obs) else "Sem leitura recente e confiável dos olhos")
        if model.get("pose") is not None:
            pose = obs.get("pose")
            if pose is None or max(abs(a - b) for a, b in zip(pose, model["pose"])) > .08:
                return hidden("A pose da câmera do NAO mudou; refaça o mapa")
        if model["map"].is_blink(obs.get("ear")):
            return hidden("Piscada ou olhos semicerrados")
        zone = self.zone_decision(key, model, obs, participant)
        point = model["map"].predict(screen_vector(obs), geometry=obs)
        if point is None:
            # Leitura válida, mas muito longe desta superfície: conta como "fora dela".
            far = model["map"].predict(screen_vector(obs), False, obs) is not None
            return {**hidden("Olhar bem fora desta área calibrada" if far else "Leitura fora da postura calibrada"),
                    "outside": far, **zone}
        validation = model["validation"] or {}
        result = {"visible": True, "x": point[0], "y": point[1], "estimated": True,
                  "age": now - obs["capture"], "sequence": obs.get("sequence"), "role": model["role"],
                  "surface": model["surface"], "validation": validation, "level": validation.get("level"),
                  "verified": validation.get("accepted") is True,
                  "reason": "Posição aproximada; não confirma foco/atenção", **zone}
        if model["surface"] == "robot":
            accuracy = validation.get("part_accuracy")
            reliable = accuracy is None or accuracy >= ROBOT_PART_MIN_ACCURACY
            # Sem zona (mapa antigo), vale a regra anterior: perto de alguma parte.
            inside = zone["in_zone"] if zone["zone_ratio"] is not None else robot_part(point) is not None
            part = robot_part(point) if inside and reliable else None
            result.update(part=part, part_label=ROBOT_PART_NAMES.get(part), inside=inside, part_reliable=reliable,
                          part_accuracy=accuracy)
        else:
            result["inside"] = zone["in_zone"] and inside_screen(point, .15 if zone["zone_ratio"] is not None else SCREEN_MARGIN)
        return result

    def zone_decision(self, key, model, observation, participant=None):
        """A leitura pertence a esta superfície? Compara com as zonas dos outros mapas da mesma câmera."""
        ratio = zone_ratio(model.get("zone"), observation)
        competitor, best = None, None
        for other_key, other in self.models_for(participant).items():
            if other_key == key or other["role"] != model["role"]:
                continue
            other_ratio = zone_ratio(other.get("zone"), observation)
            if other_ratio is not None and (best is None or other_ratio < best):
                competitor, best = other["surface"], other_ratio
        in_zone = ratio is None or ratio <= 1
        closer = None
        if ratio is not None and best is not None and best <= 1 and best < ratio:
            in_zone, closer = False, competitor
        return {"zone_ratio": ratio, "in_zone": in_zone, "closer_to": closer}

    def checking_position(self, display, now, observations):
        cal = self.calibration
        if not self.active or cal["screen"] != display or cal["index"] < len(cal["fit_points"]):
            return {"visible": False}
        obs = observations.get(cal["role"])
        if not obs or now - obs["capture"] > .6 or screen_vector(obs) is None or obs.get("uncertainty", 1) > .15:
            return {"visible": False}
        point = cal["map"].predict(screen_vector(obs), False, obs)
        if point is None:
            return {"visible": False}
        return {"visible": True, "x": point[0], "y": point[1], "age": now - obs["capture"], "checking": True}

    def summary(self, key, participant=None):
        model = self.models_for(participant).get(key)
        if not model:
            return {"calibrated": False}
        data = model["map"].data
        robot = model["surface"] == "robot"
        return {"calibrated": True, "key": key, "role": model["role"], "surface": model["surface"],
                "participant": model["participant"], "layout": model["layout"][:2] if model.get("layout") else list(ROBOT_SIZE),
                "model": data["model_label"], "loo_error_px": data["loo_error_px"],
                "loo_error_robot_pct": 100 * data["loo_error_px"] / ROBOT_SIZE[1] if robot else None,
                "validation": model["validation"], "validation_text": self.validation_text(model["validation"])}

    def export(self, participant=None):
        return {key: {**{k: v for k, v in model.items() if k != "map"}, "map": model["map"].data}
                for key, model in self.models_for(participant).items()}

    def diagnostic(self):
        cal = self.calibration
        if not cal:
            return None
        return {**{k: v for k, v in cal.items() if k != "map"}, "map": cal["map"].data,
                "display": cal["screen"]}
