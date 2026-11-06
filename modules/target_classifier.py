"""Classificação empírica por alvo. O escore não é probabilidade de atenção."""
from __future__ import annotations

import math
import numpy as np

TARGETS = ("NOTEBOOK", "TV", "NAO", "OUTRO_LOCAL")
UNKNOWN = "INDETERMINADO"
ROLES = ("notebook", "nao", "validation")
REGIONS = {"screens": {"NOTEBOOK", "TV"}, "robot": {"NAO"}, "task": {"NOTEBOOK", "TV", "NAO"}}


def region_votes(prediction):
    if "regions" in prediction:
        return prediction["regions"]
    target = prediction.get("target", UNKNOWN)
    return {key: {"target": ("DENTRO" if target in labels else "FORA") if target in TARGETS else UNKNOWN,
                  "score": prediction.get("score", 0), "reason": prediction.get("reason", "")}
            for key, labels in REGIONS.items()}


def classify_regions(distances):
    """Margem contra a região oposta; ambiguidade entre TV/notebook não é rejeição da tarefa."""
    result = {}
    for key, labels in REGIONS.items():
        inside = min(d[0] for d in distances if d[1] in labels)
        outside = min(d[0] for d in distances if d[1] not in labels)
        best, other = min(inside, outside), max(inside, outside)
        margin = (other - best) / max(other, 1e-6)
        if best > 1 or margin < 0.18:
            result[key] = {"target": UNKNOWN, "score": 0.0, "reason": "região distante ou ambígua"}
        else:
            result[key] = {"target": "DENTRO" if inside < outside else "FORA",
                           "score": round(float((1 - best / 2) * margin), 3), "reason": "região calibrada"}
    return result


def feature_status(observation):
    if getattr(observation, "face_count", 1) > 1:
        return None, "Mais de um rosto na câmera; deixe somente o participante no enquadramento"
    values = [observation.gaze_feature_x, observation.gaze_feature_y,
              observation.head_x, observation.head_y]
    if not observation.detected:
        return None, "Rosto/olhos não detectados"
    if any(v is None or not math.isfinite(v) for v in values):
        return None, "Marcos dos olhos/cabeça incompletos"
    if observation.ear is None or observation.ear < 0.16:
        return None, "Pálpebras sem leitura confiável (olhos fechados ou ângulo baixo)"
    if observation.iris_diameter is None or observation.iris_diameter < 2:
        return None, "Íris pequena ou sem leitura: ajuste distância/enquadramento"
    if not all(-0.5 <= v <= 1.5 for v in values):
        return None, "Características do olhar fora da faixa confiável"
    return values, None


def features(observation):
    return feature_status(observation)[0]


def screen_feature_status(observation):
    """Leitura para calibrar coordenadas, independente dos filtros do classificador de alvos."""
    if not observation.detected:
        return None, "Rosto/olhos não detectados; confira a prévia da webcam"
    values = [observation.gaze_feature_x, observation.gaze_feature_y, observation.head_x, observation.head_y]
    if any(v is None or not math.isfinite(v) for v in values):
        return None, "Marcos dos olhos incompletos"
    if observation.iris_diameter is None or observation.iris_diameter < 2:
        return None, "Olhos pequenos na imagem; aproxime o rosto da webcam"
    # Não reutilizar EAR 0,16 do classificador como condição da calibração da tela.
    if observation.ear is None or observation.ear < .08:
        return None, "Aguardando os olhos abertos"
    return values, None


class TargetClassifier:
    """Protótipos robustos com rejeição por distância e ambiguidade.

    Treinado por câmera e participante, sem inferência de estado cognitivo.
    A verificação independente é necessária antes de iniciar a sessão.
    """
    def __init__(self):
        self.models = {}

    def fit(self, samples):
        models = {}
        counts = {}
        for role in ROLES:
            grouped = {label: [s["features"] for s in samples
                               if s["role"] == role and s["target"] == label]
                       for label in TARGETS}
            counts[role] = {label: len(rows) for label, rows in grouped.items()}
            # Uma câmera sem exemplos de um alvo poderia confundi-lo com os demais.
            if any(len(rows) < 8 for rows in grouped.values()):
                continue
            all_rows = np.asarray([x for rows in grouped.values() for x in rows])
            scale = np.maximum(np.std(all_rows, axis=0), [0.06, 0.12, 0.06, 0.06])
            classes = {}
            for label, rows in grouped.items():
                # Preserva fixações separadas: "outro local" à esquerda e à
                # direita não deve gerar um protótipo médio no centro da TV.
                step_ids = {s["step"] for s in samples if s["role"] == role and s["target"] == label}
                prototypes = []
                for step in step_ids:
                    cluster = [s["features"] for s in samples
                               if s["role"] == role and s["target"] == label and s["step"] == step]
                    if len(cluster) < 4:
                        continue
                    center = np.median(cluster, axis=0)
                    residual = np.sqrt(np.mean(((np.asarray(cluster) - center) / scale) ** 2, axis=1))
                    radius = float(np.clip(np.percentile(residual, 95) * 1.5, 0.55, 2.0))
                    # Parte física da fixação (ex.: cabeça/pernas do NAO), para o painel.
                    part = next((s.get("part") for s in samples if s["role"] == role and s["target"] == label
                                 and s["step"] == step and s.get("part")), None)
                    prototypes.append({"center": center.tolist(), "radius": radius, "part": part})
                if not prototypes:
                    break
                classes[label] = {"prototypes": prototypes}
            if len(classes) != len(TARGETS):
                continue
            models[role] = {"scale": scale.tolist(), "classes": classes}
        if not models:
            raise ValueError("Nenhuma câmera tem 8 leituras válidas de cada alvo. Refaça as etapas com poucas amostras.")
        self.models = models
        return {"counts": counts, "usable_cameras": list(models),
                "warnings": [f"{r}: excluída da classificação por amostras insuficientes"
                             for r in ROLES if r not in models and any(counts[r].values())]}

    def predict(self, role, vector):
        if role not in self.models or vector is None:
            return {"target": UNKNOWN, "score": 0.0, "reason": "sem leitura/modelo válido"}
        model = self.models[role]
        distances = []
        for label, item in model["classes"].items():
            candidates = [(float(np.sqrt(np.mean(
                ((np.asarray(vector) - p["center"]) / model["scale"]) ** 2))), p["radius"], p.get("part"))
                for p in item["prototypes"]]
            distance, radius, part = min(candidates, key=lambda item: item[0] / item[1])
            distances.append((distance / radius, label, distance, radius, part))
        distances.sort(key=lambda item: item[0])
        regions = classify_regions(distances)
        _, label, distance, radius, part = distances[0]
        margin = (distances[1][0] - distances[0][0]) / max(distances[1][0], 1e-6)
        if distance > radius or margin < 0.18:
            return {"target": UNKNOWN, "score": 0.0, "reason": "distância/ambiguidade", "regions": regions}
        return {"target": label, "score": round(float((1 - distance / (2 * radius)) * margin), 3),
                "reason": "protótipo calibrado", "regions": regions, "part": part}


def _vote(predictions):
    valid = [p for p in predictions if p["target"] != UNKNOWN and p["score"] >= 0.35]
    if not valid:
        return {"target": UNKNOWN, "score": 0.0, "reason": "sem evidência suficiente"}
    if len({p["target"] for p in valid}) != 1:
        return {"target": UNKNOWN, "score": 0.0, "reason": "câmeras discordam"}
    # Uma só câmera é aceita apenas com escore mais alto.
    if len(valid) == 1 and valid[0]["score"] < 0.60:
        return {"target": UNKNOWN, "score": 0.0, "reason": "apenas uma câmera com escore baixo"}
    best = max(valid, key=lambda p: p["score"])
    return {"target": valid[0]["target"], "score": round(sum(p["score"] for p in valid) / len(valid), 3),
            "reason": "concordância" if len(valid) > 1 else "uma câmera disponível", "part": best.get("part")}


def fuse(predictions):
    result = _vote(predictions)
    votes = [region_votes(p) for p in predictions]
    result["regions"] = {key: _vote([p[key] for p in votes]) for key in REGIONS}
    return result
