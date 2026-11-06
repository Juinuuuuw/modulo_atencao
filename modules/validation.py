"""Comparação temporal com anotação humana, incluindo abstenções e lacunas."""
from __future__ import annotations

import math
from .target_classifier import TARGETS, UNKNOWN


def validate_annotations(intervals, duration):
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Duração inválida")
    cleaned = []
    for row in intervals:
        start, end = float(row["start"]), float(row["end"])
        label = row["target"]
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration + 0.001:
            raise ValueError("Intervalo deve estar dentro da sessão, com início menor que fim")
        if label not in (*TARGETS, UNKNOWN):
            raise ValueError("Alvo inválido")
        cleaned.append({"start": start, "end": end, "target": label})
    cleaned.sort(key=lambda row: row["start"])
    if any(a["end"] > b["start"] for a, b in zip(cleaned, cleaned[1:])):
        raise ValueError("Anotações do mesmo avaliador não podem se sobrepor")
    return cleaned


def summarize(matrix):
    labels = (*TARGETS, UNKNOWN)
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in labels)
    covered = sum(matrix[truth][pred] for truth in labels for pred in TARGETS)
    per_target = {}
    for label in TARGETS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_target[label] = {"precision": precision, "recall": recall,
                             "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                             "support": support}
    expected = sum(sum(matrix[label].values()) * sum(matrix[t][label] for t in labels)
                   for label in labels) / total ** 2 if total else 0
    kappa = ((correct / total - expected) / (1 - expected)
             if total and expected < 1 - 1e-12 else None)
    supported = [v["f1"] for v in per_target.values() if v["support"] > 0]
    return {"matrix": matrix, "total": total, "accuracy": correct / total if total else None,
            "coverage": covered / total if total else None,
            "selective_accuracy": correct / covered if covered else None,
            "macro_f1": sum(supported) / len(supported) if supported else None,
            "kappa": kappa, "per_target": per_target}


def empty_matrix():
    return {truth: {pred: 0.0 for pred in (*TARGETS, UNKNOWN)} for truth in (*TARGETS, UNKNOWN)}


def evaluate_intervals(predictions, annotations, duration, max_gap=0.5):
    """Matriz ponderada em segundos. Lacunas > max_gap viram indeterminado.

    Anotações humanas ambíguas são excluídas das métricas, mas contabilizadas.
    Não propaga a última previsão indefinidamente durante uma desconexão.
    """
    intervals = validate_annotations(annotations, duration)
    rows = sorted(predictions, key=lambda row: row["t"])
    matrix = empty_matrix()
    ambiguous = 0.0
    for interval in intervals:
        start, end, truth = interval["start"], interval["end"], interval["target"]
        if truth == UNKNOWN:
            ambiguous += end - start
            continue
        accounted = 0.0
        for index, row in enumerate(rows):
            next_t = rows[index + 1]["t"] if index + 1 < len(rows) else duration
            left, right = max(start, row["t"]), min(end, next_t, row["t"] + max_gap)
            if right <= left:
                continue
            weight = right - left
            label = row["target"] if row["target"] in TARGETS else UNKNOWN
            matrix[truth][label] += weight
            accounted += weight
        matrix[truth][UNKNOWN] += max(0.0, end - start - accounted)
    result = summarize(matrix)
    result.update({"unit": "seconds", "ambiguous_seconds": ambiguous,
                   "annotated_seconds": sum(r["end"] - r["start"] for r in intervals),
                   "session_seconds": duration,
                   "annotation_coverage": sum(r["end"] - r["start"] for r in intervals) / duration if duration else 0})
    return result


def evaluate_checks(rows):
    matrix = empty_matrix()
    for row in rows:
        matrix[row["expected"]][row["target"]] += 1
    result = summarize(matrix)
    result["unit"] = "samples"
    task_matrix = {truth: {pred: 0 for pred in ("DENTRO", "FORA", UNKNOWN)} for truth in ("DENTRO", "FORA")}
    for row in rows:
        truth = "FORA" if row["expected"] == "OUTRO_LOCAL" else "DENTRO"
        predicted = row.get("regions", {}).get("task", {}).get("target")
        if predicted is None:
            predicted = UNKNOWN if row["target"] == UNKNOWN else ("FORA" if row["target"] == "OUTRO_LOCAL" else "DENTRO")
        task_matrix[truth][predicted] += 1
    total = len(rows)
    task_correct = task_matrix["DENTRO"]["DENTRO"] + task_matrix["FORA"]["FORA"]
    known = total - task_matrix["DENTRO"][UNKNOWN] - task_matrix["FORA"][UNKNOWN]
    result["task_metrics"] = {"matrix": task_matrix, "total": total,
                               "accuracy": task_correct / total if total else None,
                               "coverage": known / total if total else None}
    return result
